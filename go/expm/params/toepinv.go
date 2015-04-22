package main

import (
	"fmt"
	"image"
	"log"
	"math"

	"github.com/gonum/floats"
	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-cv/slide"
	"github.com/jvlmdr/go-fftw/fftw"
	"github.com/jvlmdr/lin-go/clap"
	"github.com/jvlmdr/lin-go/cmat"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/jvlmdr/shift-invar/go/toepcov"
	"github.com/nfnt/resize"
)

type ToepInvTrainer struct {
	Lambda float64
	Len    int
	Sigma  float64
	Crop   int
}

func (t *ToepInvTrainer) Field(name string) string {
	switch name {
	case "Lambda":
		return fmt.Sprint(t.Lambda)
	case "Len":
		return fmt.Sprint(t.Len)
	case "Sigma":
		return fmt.Sprint(t.Sigma)
	case "Crop":
		return fmt.Sprint(t.Crop)
	default:
		return ""
	}
}

// ToepInvTrainerSet specifies a set of ToepInvTrainers.
type ToepInvTrainerSet struct {
	Lambda []float64
	Len    []int
	Sigma  []float64
	Crop   []int
}

func (set *ToepInvTrainerSet) Fields() []string {
	return []string{"Lambda", "Len", "Sigma", "Crop"}
}

func (set *ToepInvTrainerSet) Enumerate() []Trainer {
	var ts []Trainer
	for _, lambda := range set.Lambda {
		for _, n := range set.Len {
			for _, sigma := range set.Sigma {
				for _, crop := range set.Crop {
					t := &ToepInvTrainer{Lambda: lambda, Len: n, Sigma: sigma, Crop: crop}
					ts = append(ts, t)
				}
			}
		}
	}
	return ts
}

func (t *ToepInvTrainer) Train(posIms, negIms []string, dataset data.ImageSet, phi feat.Image, covarFile string, region detect.PadRect, exampleOpts data.ExampleOpts, flip bool, interp resize.InterpolationFunction, searchOpts detect.MultiScaleOpts) (*detect.FeatTmpl, error) {
	posRects, err := data.PosExampleRects(posIms, dataset, searchOpts.Pad.Margin, region, exampleOpts)
	if err != nil {
		return nil, err
	}
	// Extract positive examples.
	pos, err := data.Examples(posIms, posRects, dataset, phi, searchOpts.Pad.Extend, region, flip, interp)
	if err != nil {
		return nil, err
	}
	if len(pos) == 0 {
		return nil, fmt.Errorf("empty positive set")
	}

	// Compute mean of positive examples.
	featsize, channels := phi.Size(region.Size), phi.Channels()
	meanPos := rimg64.NewMulti(featsize.X, featsize.Y, channels)
	for _, x := range pos {
		floats.Add(meanPos.Elems, x.Elems)
	}
	floats.Scale(1/float64(len(pos)), meanPos.Elems)

	// Load covariance from file.
	total, err := toepcov.LoadTotalExt(covarFile)
	if err != nil {
		return nil, err
	}
	// Take subset with enough elements.
	var minReqCount int64 = 10000
	var b int
	for ; b <= total.Count.Band; b++ {
		minCount := total.Count.At(b, b)
		minCount = min64(minCount, total.Count.At(b, -b))
		minCount = min64(minCount, total.Count.At(-b, b))
		minCount = min64(minCount, total.Count.At(-b, -b))
		if minCount < minReqCount {
			break
		}
	}
	maxBand := b - 1
	log.Println("max bandwidth to have min count:", maxBand)

	// Obtain covariance and mean from sums.
	distr := toepcov.Normalize(total, true)
	covar := distr.Covar.CloneBandwidth(maxBand)
	for u := -maxBand; u <= maxBand; u++ {
		for v := -maxBand; v <= maxBand; v++ {
			x, y := float64(u), float64(v)
			alpha := math.Exp(-(x*x + y*y) / (2 * t.Sigma * t.Sigma))
			for p := 0; p < covar.Channels; p++ {
				for q := 0; q < covar.Channels; q++ {
					covar.Set(u, v, p, q, alpha*covar.At(u, v, p, q))
				}
			}
		}
	}

	// Obtain approximate inverse.
	size := image.Pt(t.Len, t.Len)
	log.Printf("size %v, lambda %v, sigma %v, crop %v", size, t.Lambda, t.Sigma, t.Crop)
	prec, err := approxInverse(covar, size, covar.Bandwidth, t.Lambda)
	if err != nil {
		return nil, err
	}

	// Subtract negative mean from positive example.
	delta := toepcov.SubMean(meanPos, distr.Mean)
	weights := toepcov.MulFFT(prec, delta)

	// Set boundary to zero.
	interior := image.Rect(0, 0, weights.Width, weights.Height).Inset(t.Crop)
	for u := 0; u < weights.Width; u++ {
		for v := 0; v < weights.Height; v++ {
			if image.Pt(u, v).In(interior) {
				continue
			}
			for p := 0; p < weights.Channels; p++ {
				weights.Set(u, v, p, 0)
			}
		}
	}

	// Pack weights into image in detection template.
	tmpl := &detect.FeatTmpl{
		Scorer:     &slide.AffineScorer{Tmpl: weights},
		PixelShape: region,
	}
	return tmpl, nil
}

func approxInverse(cov *toepcov.Covar, size image.Point, bandOut int, lambda float64) (*toepcov.Covar, error) {
	bandIn := cov.Bandwidth
	if 2*bandIn+1 > size.X || 2*bandIn+1 > size.Y {
		// TODO: Handle this better.
		panic("input bandwidth greater than size")
	}
	if 2*bandOut+1 > size.X || 2*bandOut+1 > size.Y {
		// TODO: Handle this better.
		panic("output bandwidth greater than size")
	}
	num := float64(size.X * size.Y)
	// Take Fourier transform of each channel pair.
	a := make([][]*fftw.Array2, cov.Channels)
	for p := range a {
		a[p] = make([]*fftw.Array2, cov.Channels)
		for q := range a[p] {
			apq := fftw.NewArray2(size.X, size.Y)
			a[p][q] = apq
			for u := -bandIn; u <= bandIn; u++ {
				for v := -bandIn; v <= bandIn; v++ {
					umod, vmod := mod(u, size.X), mod(v, size.Y)
					apq.Set(umod, vmod, apq.At(umod, vmod)+complex(cov.At(u, v, p, q), 0))
				}
			}
			//	if p == q {
			//		apq.Set(0, 0, apq.At(0, 0)+complex(lambda, 0))
			//	}
			fftw.FFT2To(apq, apq)
		}
	}
	// Take inverse of each channels x channels block.
	// TODO: Re-use other memory so that this is not futile?
	auv := cmat.New(cov.Channels, cov.Channels)
	d := cmat.New(cov.Channels, cov.Channels)
	for u := 0; u < size.X; u++ {
		for v := 0; v < size.Y; v++ {
			// Over-write matrix.
			for p := 0; p < cov.Channels; p++ {
				for q := 0; q < cov.Channels; q++ {
					auv.Set(p, q, a[p][q].At(u, v))
				}
			}
			vecs, vals, err := clap.EigHerm(auv)
			if err != nil {
				return nil, err
			}
			for i := range vals {
				if vals[i] < 0 {
					vals[i] = 0
				}
				vals[i] += lambda
				d.Set(i, i, complex(1/vals[i], 0))
			}
			inv := cmat.Mul(cmat.Mul(vecs, d), cmat.H(vecs))
			for p := 0; p < cov.Channels; p++ {
				for q := 0; q < cov.Channels; q++ {
					a[p][q].Set(u, v, inv.At(p, q))
				}
			}
		}
	}
	// Take inverse FFT of each channel pair.
	prec := toepcov.NewCovar(cov.Channels, bandOut)
	for p := range a {
		for q := range a[p] {
			apq := a[p][q]
			fftw.IFFT2To(apq, apq)
			for u := -bandOut; u <= bandOut; u++ {
				for v := -bandOut; v <= bandOut; v++ {
					umod, vmod := mod(u, size.X), mod(v, size.Y)
					x := apq.At(umod, vmod)
					// TODO: Check real.
					prec.Set(u, v, p, q, real(x)/num)
				}
			}
		}
	}
	return prec, nil
}
