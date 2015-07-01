package main

import (
	"fmt"
	"math"
	"reflect"

	"github.com/gonum/floats"
	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-cv/slide"
	"github.com/jvlmdr/shift-invar/go/circcov"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/jvlmdr/shift-invar/go/toepcov"
	"github.com/nfnt/resize"
)

type ToeplitzTrainer struct {
	Lambda float64
	Circ   bool
	// Bandwidth parameter of Gaussian mask.
	// Non-positive means no mask.
	Sigma float64
	// Maximum bandwidth of Toeplitz matrix (-2*Band+1, ..., 2*Band-1).
	// Zero means full bandwidth.
	Band int
}

func (t *ToeplitzTrainer) Field(name string) string {
	value := reflect.ValueOf(t).Elem().FieldByName(name)
	if !value.IsValid() {
		return ""
	}
	return fmt.Sprint(value.Interface())
}

// ToeplitzTrainerSet provides a mechanism to specify a set of ToeplitzTrainers.
type ToeplitzTrainerSet struct {
	Lambda []float64
	Circ   []bool
	Sigma  []float64
	Band   []int
}

func (set *ToeplitzTrainerSet) Fields() []string {
	return []string{"Lambda", "Circ", "Sigma", "Band"}
}

func (set *ToeplitzTrainerSet) Enumerate() []Trainer {
	var ts []Trainer
	for _, lambda := range set.Lambda {
		for _, circ := range set.Circ {
			for _, sigma := range set.Sigma {
				for _, band := range set.Band {
					t := &ToeplitzTrainer{Lambda: lambda, Circ: circ, Sigma: sigma, Band: band}
					ts = append(ts, t)
				}
			}
		}
	}
	return ts
}

func (t *ToeplitzTrainer) Train(posIms, negIms []string, dataset data.ImageSet, phi feat.Image, statsFile string, region detect.PadRect, exampleOpts data.ExampleOpts, flip bool, interp resize.InterpolationFunction, searchOpts detect.MultiScaleOpts) (*TrainResult, error) {
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
	total, err := toepcov.LoadTotalExt(statsFile)
	if err != nil {
		return nil, err
	}
	// Obtain covariance and mean from sums.
	distr := toepcov.Normalize(total, true)
	if t.Band > 0 && t.Band < distr.Covar.Bandwidth {
		distr.Covar = distr.Covar.CloneBandwidth(t.Band)
	}
	bandwidth := distr.Covar.Bandwidth
	if t.Sigma > 0 {
		for u := -bandwidth; u <= bandwidth; u++ {
			for v := -bandwidth; v <= bandwidth; v++ {
				x, y := float64(u), float64(v)
				alpha := math.Exp(-(x*x + y*y) / (2 * t.Sigma * t.Sigma))
				for p := 0; p < channels; p++ {
					for q := 0; q < channels; q++ {
						distr.Covar.Set(u, v, p, q, alpha*distr.Covar.At(u, v, p, q))
					}
				}
			}
		}
	}
	// Regularize.
	distr.Covar.AddLambdaI(t.Lambda)
	// Subtract negative mean from positive example.
	delta := toepcov.SubMean(meanPos, distr.Mean)
	var weights *rimg64.Multi
	if t.Circ {
		weights, err = circcov.InvMul(distr.Covar, delta)
	} else {
		weights, err = toepcov.InvMulDirect(distr.Covar, delta)
	}
	if err != nil {
		return &TrainResult{Error: err.Error()}, nil
	}

	// Pack weights into image in detection template.
	tmpl := &detect.FeatTmpl{
		Scorer:     &slide.AffineScorer{Tmpl: weights},
		PixelShape: region,
	}
	return &TrainResult{Tmpl: tmpl}, nil
}
