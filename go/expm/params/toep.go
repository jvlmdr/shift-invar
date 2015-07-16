package main

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"time"

	"github.com/gonum/floats"
	"github.com/jvlmdr/go-cg/cg"
	"github.com/jvlmdr/go-cg/pcg"
	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-cv/slide"
	"github.com/jvlmdr/lin-go/lapack"
	"github.com/jvlmdr/shift-invar/go/circcov"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/jvlmdr/shift-invar/go/toepcov"
	"github.com/nfnt/resize"
)

type ToeplitzTrainer struct {
	Lambda float64
	Method ToeplitzMethod
	// Bandwidth parameter of Gaussian mask.
	// Non-positive means no mask.
	Sigma float64
	// Maximum bandwidth of Toeplitz matrix (-2*Band+1, ..., 2*Band-1).
	// Zero means full bandwidth.
	Band int
}

func (t *ToeplitzTrainer) Field(name string) string {
	switch name {
	case "Circ":
		return fmt.Sprint(t.Method.Circ)
	case "Algo":
		return t.Method.Algo
	case "Tol":
		return fmt.Sprint(t.Method.Tol)
	}
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
	Algo   []string
	Tol    []float64
	Sigma  []float64
	Band   []int
}

type ToeplitzMethod struct {
	Circ bool
	// Algorithm to use when Circ is false.
	// Can be "chol", "cg" or "pcg".
	Algo string
	// Tolerance for when Algo is "cg" or "pcg".
	Tol float64
}

func (set *ToeplitzTrainerSet) Fields() []string {
	return []string{"Lambda", "Circ", "Sigma", "Band", "Algo", "Tol"}
}

func (set *ToeplitzTrainerSet) Enumerate() []Trainer {
	var methods []ToeplitzMethod
	for _, circ := range set.Circ {
		if circ {
			methods = append(methods, ToeplitzMethod{Circ: true})
			continue
		}
		for _, algo := range set.Algo {
			switch algo {
			case "cg", "pcg":
				for _, tol := range set.Tol {
					methods = append(methods, ToeplitzMethod{Algo: algo, Tol: tol})
				}
			default:
				methods = append(methods, ToeplitzMethod{Algo: algo})
			}
		}
	}

	var ts []Trainer
	for _, lambda := range set.Lambda {
		for _, method := range methods {
			for _, sigma := range set.Sigma {
				for _, band := range set.Band {
					t := &ToeplitzTrainer{Lambda: lambda, Method: method, Sigma: sigma, Band: band}
					ts = append(ts, t)
				}
			}
		}
	}
	return ts
}

func (t *ToeplitzTrainer) Train(posIms, negIms []string, dataset data.ImageSet, phi feat.Image, statsFile string, region detect.PadRect, exampleOpts data.ExampleOpts, flip bool, interp resize.InterpolationFunction, searchOpts detect.MultiScaleOpts) (*SolveResult, error) {
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
	var (
		weights *rimg64.Multi
		dur     SolveDuration
	)
	start := time.Now()
	if t.Method.Circ {
		weights, dur.Subst, err = solveCirculant(distr.Covar, delta)
	} else {
		weights, dur.Subst, err = solveToeplitz(distr.Covar, delta, t.Method.Algo, t.Method.Tol)
	}
	if err != nil {
		return &SolveResult{Error: err.Error()}, nil
	}
	dur.Total = time.Since(start)

	// Pack weights into image in detection template.
	tmpl := &detect.FeatTmpl{
		Scorer:     &slide.AffineScorer{Tmpl: weights},
		PixelShape: region,
	}
	return &SolveResult{Tmpl: tmpl, Dur: dur}, nil
}

func solveToeplitz(cov *toepcov.Covar, r *rimg64.Multi, algo string, tol float64) (*rimg64.Multi, time.Duration, error) {
	switch algo {
	case "chol":
		// Instantiate full covariance matrix.
		s := cov.Matrix(r.Width, r.Height)
		chol, err := lapack.Chol(s)
		if err != nil {
			return nil, 0, err
		}
		start := time.Now()
		x, err := chol.Solve(r.Elems)
		if err != nil {
			return nil, 0, err
		}
		dur := time.Since(start)
		w := &rimg64.Multi{x, r.Width, r.Height, r.Channels}
		return w, dur, nil

	case "cg":
		var muler toepcov.MulerFFT
		muler.Init(cov, r.Width, r.Height)
		a := func(x []float64) []float64 {
			f := &rimg64.Multi{x, r.Width, r.Height, r.Channels}
			g := muler.Mul(f)
			return g.Elems
		}
		guess := make([]float64, r.Width*r.Height*r.Channels)
		start := time.Now()
		x, err := cg.Solve(a, r.Elems, guess, tol, 0, os.Stderr)
		if err != nil {
			return nil, 0, err
		}
		dur := time.Since(start)
		w := &rimg64.Multi{x, r.Width, r.Height, r.Channels}
		return w, dur, nil

	case "pcg":
		var muler toepcov.MulerFFT
		muler.Init(cov, r.Width, r.Height)
		a := func(x []float64) []float64 {
			f := &rimg64.Multi{x, r.Width, r.Height, r.Channels}
			g := muler.Mul(f)
			return g.Elems
		}
		var invmuler circcov.InvMuler
		if err := invmuler.Init(cov, r.Width, r.Height); err != nil {
			return nil, 0, err
		}
		cinv := func(x []float64) []float64 {
			f := &rimg64.Multi{x, r.Width, r.Height, r.Channels}
			g := invmuler.Mul(f)
			return g.Elems
		}
		guess := make([]float64, r.Width*r.Height*r.Channels)
		start := time.Now()
		x, err := pcg.Solve(a, r.Elems, cinv, guess, tol, 0, os.Stderr)
		if err != nil {
			return nil, 0, err
		}
		dur := time.Since(start)
		w := &rimg64.Multi{x, r.Width, r.Height, r.Channels}
		return w, dur, nil

	default:
		panic(fmt.Sprintf("unknown algorithm: %q", algo))
	}
}

func solveCirculant(cov *toepcov.Covar, r *rimg64.Multi) (*rimg64.Multi, time.Duration, error) {
	muler := new(circcov.InvMuler)
	if err := muler.Init(cov, r.Width, r.Height); err != nil {
		return nil, 0, err
	}
	start := time.Now()
	w := muler.Mul(r)
	dur := time.Since(start)
	return w, dur, nil
}
