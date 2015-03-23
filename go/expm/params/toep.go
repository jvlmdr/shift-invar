package main

import (
	"fmt"

	"github.com/gonum/floats"
	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-cv/slide"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/jvlmdr/shift-invar/go/toepcov"
	"github.com/nfnt/resize"
)

type ToeplitzTrainer struct {
	Lambda float64
}

func (t *ToeplitzTrainer) Field(name string) string {
	switch name {
	case "Lambda":
		return fmt.Sprint(t.Lambda)
	default:
		return ""
	}
}

// ToeplitzTrainerSet provides a mechanism to specify a set of ToeplitzTrainers.
type ToeplitzTrainerSet struct {
	Lambda []float64
}

func (set *ToeplitzTrainerSet) Fields() []string {
	return []string{"Lambda"}
}

func (set *ToeplitzTrainerSet) Enumerate() []Trainer {
	var ts []Trainer
	for _, lambda := range set.Lambda {
		t := &ToeplitzTrainer{Lambda: lambda}
		ts = append(ts, t)
	}
	return ts
}

func (t *ToeplitzTrainer) Train(posIms, negIms []string, dataset data.ImageSet, phi feat.Image, covarFile string, region detect.PadRect, exampleOpts data.ExampleOpts, flip bool, interp resize.InterpolationFunction, searchOpts detect.MultiScaleOpts) (*detect.FeatTmpl, error) {
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
	// Obtain covariance and mean from sums.
	distr := toepcov.Normalize(total, true)
	// Regularize.
	distr.Covar.AddLambdaI(t.Lambda)
	// Subtract negative mean from positive example.
	delta := toepcov.SubMean(meanPos, distr.Mean)
	weights, err := toepcov.InvMulDirect(distr.Covar, delta)
	if err != nil {
		return nil, err
	}

	// Pack weights into image in detection template.
	tmpl := &detect.FeatTmpl{
		Scorer:     &slide.AffineScorer{Tmpl: weights},
		PixelShape: region,
	}
	return tmpl, nil
}
