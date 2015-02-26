package main

import (
	"fmt"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-svm/setsvm"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/jvlmdr/shift-invar/go/vecset"
	"github.com/nfnt/resize"
)

type SetSVMTrainer struct {
	Bias   float64
	Lambda float64
	Gamma  float64
	Epochs int
}

func (t *SetSVMTrainer) Field(name string) string {
	switch name {
	case "Lambda":
		return fmt.Sprint(t.Lambda)
	case "Gamma":
		return fmt.Sprint(t.Gamma)
	case "Epochs":
		return fmt.Sprint(t.Epochs)
	default:
		return ""
	}
}

// SetSVMTrainerSet provides a mechanism to specify a set of SetSVMTrainers.
type SetSVMTrainerSet struct {
	Bias   float64
	Lambda []float64
	Gamma  []float64
	Epochs []int
}

func (set *SetSVMTrainerSet) Fields() []string {
	return []string{"Lambda", "Gamma", "Epochs"}
}

func (set *SetSVMTrainerSet) Enumerate() []Trainer {
	var ts []Trainer
	for _, lambda := range set.Lambda {
		for _, gamma := range set.Gamma {
			for _, epochs := range set.Epochs {
				t := &SetSVMTrainer{
					Bias:   set.Bias,
					Lambda: lambda,
					Gamma:  gamma,
					Epochs: epochs,
				}
				ts = append(ts, t)
			}
		}
	}
	return ts
}

func (t *SetSVMTrainer) Train(posIms, negIms []string, dataset data.ImageSet, phi feat.Image, region detect.PadRect, exampleOpts data.ExampleOpts, flip bool, interp resize.InterpolationFunction, searchOpts detect.MultiScaleOpts) (*detect.FeatTmpl, error) {
	posRects, err := data.PosExampleRects(posIms, dataset, searchOpts.Pad.Margin, region, exampleOpts)
	if err != nil {
		return nil, err
	}
	// Positive examples are extracted and stored as vectors.
	pos, err := data.Examples(posIms, posRects, dataset, phi, searchOpts.Pad.Extend, t.Bias, region, flip, interp)
	if err != nil {
		return nil, err
	}
	if len(pos) == 0 {
		return nil, fmt.Errorf("empty positive set")
	}
	// Negative examples are represented as indices into an image.
	neg, err := data.WindowSets(negIms, dataset, phi, searchOpts.Pad, t.Bias, region, interp)
	if err != nil {
		return nil, err
	}
	if len(neg) == 0 {
		return nil, fmt.Errorf("empty negative set")
	}

	var (
		x []setsvm.Set
		y []float64
		c []float64
	)
	// Add each positive example as a single-element set.
	for _, xi := range pos {
		x = append(x, vecset.Slice([][]float64{xi}))
		y = append(y, 1)
		c = append(c, t.Gamma/t.Lambda/float64(len(pos)))
	}
	// Add each set of negative vectors.
	for _, xi := range neg {
		x = append(x, xi)
		y = append(y, -1)
		c = append(c, (1-t.Gamma)/t.Lambda/float64(len(neg)))
	}

	weights, err := setsvm.Train(x, y, c,
		func(epoch int, f, fPrev, g, gPrev float64, w, wPrev []float64, a, aPrev map[setsvm.Index]float64) (bool, error) {
			if epoch < t.Epochs {
				return false, nil
			}
			return true, nil
		},
	)
	if err != nil {
		return nil, err
	}

	featsize := phi.Size(region.Size)
	channels := phi.Channels()
	// Exclude bias term if present.
	weights = weights[:featsize.X*featsize.Y*channels]
	// Pack weights into image in detection template.
	tmpl := &detect.FeatTmpl{
		Image: &rimg64.Multi{
			Width:    featsize.X,
			Height:   featsize.Y,
			Channels: channels,
			Elems:    weights,
		},
		Size:     region.Size,
		Interior: region.Int,
	}
	return tmpl, nil
}
