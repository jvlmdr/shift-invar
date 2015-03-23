package main

import (
	"fmt"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-cv/slide"
	"github.com/jvlmdr/go-svm/svm"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/jvlmdr/shift-invar/go/imset"
	"github.com/jvlmdr/shift-invar/go/vecset"
	"github.com/nfnt/resize"
)

type SVMTrainer struct {
	Bias   float64
	Lambda float64
	Gamma  float64
	Epochs int
}

func (t *SVMTrainer) Field(name string) string {
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

// SVMTrainerSet provides a mechanism to specify a set of SVMTrainers.
type SVMTrainerSet struct {
	Bias   float64
	Lambda []float64
	Gamma  []float64
	Epochs []int
}

func (set *SVMTrainerSet) Fields() []string {
	return []string{"Lambda", "Gamma", "Epochs"}
}

func (set *SVMTrainerSet) Enumerate() []Trainer {
	var ts []Trainer
	for _, lambda := range set.Lambda {
		for _, gamma := range set.Gamma {
			for _, epochs := range set.Epochs {
				t := &SVMTrainer{
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

func (t *SVMTrainer) Train(posIms, negIms []string, dataset data.ImageSet, phi feat.Image, covarFile string, region detect.PadRect, exampleOpts data.ExampleOpts, flip bool, interp resize.InterpolationFunction, searchOpts detect.MultiScaleOpts) (*detect.FeatTmpl, error) {
	posRects, err := data.PosExampleRects(posIms, dataset, searchOpts.Pad.Margin, region, exampleOpts)
	if err != nil {
		return nil, err
	}
	// Positive examples are extracted and stored as vectors.
	pos, err := data.Examples(posIms, posRects, dataset, phi, searchOpts.Pad.Extend, region, flip, interp)
	if err != nil {
		return nil, err
	}
	if len(pos) == 0 {
		return nil, fmt.Errorf("empty positive set")
	}
	// Negative examples are represented as indices into an image.
	neg, err := data.WindowSets(negIms, dataset, phi, searchOpts.Pad, region, interp)
	if err != nil {
		return nil, err
	}
	if len(neg) == 0 {
		return nil, fmt.Errorf("empty negative set")
	}
	// Count number of examples for cost normalization.
	var numNegWindows int
	for i := range neg {
		numNegWindows += neg[i].Len()
	}

	var (
		x []vecset.Set
		y []float64
		c []float64
	)
	// Add positive examples as a set of vectors.
	x = append(x, &imset.VecSet{Set: imset.Slice(pos), Bias: t.Bias})
	for _ = range pos {
		y = append(y, 1)
		c = append(c, t.Gamma/t.Lambda/float64(len(pos)))
	}
	// Add each set of negative vectors.
	for i := range neg {
		x = append(x, &imset.VecSet{Set: neg[i], Bias: t.Bias})
		ni := neg[i].Len()
		// Labels and costs for every positive and negative example.
		for j := 0; j < ni; j++ {
			y = append(y, -1)
			c = append(c, (1-t.Gamma)/t.Lambda/float64(numNegWindows))
		}
	}

	weights, err := svm.Train(vecset.NewUnion(x), y, c,
		func(epoch int, f, fPrev, g, gPrev float64, w, wPrev []float64, a, aPrev map[int]float64) (bool, error) {
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
		Scorer: &slide.AffineScorer{
			Tmpl: &rimg64.Multi{
				Width:    featsize.X,
				Height:   featsize.Y,
				Channels: channels,
				Elems:    weights,
			},
		},
		PixelShape: region,
	}
	return tmpl, nil
}
