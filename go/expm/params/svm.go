package main

import (
	"fmt"
	"log"
	"reflect"
	"strings"
	"time"

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
	Term   SVMTerm
}

func (t *SVMTrainer) Field(name string) string {
	if strings.HasPrefix(name, "Term.") {
		return t.Term.Field(strings.TrimPrefix(name, "Term."))
	}
	value := reflect.ValueOf(t).Elem().FieldByName(name)
	if !value.IsValid() {
		return ""
	}
	return fmt.Sprint(value.Interface())
}

// SVMTrainerSet provides a mechanism to specify a set of SVMTrainers.
type SVMTrainerSet struct {
	Bias   float64
	Lambda []float64
	Gamma  []float64
	Term   []SVMTermSet
}

func (set *SVMTrainerSet) Fields() []string {
	return []string{"Lambda", "Gamma", "Term.Epochs", "Term.RelGap", "Term.AbsGap"}
}

func (set *SVMTrainerSet) Enumerate() []Trainer {
	var terms []SVMTerm
	for _, term := range set.Term {
		terms = append(terms, term.Enumerate()...)
	}

	var ts []Trainer
	for _, lambda := range set.Lambda {
		for _, gamma := range set.Gamma {
			for _, term := range terms {
				t := &SVMTrainer{
					Bias:   set.Bias,
					Lambda: lambda,
					Gamma:  gamma,
					Term:   term,
				}
				ts = append(ts, t)
			}
		}
	}
	return ts
}

type SVMTerm struct {
	Epochs int // Zero or less means no limit.
	RelGap float64
	AbsGap float64
}

func (term SVMTerm) Field(name string) string {
	value := reflect.ValueOf(term).FieldByName(name)
	if !value.IsValid() {
		return ""
	}
	return fmt.Sprint(value.Interface())
}

func (term SVMTerm) Terminate(epoch int, f, g float64, w []float64, a map[int]float64) (bool, error) {
	log.Printf("bounds: [%.6g, %.6g]", g, f)
	gap := f - g
	// gap and f are positive.
	if relGap := gap / f; relGap <= term.RelGap {
		log.Printf("relative gap %.3g <= %.3g", relGap, term.RelGap)
		return true, nil
	}
	if gap <= term.AbsGap {
		log.Printf("absolute gap %.3g <= %.3g", gap, term.AbsGap)
		return true, nil
	}
	// epoch is the number of completed epochs.
	if term.Epochs > 0 && epoch >= term.Epochs {
		log.Printf("reach iteration limit")
		return true, nil
	}
	return false, nil
}

type SVMTermSet struct {
	Epochs []int
	RelGap []float64
	AbsGap []float64
}

func (set *SVMTermSet) Enumerate() []SVMTerm {
	var x []SVMTerm
	for _, epochs := range set.Epochs {
		for _, relGap := range set.RelGap {
			for _, absGap := range set.AbsGap {
				x = append(x, SVMTerm{Epochs: epochs, RelGap: relGap, AbsGap: absGap})
			}
		}
	}
	return x
}

func (t *SVMTrainer) Train(posIms, negIms []string, dataset data.ImageSet, phi feat.Image, statsFile string, region detect.PadRect, exampleOpts data.ExampleOpts, flip bool, interp resize.InterpolationFunction, searchOpts detect.MultiScaleOpts) (*SolveResult, error) {
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

	start := time.Now()
	weights, err := svm.Train(vecset.NewUnion(x), y, c, t.Term.Terminate)
	if err != nil {
		return nil, err
	}
	dur := time.Since(start)

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
	return &SolveResult{Tmpl: tmpl, Dur: SolveDuration{Total: dur}}, nil
}
