package main

import (
	"fmt"
	"log"
	"reflect"
	"strings"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-cv/slide"
	"github.com/jvlmdr/go-svm/setsvm"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/jvlmdr/shift-invar/go/imset"
	"github.com/nfnt/resize"
)

type SetSVMTrainer struct {
	Bias   float64
	Lambda float64
	Gamma  float64
	Term   SetSVMTerm
}

func (t *SetSVMTrainer) Field(name string) string {
	if strings.HasPrefix(name, "Term.") {
		return t.Term.Field(strings.TrimPrefix(name, "Term."))
	}
	value := reflect.ValueOf(t).Elem().FieldByName(name)
	if !value.IsValid() {
		return ""
	}
	return fmt.Sprint(value.Interface())
}

// SetSVMTrainerSet provides a mechanism to specify a set of SetSVMTrainers.
type SetSVMTrainerSet struct {
	Bias   float64
	Lambda []float64
	Gamma  []float64
	Term   []SetSVMTermList
}

func (set *SetSVMTrainerSet) Fields() []string {
	return []string{"Lambda", "Gamma", "Term.Epochs", "Term.RelGap", "Term.AbsGap"}
}

func (set *SetSVMTrainerSet) Enumerate() []Trainer {
	var terms []SetSVMTerm
	for _, term := range set.Term {
		terms = append(terms, term.Enumerate()...)
	}

	var ts []Trainer
	for _, lambda := range set.Lambda {
		for _, gamma := range set.Gamma {
			for _, term := range terms {
				t := &SetSVMTrainer{
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

type SetSVMTerm struct {
	Epochs int // Zero for no maximum.
	RelGap float64
	AbsGap float64
}

func (term SetSVMTerm) Field(name string) string {
	value := reflect.ValueOf(term).FieldByName(name)
	if !value.IsValid() {
		return ""
	}
	return fmt.Sprint(value.Interface())
}

func (term SetSVMTerm) Terminate(epoch int, f, g float64, w []float64, a map[setsvm.Index]float64) (bool, error) {
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

type SetSVMTermList struct {
	Epochs []int
	RelGap []float64
	AbsGap []float64
}

func (set *SetSVMTermList) Enumerate() []SetSVMTerm {
	var x []SetSVMTerm
	for _, epochs := range set.Epochs {
		for _, relGap := range set.RelGap {
			for _, absGap := range set.AbsGap {
				x = append(x, SetSVMTerm{Epochs: epochs, RelGap: relGap, AbsGap: absGap})
			}
		}
	}
	return x
}

func (t *SetSVMTrainer) Train(posIms, negIms []string, dataset data.ImageSet, phi feat.Image, statsFile string, region detect.PadRect, exampleOpts data.ExampleOpts, flip bool, interp resize.InterpolationFunction, searchOpts detect.MultiScaleOpts) (*detect.FeatTmpl, error) {
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

	var (
		x []setsvm.Set
		y []float64
		c []float64
	)
	// Add each positive example as a single-element set.
	for _, xi := range pos {
		x = append(x, &imset.VecSet{Set: imset.Slice([]*rimg64.Multi{xi}), Bias: t.Bias})
		y = append(y, 1)
		c = append(c, t.Gamma/t.Lambda/float64(len(pos)))
	}
	// Add each set of negative vectors.
	for _, xi := range neg {
		x = append(x, &imset.VecSet{Set: xi, Bias: t.Bias})
		y = append(y, -1)
		c = append(c, (1-t.Gamma)/t.Lambda/float64(len(neg)))
	}

	weights, err := setsvm.Train(x, y, c, t.Term.Terminate, false)
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
