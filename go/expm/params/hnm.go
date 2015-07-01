package main

import (
	"fmt"
	"image"
	"log"
	"sort"
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

type HardNegTrainer struct {
	Gamma  float64 // Cost fraction of positives vs negatives (0 to 1).
	Lambda float64
	Bias   float64
	// Hard negative options.
	NegBehav   NegBehavior
	InitNeg    int  // Initial number of random negatives.
	PerRound   int  // Maximum number to add in each round.
	RequirePos bool // Check that score is positive.
	// SVM options.
	Epochs int
}

type NegBehavior struct {
	Init   InitNegBehavior
	Rounds int
	// Specifies how to normalize hard negatives: by "images", or "examples".
	// Only has an effect if Rounds >= 1.
	Normalize string
	// Accum only has an effect if:
	//   IsolateInit is true and Rounds >= 2
	//   IsolateInit is false and Rounds >= 1 (replace initial negs)
	Accum bool
}

type InitNegBehavior struct {
	Isolate bool    // Normalize and penalize initial negs separate to hard negs?
	Cost    float64 // If Isolate, cost fraction of initial negs vs hard (0 to 1).
}

// Field must match HardNegTrainerSet.Fields().
func (t *HardNegTrainer) Field(name string) string {
	switch name {
	case "Gamma":
		return fmt.Sprint(t.Gamma)
	case "Lambda":
		return fmt.Sprint(t.Lambda)
	case "IsolateInit":
		return fmt.Sprint(t.NegBehav.Init.Isolate)
	case "InitNegCost":
		return fmt.Sprint(t.NegBehav.Init.Cost)
	case "Rounds":
		return fmt.Sprint(t.NegBehav.Rounds)
	case "NormalizeNeg":
		return fmt.Sprint(t.NegBehav.Normalize)
	case "Accum":
		return fmt.Sprint(t.NegBehav.Accum)
	case "InitNeg":
		return fmt.Sprint(t.InitNeg)
	case "PerRound":
		return fmt.Sprint(t.PerRound)
	case "RequirePos":
		return fmt.Sprint(t.RequirePos)
	case "Epochs":
		return fmt.Sprint(t.Epochs)
	default:
		return ""
	}
}

// HardNegTrainerSet provides a mechanism to specify a set of HardNegTrainers.
type HardNegTrainerSet struct {
	Gamma  []float64 // Cost fraction of positives vs negatives (0 to 1).
	Lambda []float64
	Bias   float64
	// Hard negative options.
	IsolateInit  []bool    // Normalize and penalize initial negs separate to hard negs?
	InitNegCost  []float64 // If IsolateInit, cost fraction of initial negs vs hard (0 to 1).
	Rounds       []int
	NormalizeNeg []string // How to normalize hard negatives.
	Accum        []bool   // Accumulate negatives?
	InitNeg      []int    // Initial number of random negatives.
	PerRound     []int    // Maximum number to add in each round.
	RequirePos   []bool   // Check that score is positive.
	// SVM options.
	Epochs []int
}

func (set *HardNegTrainerSet) Fields() []string {
	return []string{
		"Gamma", "Lambda",
		"IsolateInit", "InitNegCost", "Rounds", "NormalizeNeg", "Accum",
		"InitNeg", "PerRound", "RequirePos",
		"Epochs",
	}
}

func (set *HardNegTrainerSet) Enumerate() []Trainer {
	var initBehavs []InitNegBehavior
	for _, isolateInit := range set.IsolateInit {
		if isolateInit {
			for _, initNegCost := range set.InitNegCost {
				behav := InitNegBehavior{Isolate: true, Cost: initNegCost}
				initBehavs = append(initBehavs, behav)
			}
		} else {
			initBehavs = append(initBehavs, InitNegBehavior{Isolate: false})
		}
	}

	var behavs []NegBehavior
	for _, init := range initBehavs {
		for _, num := range set.Rounds {
			if num < 1 {
				// No rounds of mining.
				// Normalize and Accum will have no effect.
				behavs = append(behavs, NegBehavior{Init: init, Rounds: num})
				continue
			}
			for _, normalize := range set.NormalizeNeg {
				if num == 1 && init.Isolate {
					// One round with initial negatives segregated.
					// Accum will have no effect.
					behavs = append(behavs, NegBehavior{Init: init, Rounds: num, Normalize: normalize})
					continue
				}
				for _, accum := range set.Accum {
					behavs = append(behavs, NegBehavior{Init: init, Rounds: num, Normalize: normalize, Accum: accum})
				}
			}
		}
	}

	var ts []Trainer
	for _, gamma := range set.Gamma {
		for _, lambda := range set.Lambda {
			for _, epochs := range set.Epochs {
				for _, behav := range behavs {
					for _, initNeg := range set.InitNeg {
						for _, perRound := range set.PerRound {
							for _, requirePos := range set.RequirePos {
								t := &HardNegTrainer{
									Gamma:      gamma,
									Lambda:     lambda,
									Bias:       set.Bias,
									Epochs:     epochs,
									NegBehav:   behav,
									InitNeg:    initNeg,
									PerRound:   perRound,
									RequirePos: requirePos,
								}
								ts = append(ts, t)
							}
						}
					}
				}
			}
		}
	}
	return ts
}

type Det struct {
	Image string
	detect.Det
}

type byScore []Det

func (xs byScore) Len() int           { return len(xs) }
func (xs byScore) Less(i, j int) bool { return xs[i].Score < xs[j].Score }
func (xs byScore) Swap(i, j int)      { xs[i], xs[j] = xs[j], xs[i] }

func (t *HardNegTrainer) Train(posIms, negIms []string, dataset data.ImageSet, phi feat.Image, statsFile string, region detect.PadRect, exampleOpts data.ExampleOpts, flip bool, interp resize.InterpolationFunction, searchOpts detect.MultiScaleOpts) (*TrainResult, error) {
	// Over-ride MinScore in searchOpts.
	if t.RequirePos {
		searchOpts.DetFilter.MinScore = 0
	}
	posRects, err := data.PosExampleRects(posIms, dataset, searchOpts.Pad.Margin, region, exampleOpts)
	if err != nil {
		return nil, err
	}
	// Positive examples are extracted and stored as vectors.
	// TODO: Check dataset.CanTrain()?
	log.Print("sample positive examples")
	pos, err := data.Examples(posIms, posRects, dataset, phi, searchOpts.Pad.Extend, region, flip, interp)
	if err != nil {
		return nil, err
	}
	if len(pos) == 0 {
		return nil, fmt.Errorf("empty positive set")
	}

	// Choose an initial set of random negatives.
	// TODO: Check dataset.CanTrain()?
	log.Print("choose initial negative examples")
	negRects, err := data.RandomWindows(t.InitNeg, negIms, dataset, searchOpts.Pad.Margin, region.Size)
	if err != nil {
		return nil, err
	}
	log.Print("sample initial negative examples")
	initNeg, err := data.Examples(negIms, negRects, dataset, phi, searchOpts.Pad.Extend, region, false, interp)
	if err != nil {
		return nil, err
	}
	log.Println("number of negatives:", len(initNeg))

	var (
		hardNeg []*rimg64.Multi
		tmpl    *detect.FeatTmpl
	)
	if t.NegBehav.Init.Isolate {
		hardNeg = initNeg
	}

	for round := 0; round <= t.NegBehav.Rounds; round++ {
		if round > 0 {
			// Search all negative images to obtain new hard negatives.
			var dets []Det
			for i, name := range negIms {
				log.Printf("search image %d / %d: %s", i+1, len(negIms), name)
				// Load image.
				file := dataset.File(name)
				s := time.Now()
				im, err := loadImage(file)
				if err != nil {
					log.Printf("load test image: %s, error: %v", file, err)
					continue
				}
				durLoad := time.Since(s)
				imDets, durSearch, err := detect.MultiScale(im, tmpl.Scorer, tmpl.PixelShape, searchOpts)
				if err != nil {
					return nil, err
				}
				for _, det := range imDets {
					dets = append(dets, Det{Image: name, Det: det})
				}
				log.Printf(
					"load %v, resize %v, feat %v, slide %v, suppr %v",
					durLoad, durSearch.Resize, durSearch.Feat, durSearch.Slide, durSearch.Suppr,
				)
			}
			// Sort detections decreasing by score.
			sort.Sort(sort.Reverse(byScore(dets)))
			if len(dets) > t.PerRound {
				dets = dets[:t.PerRound]
			}
			log.Println("found hard negatives:", len(dets))

			// Group by image, discard score.
			objRects := make(map[string][]image.Rectangle)
			for _, det := range dets {
				objRects[det.Image] = append(objRects[det.Image], det.Rect)
			}
			// Need to go from tight rectangles to example rectangles.
			exampleRects := make(map[string][]image.Rectangle)
			var count int
			var totalExcl data.ExcludeCount
			for im, objs := range objRects {
				size, err := loadImageSize(dataset.File(im))
				if err != nil {
					return nil, err
				}
				examples, excl, err := data.ObjectsToExamples(objs, region, exampleOpts, size, searchOpts.Pad.Margin)
				if err != nil {
					return nil, err
				}
				exampleRects[im] = examples
				totalExcl = totalExcl.Plus(excl)
				count += len(examples)
			}
			log.Printf(
				"valid: %d, bad aspect: %d, too small: %d, not inside: %d",
				count, totalExcl.BadAspect, totalExcl.TooSmall, totalExcl.NotInside,
			)
			// Extract vectors.
			nextHardNeg, err := data.Examples(negIms, exampleRects, dataset, phi, searchOpts.Pad.Extend, region, false, interp)
			if err != nil {
				return nil, err
			}
			log.Println("new hard negatives:", len(nextHardNeg))

			if t.NegBehav.Accum {
				hardNeg = append(hardNeg, nextHardNeg...)
			} else {
				hardNeg = nextHardNeg
			}
			log.Println("total hard negatives:", len(hardNeg))
		}

		// Train an SVM, update template.
		var (
			x []vecset.Set
			y []float64
			c []float64
		)
		// Add positive examples.
		posCost := t.Gamma / t.Lambda / float64(len(pos))
		x = append(x, &imset.VecSet{Set: imset.Slice(pos), Bias: t.Bias})
		for _ = range pos {
			y = append(y, 1)
			c = append(c, posCost)
		}
		// Add initial negatives if keeping separate.
		if t.NegBehav.Init.Isolate {
			initNegCost := (1 - t.Gamma) * t.NegBehav.Init.Cost / t.Lambda / float64(len(initNeg))
			x = append(x, &imset.VecSet{Set: imset.Slice(initNeg), Bias: t.Bias})
			for _ = range initNeg {
				y = append(y, -1)
				c = append(c, initNegCost)
			}
		}
		// Add other negatives.
		if len(hardNeg) > 0 {
			hardNegCost := (1 - t.Gamma) / t.Lambda
			switch t.NegBehav.Normalize {
			case "images":
				hardNegCost /= float64(len(negIms))
				if t.NegBehav.Accum {
					// Divide by number of images and number of rounds.
					hardNegCost /= float64(round)
				}
			case "examples":
				hardNegCost /= float64(len(hardNeg))
			}
			if t.NegBehav.Init.Isolate {
				hardNegCost *= (1 - t.NegBehav.Init.Cost)
			}

			x = append(x, &imset.VecSet{Set: imset.Slice(hardNeg), Bias: t.Bias})
			for _ = range hardNeg {
				y = append(y, -1)
				c = append(c, hardNegCost)
			}
		}

		weights, err := svm.Train(vecset.NewUnion(x), y, c,
			func(epoch int, f, g float64, w []float64, a map[int]float64) (bool, error) {
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
		// Extract bias.
		var bias float64
		if t.Bias != 0 {
			bias = weights[featsize.X*featsize.Y*channels] * t.Bias
		}
		// Exclude bias if present.
		weights = weights[:featsize.X*featsize.Y*channels]
		// Pack weights into image in detection template.
		tmpl = &detect.FeatTmpl{
			Scorer: &slide.AffineScorer{
				Tmpl: &rimg64.Multi{
					Width:    featsize.X,
					Height:   featsize.Y,
					Channels: channels,
					Elems:    weights,
				},
				Bias: bias,
			},
			PixelShape: region,
		}
	}
	return &TrainResult{Tmpl: tmpl}, nil
}
