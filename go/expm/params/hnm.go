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
	"github.com/jvlmdr/go-svm/svm"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/jvlmdr/shift-invar/go/vecset"
	"github.com/nfnt/resize"
)

type HardNegTrainer struct {
	Gamma       float64 // Cost fraction of positives vs negatives (0 to 1).
	InitNegCost float64 // Cost fraction of hard negatives vs initial (0 to 1).
	Lambda      float64
	Bias        float64
	// SVM options.
	Epochs int
	// Hard negative options.
	Rounds   int
	InitNeg  int  // Initial number of random negatives.
	PerRound int  // Number to add in each round.
	Accum    bool // Accumulate negatives?
}

// Field must match HardNegTrainerSet.Fields().
func (t *HardNegTrainer) Field(name string) string {
	switch name {
	case "Gamma":
		return fmt.Sprint(t.Gamma)
	case "InitNegCost":
		return fmt.Sprint(t.InitNegCost)
	case "Lambda":
		return fmt.Sprint(t.Lambda)
	case "Epochs":
		return fmt.Sprint(t.Epochs)
	case "Rounds":
		return fmt.Sprint(t.Rounds)
	case "InitNeg":
		return fmt.Sprint(t.InitNeg)
	case "PerRound":
		return fmt.Sprint(t.PerRound)
	case "Accum":
		return fmt.Sprint(t.Accum)
	default:
		return ""
	}
}

// HardNegTrainerSet provides a mechanism to specify a set of HardNegTrainers.
type HardNegTrainerSet struct {
	Gamma       []float64 // Cost fraction of positives vs negatives (0 to 1).
	InitNegCost []float64 // Cost fraction of hard negatives vs initial (0 to 1).
	Lambda      []float64
	Bias        float64
	// SVM options.
	Epochs []int
	// Hard negative options.
	Rounds   []int
	InitNeg  []int  // Initial number of random negatives.
	PerRound []int  // Number to add in each round.
	Accum    []bool // Accumulate negatives?
}

func (set *HardNegTrainerSet) Fields() []string {
	return []string{
		"Gamma", "InitNegCost", "Lambda",
		"Rounds", "InitNeg", "PerRound", "Accum",
		"Epochs",
	}
}

func (set *HardNegTrainerSet) Enumerate() []Trainer {
	var ts []Trainer
	for _, gamma := range set.Gamma {
		for _, initNegCost := range set.InitNegCost {
			for _, lambda := range set.Lambda {
				for _, epochs := range set.Epochs {
					for _, rounds := range set.Rounds {
						for _, initNeg := range set.InitNeg {
							for _, perRound := range set.PerRound {
								for _, accum := range set.Accum {
									t := &HardNegTrainer{
										Gamma:       gamma,
										InitNegCost: initNegCost,
										Lambda:      lambda,
										Bias:        set.Bias,
										Epochs:      epochs,
										Rounds:      rounds,
										InitNeg:     initNeg,
										PerRound:    perRound,
										Accum:       accum,
									}
									ts = append(ts, t)
								}
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

func (t *HardNegTrainer) Train(posIms, negIms []string, dataset data.ImageSet, phi feat.Image, region detect.PadRect, exampleOpts data.ExampleOpts, flip bool, interp resize.InterpolationFunction, searchOpts detect.MultiScaleOpts) (*detect.FeatTmpl, error) {
	posRects, err := data.PosExampleRects(posIms, dataset, region, exampleOpts)
	if err != nil {
		return nil, err
	}
	// Positive examples are extracted and stored as vectors.
	// TODO: Check dataset.CanTrain()?
	log.Print("sample positive examples")
	pos, err := data.Examples(posIms, posRects, dataset, phi, t.Bias, region, flip, interp)
	if err != nil {
		return nil, err
	}
	if len(pos) == 0 {
		return nil, fmt.Errorf("empty positive set")
	}

	// Choose an initial set of random negatives.
	// TODO: Check dataset.CanTrain()?
	log.Print("choose initial negative examples")
	negRects, err := data.RandomWindows(t.InitNeg, negIms, dataset, region.Size)
	if err != nil {
		return nil, err
	}
	log.Print("sample initial negative examples")
	initNeg, err := data.Examples(negIms, negRects, dataset, phi, t.Bias, region, false, interp)
	if err != nil {
		return nil, err
	}
	log.Println("number of negatives:", len(initNeg))

	var (
		hardNeg [][]float64
		tmpl    *detect.FeatTmpl
	)

	for round := 0; round <= t.Rounds; round++ {
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
				imDets, durSearch, err := detect.MultiScale(im, tmpl, searchOpts)
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
				examples, excl, err := data.ObjectsToExamples(dataset.File(im), objs, region, exampleOpts)
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
			nextHardNeg, err := data.Examples(negIms, exampleRects, dataset, phi, t.Bias, region, false, interp)
			if err != nil {
				return nil, err
			}
			log.Println("new hard negatives:", len(nextHardNeg))

			if t.Accum {
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
		// Determine absolute costs.
		var (
			posCost     = t.Gamma / t.Lambda / float64(len(pos))
			initNegCost = (1 - t.Gamma) * t.InitNegCost / t.Lambda / float64(len(initNeg))
		)
		x = append(x, vecset.Slice(pos))
		for _ = range pos {
			y = append(y, 1)
			c = append(c, posCost)
		}
		x = append(x, vecset.Slice(initNeg))
		for _ = range initNeg {
			y = append(y, -1)
			c = append(c, initNegCost)
		}
		if len(hardNeg) > 0 {
			hardNegCost := (1 - t.Gamma) * (1 - t.InitNegCost) / t.Lambda / float64(len(hardNeg))
			x = append(x, vecset.Slice(hardNeg))
			for _ = range hardNeg {
				y = append(y, -1)
				c = append(c, hardNegCost)
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
		tmpl = &detect.FeatTmpl{
			Image: &rimg64.Multi{
				Width:    featsize.X,
				Height:   featsize.Y,
				Channels: channels,
				Elems:    weights,
			},
			Size:     region.Size,
			Interior: region.Int,
		}
	}
	return tmpl, nil
}
