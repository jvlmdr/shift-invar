package data

import (
	"errors"
	"image"
	"log"
	"math"

	"github.com/jvlmdr/go-cv/detect"
)

// TrainingSet is a subset of an ImageSet.
// It contains a list of rectangles for each positive image
// which may have been modified from the original labels.
type TrainingSet struct {
	PosImages []string
	NegImages []string
	PosRects  map[string][]image.Rectangle
}

var (
	errTooSmall  = errors.New("too small")
	errBadAspect = errors.New("bad aspect ratio")
	errNotInside = errors.New("not inside image")
)

type ExampleOpts struct {
	// Maximum relative aspect ratio before example is discarded.
	// Can be above or below one.
	// All that matters is abs(log(AspectReject)).
	// Ignored if zero.
	AspectReject float64
	// Argument to detect.FitRect().
	FitMode string
	// Maximum zoom before an example is discarded.
	MaxScale float64
}

type ExcludeCount struct {
	BadAspect int
	TooSmall  int
	NotInside int
}

func (excl ExcludeCount) Plus(other ExcludeCount) ExcludeCount {
	excl.BadAspect += other.BadAspect
	excl.TooSmall += other.TooSmall
	excl.NotInside += other.NotInside
	return excl
}

// ObjectsToExamples takes a set of tight bounding boxes and returns
// rectangles to use for examples.
// Some objects may be excluded due to the criteria.
func ObjectsToExamples(imfile string, objs []image.Rectangle, region detect.PadRect, opts ExampleOpts) ([]image.Rectangle, ExcludeCount, error) {
	if len(objs) == 0 {
		return nil, ExcludeCount{}, nil
	}
	size, err := loadImageSize(imfile)
	if err != nil {
		return nil, ExcludeCount{}, err
	}
	var examples []image.Rectangle
	limits := image.Rectangle{image.ZP, size}
	var excl ExcludeCount
	for _, obj := range objs {
		// Attempt to standardize each rectangle.
		example, err := adjustRect(obj, limits, region, opts)
		if err != nil {
			switch err {
			case errTooSmall:
				excl.TooSmall++
			case errBadAspect:
				excl.BadAspect++
			case errNotInside:
				excl.NotInside++
			default:
				return nil, ExcludeCount{}, err
			}
			continue
		}
		examples = append(examples, example)
	}
	return examples, excl, nil
}

// PosExampleRects produces example rectangles from dataset annotations.
func PosExampleRects(ims []string, dataset ImageSet, region detect.PadRect, opts ExampleOpts) (map[string][]image.Rectangle, error) {
	var (
		valid     int
		totalExcl ExcludeCount
	)
	rects := make(map[string][]image.Rectangle)
	for _, im := range ims {
		// Get tight object bounding rectangles.
		objs := dataset.Annot(im).Instances
		if len(objs) == 0 {
			continue
		}
		examples, excl, err := ObjectsToExamples(dataset.File(im), objs, region, opts)
		if err != nil {
			return nil, err
		}
		totalExcl = totalExcl.Plus(excl)
		valid += len(examples)
		// Do not add empty positive images to the list.
		if len(examples) == 0 {
			// No positive windows.
			continue
		}
		rects[im] = examples
	}
	log.Printf(
		"valid: %d, bad aspect: %d, too small: %d, not inside: %d",
		valid, totalExcl.BadAspect, totalExcl.TooSmall, totalExcl.NotInside,
	)
	return rects, nil
}

//	// ExtractTrainingSet generates training data for a subset of images.
//	func ExtractTrainingSet(dataset ImageSet, ims []string, region detect.PadRect, opts ExampleOpts) (*TrainingSet, error) {
//		var (
//			valid int
//			excl  ExcludeCount
//		)
//		train := new(TrainingSet)
//		train.PosRects = make(map[string][]image.Rectangle)
//		for _, im := range ims {
//			if dataset.IsNeg(im) {
//				train.NegImages = append(train.NegImages, im)
//				continue
//			}
//			objs := dataset.Annot(im).Instances
//			if len(objs) == 0 {
//				// Not every window is negative and not one window is positive.
//				continue
//			}
//			examples, imExcl, err := ObjectsToExamples(dataset.File(im), objs, region, opts)
//			if err != nil {
//				return nil, err
//			}
//			excl = excl.Plus(imExcl)
//			valid += len(examples)
//			// Do not add empty positive images to the list.
//			if len(examples) == 0 {
//				// No positive windows.
//				continue
//			}
//			train.PosImages = append(train.PosImages, im)
//			train.PosRects[im] = examples
//		}
//		log.Printf(
//			"valid: %d, bad aspect: %d, too small: %d, not inside: %d",
//			valid, excl.BadAspect, excl.TooSmall, excl.NotInside,
//		)
//		return train, nil
//	}

func adjustRect(orig, limits image.Rectangle, region detect.PadRect, opts ExampleOpts) (image.Rectangle, error) {
	// Check if aspect is too far from desired.
	aspect := float64(region.Int.Dx()) / float64(region.Int.Dy())
	origAspect := float64(orig.Dx()) / float64(orig.Dy())
	if opts.AspectReject > 0 {
		if math.Abs(math.Log(origAspect)-math.Log(aspect)) > math.Abs(math.Log(opts.AspectReject)) {
			return image.ZR, errBadAspect
		}
	}
	scale, rect := detect.FitRect(orig, region, opts.FitMode)
	if scale > opts.MaxScale {
		return image.ZR, errTooSmall
	}
	if !rect.In(limits) {
		return image.ZR, errNotInside
	}
	return rect, nil
}
