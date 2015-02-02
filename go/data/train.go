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

// ExtractTrainingSet generates training data for a subset of images.
func ExtractTrainingSet(dataset ImageSet, ims []string, region detect.PadRect, opts ExampleOpts) (*TrainingSet, error) {
	var valid, badAspect, tooSmall, notInside int
	train := new(TrainingSet)
	train.PosRects = make(map[string][]image.Rectangle)
	for _, im := range ims {
		if dataset.IsNeg(im) {
			train.NegImages = append(train.NegImages, im)
			continue
		}
		rects := dataset.Annot(im).Instances
		if len(rects) == 0 {
			// Not every window is negative and not one window is positive.
			continue
		}
		// Get size of image.
		file := dataset.File(im)
		size, err := loadImageSize(file)
		if err != nil {
			log.Printf("load image size: %s, error: %v", file, err)
			continue
		}
		bounds := image.Rectangle{image.ZP, size}
		var examples []image.Rectangle
		for _, label := range rects {
			// Attempt to standardize each rectangle.
			example, err := adjustRect(label, bounds, region, opts)
			if err != nil {
				switch err {
				case errTooSmall:
					tooSmall++
				case errBadAspect:
					badAspect++
				case errNotInside:
					notInside++
				default:
					return nil, err
				}
				continue
			}
			examples = append(examples, example)
			valid++
		}
		// Do not add empty positive images to the list.
		if len(examples) == 0 {
			// No positive windows.
			continue
		}
		train.PosImages = append(train.PosImages, im)
		train.PosRects[im] = examples
	}
	log.Printf("valid: %d, bad aspect: %d, too small: %d, not inside: %d", valid, badAspect, tooSmall, notInside)
	return train, nil
}

func adjustRect(orig, bounds image.Rectangle, region detect.PadRect, opts ExampleOpts) (image.Rectangle, error) {
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
	if !rect.In(bounds) {
		return image.ZR, errNotInside
	}
	return rect, nil
}
