package data

import (
	"errors"
	"image"
	"log"
	"math"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
)

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
// Examples which do not lie inside lims are discarded.
func ObjectsToExamples(objs []image.Rectangle, region detect.PadRect, opts ExampleOpts, size image.Point, margin feat.Margin) ([]image.Rectangle, ExcludeCount, error) {
	if len(objs) == 0 {
		return nil, ExcludeCount{}, nil
	}
	var examples []image.Rectangle
	var excl ExcludeCount
	for _, obj := range objs {
		// Attempt to standardize each rectangle.
		example, err := adjustRect(obj, size, margin, region, opts)
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
func PosExampleRects(ims []string, dataset ImageSet, margin feat.Margin, region detect.PadRect, opts ExampleOpts) (map[string][]image.Rectangle, error) {
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
		size, err := loadImageSize(dataset.File(im))
		if err != nil {
			return nil, err
		}
		examples, excl, err := ObjectsToExamples(objs, region, opts, size, margin)
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

func adjustRect(orig image.Rectangle, size image.Point, margin feat.Margin, region detect.PadRect, opts ExampleOpts) (image.Rectangle, error) {
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
	// Scale image and then add padding.
	scaledIm := image.Pt(int(0.5+scale*float64(size.X)), int(0.5+scale*float64(size.Y)))
	lims := margin.AddTo(image.Rect(0, 0, scaledIm.X, scaledIm.Y))
	scaledRect := image.Rect(
		int(0.5+scale*float64(rect.Min.X)),
		int(0.5+scale*float64(rect.Min.Y)),
		int(0.5+scale*float64(rect.Max.X)),
		int(0.5+scale*float64(rect.Max.Y)),
	)
	if !scaledRect.In(lims) {
		return image.ZR, errNotInside
	}
	return rect, nil
}
