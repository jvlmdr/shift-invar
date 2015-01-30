package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"log"
	"math"
	"path"

	"github.com/jvlmdr/go-cv/dataset/inria"
	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-file/fileutil"
)

// Dataset may be used to generate training or testing data.
// Every image can be used as a testing image,
// but not all can be used as training images.
type Dataset interface {
	Images() []string
	File(string) string
	// Can every window in the image be a negative example?
	IsNeg(im string) bool
	// Does the image contain any positive examples?
	// If so, it can be used as a positive training image.
	// Note that IsNeg() == true implies that
	// Annot().Instances is empty but not the reverse.
	Annot(im string) Annot
}

// Annot gives the annotation of an image
// for the purpose of evaluating a single detector.
type Annot struct {
	// Things which should be detected.
	Instances []image.Rectangle
	// Regions in which it does not matter whether
	// the detector fires or not.
	Ignore []image.Rectangle
}

func loadDataset(name, specJSON string) (Dataset, error) {
	if name == "inria" {
		var spec INRIASpec
		if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
			return nil, err
		}
		return loadINRIA(spec)
	}
	if name == "caltech" {
		panic("caltech unimplemented")
	}
	panic(fmt.Sprintf("unknown dataset: %s", name))
}

type INRIASpec struct {
	Dir string
	// "Train" or "Test"
	Set string
}

type inriaDataset struct {
	dir string
	ims []string
	// Positive images have an entry in this map.
	annots map[string]inria.Annot
	// Negative images have an entry in this map.
	isNeg map[string]bool
}

func loadINRIA(spec INRIASpec) (Dataset, error) {
	d := new(inriaDataset)
	d.dir = spec.Dir
	// Load list of annotations.
	annotFiles, err := fileutil.LoadLines(path.Join(spec.Dir, spec.Set, "annotations.lst"))
	if err != nil {
		return nil, err
	}
	// Add positive images to list.
	d.annots = make(map[string]inria.Annot, len(annotFiles))
	for _, file := range annotFiles {
		annot, err := inria.LoadAnnot(path.Join(spec.Dir, file))
		if err != nil {
			return nil, err
		}
		d.annots[annot.Image] = annot
		d.ims = append(d.ims, annot.Image)
	}
	// Load list of negative images.
	negIms, err := fileutil.LoadLines(path.Join(spec.Dir, spec.Set, "neg.lst"))
	if err != nil {
		return nil, err
	}
	d.isNeg = make(map[string]bool, len(negIms))
	for _, im := range negIms {
		d.isNeg[im] = true
		d.ims = append(d.ims, im)
	}
	return d, nil
}

func (d *inriaDataset) Images() []string {
	return d.ims
}

func (d *inriaDataset) File(name string) string {
	return path.Join(d.dir, name)
}

func (d *inriaDataset) IsNeg(name string) bool {
	return d.isNeg[name]
}

func (d *inriaDataset) Annot(name string) Annot {
	return Annot{Instances: d.annots[name].Rects}
}

type TrainData struct {
	PosImages []string
	NegImages []string
	PosRects  map[string][]image.Rectangle
}

var (
	errTooSmall  = errors.New("too small")
	errBadAspect = errors.New("bad aspect ratio")
	errNotInside = errors.New("not inside image")
)

// Extracts training data for a subset of images.
func extractTrainingData(data Dataset, ims []string, region detect.PadRect, aspectReject float64, resizeFor string, maxScale float64) (*TrainData, error) {
	var valid, badAspect, tooSmall, notInside int
	trainData := new(TrainData)
	trainData.PosRects = make(map[string][]image.Rectangle)
	for _, im := range ims {
		if data.IsNeg(im) {
			trainData.NegImages = append(trainData.NegImages, im)
			continue
		}
		rects := data.Annot(im).Instances
		if len(rects) == 0 {
			// Not every window is negative and not one window is positive.
			continue
		}
		// Get size of image.
		file := data.File(im)
		size, err := loadImageSize(file)
		if err != nil {
			log.Printf("load image size: %s, error: %v", file, err)
			continue
		}
		bounds := image.Rectangle{image.ZP, size}
		var examples []image.Rectangle
		for _, label := range rects {
			// Attempt to standardize each rectangle.
			example, err := adjustRect(label, bounds, region, aspectReject, resizeFor, maxScale)
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
		trainData.PosImages = append(trainData.PosImages, im)
		trainData.PosRects[im] = examples
	}
	log.Printf("valid: %d, bad aspect: %d, too small: %d, not inside: %d", valid, badAspect, tooSmall, notInside)
	return trainData, nil
}

func adjustRect(orig, bounds image.Rectangle, region detect.PadRect, aspectReject float64, resizeFor string, maxScale float64) (image.Rectangle, error) {
	// Check if aspect is too far from desired.
	aspect := float64(region.Int.Dx()) / float64(region.Int.Dy())
	origAspect := float64(orig.Dx()) / float64(orig.Dy())
	if aspectReject > 0 {
		if math.Abs(math.Log(origAspect)-math.Log(aspect)) > math.Abs(math.Log(aspectReject)) {
			return image.ZR, errBadAspect
		}
	}
	scale, rect := detect.FitRect(orig, region, resizeFor)
	if scale > maxScale {
		return image.ZR, errTooSmall
	}
	if !rect.In(bounds) {
		return image.ZR, errNotInside
	}
	return rect, nil
}
