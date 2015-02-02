package data

import (
	"path"

	"github.com/jvlmdr/go-cv/dataset/inria"
	"github.com/jvlmdr/go-file/fileutil"
)

type INRIASpec struct {
	// Root containing Train/ and Test/.
	Dir string
	// "Train" or "Test"
	Set string
	// Exclude negative images from the test set?
	ExclNegTest bool
}

type inriaDataset struct {
	dir string
	ims []string
	// Positive images have an entry in this map.
	annots map[string]inria.Annot
	// Negative images have an entry in this map.
	isNeg map[string]bool
	// Exclude negative images from the test set?
	exclNegTest bool
}

func loadINRIA(spec INRIASpec) (ImageSet, error) {
	d := new(inriaDataset)
	d.dir = spec.Dir
	d.exclNegTest = spec.ExclNegTest
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

func (d *inriaDataset) CanTrain(name string) bool {
	// Include all images in the training set.
	return true
}

func (d *inriaDataset) CanTest(name string) bool {
	if d.exclNegTest && d.isNeg[name] {
		return false
	}
	return true
}

func (d *inriaDataset) IsNeg(name string) bool {
	return d.isNeg[name]
}

func (d *inriaDataset) Annot(name string) Annot {
	return Annot{Instances: d.annots[name].Rects}
}
