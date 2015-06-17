package data

import (
	"fmt"
	"os"
	"path"

	"github.com/jvlmdr/go-cv/dataset/caltechped"
)

type CaltechSpec struct {
	// Files are in Dir/data-Subdir
	Dir    string
	Subdir string
	// Sub-sample rate.
	Skip int
	// Extension without dot e.g. "jpg".
	Ext string
	// The videos to use within each set.
	Sets []CaltechSet
}

type CaltechSet struct {
	Index  int
	Videos []int
}

// caltechDataset is a list of image names with annotations.
// It provides the path to an image given its name.
// The name of an image is its path from Dir/ without its extension.
type caltechDataset struct {
	dir    string
	subdir string
	ext    string
	ims    []string
	annots map[string]Annot
}

func loadCaltech(spec CaltechSpec, filter caltechped.ObjectFilter) (ImageSet, error) {
	d := new(caltechDataset)
	d.dir = spec.Dir
	d.subdir = spec.Subdir
	d.ext = spec.Ext
	d.annots = make(map[string]Annot)

	topDir := path.Join(d.dir, "data-"+d.subdir)
	for _, set := range spec.Sets {
		for _, vid := range set.Videos {
			relDir := path.Join(fmt.Sprintf("set%02d", set.Index), fmt.Sprintf("V%03d", vid))
			// Check that image directory exists.
			imDir := path.Join(topDir, "images", relDir)
			if err := isDir(imDir); err != nil {
				return nil, fmt.Errorf("images dir: %v", err)
			}
			// Check that annotation directory exists.
			annotDir := path.Join(topDir, "annotations", relDir)
			if err := isDir(annotDir); err != nil {
				return nil, fmt.Errorf("annotations dir: %v", err)
			}

			for j := 0; ; j++ {
				// If skip is 30, use frames 29, 59, ...
				frame := (j+1)*spec.Skip - 1
				name := fmt.Sprintf("set%02d/V%03d/I%05d", set.Index, vid, frame)
				// TODO: This might not be OS-safe.
				imFile := path.Join(topDir, "images", name+"."+d.ext)
				// Continue until image file does not exist.
				if _, err := os.Stat(imFile); os.IsNotExist(err) {
					break
				} else if err != nil {
					return nil, err
				}
				// Load annotation.
				annotFile := path.Join(topDir, "annotations", name+".txt")
				annot, err := caltechped.LoadAnnot(annotFile)
				if err != nil {
					return nil, fmt.Errorf("load annotation: %v", err)
				}
				d.ims = append(d.ims, name)
				d.annots[name] = annotFromCaltech(annot, filter)
			}
		}
	}
	return d, nil
}

func annotFromCaltech(x caltechped.ImageAnnot, filter caltechped.ObjectFilter) Annot {
	var y Annot
	for _, obj := range x.Objects {
		if filter(obj) {
			y.Instances = append(y.Instances, obj.Rect)
		} else {
			y.Ignore = append(y.Ignore, obj.Rect)
		}
	}
	return y
}

func isDir(fname string) error {
	if info, err := os.Stat(fname); err != nil {
		return err
	} else if !info.IsDir() {
		return fmt.Errorf("is not dir: %s", fname)
	}
	return nil
}

func (d *caltechDataset) Images() []string {
	return d.ims
}

func (d *caltechDataset) File(name string) string {
	return path.Join(d.dir, "data-"+d.subdir, "images", name+"."+d.ext)
}

func (d *caltechDataset) CanTrain(name string) bool {
	// Do not exclude any images when training.
	return true
}

func (d *caltechDataset) CanTest(name string) bool {
	// Do not exclude any images when testing.
	return true
}

func (d *caltechDataset) IsNeg(name string) bool {
	// Caltech does not contain any negative images.
	return false
}

func (d *caltechDataset) Annot(name string) Annot {
	return d.annots[name]
}

type CaltechPreset struct {
	Dir string
	// usa, usatrain, usatest, inriatrain, inriatest, ...
	Name string
}

var caltechUSASets = []CaltechSet{
	{Index: 0, Videos: interval(0, 15)},
	{Index: 1, Videos: interval(0, 6)},
	{Index: 2, Videos: interval(0, 12)},
	{Index: 3, Videos: interval(0, 13)},
	{Index: 4, Videos: interval(0, 12)},
	{Index: 5, Videos: interval(0, 13)},
	{Index: 6, Videos: interval(0, 19)},
	{Index: 7, Videos: interval(0, 12)},
	{Index: 8, Videos: interval(0, 11)},
	{Index: 9, Videos: interval(0, 12)},
	{Index: 10, Videos: interval(0, 12)},
}

// Matches definitions in dbInfo.m of Piotr's toolbox.
func (p CaltechPreset) Spec() CaltechSpec {
	switch p.Name {
	case "usa":
		return CaltechSpec{
			Dir:    p.Dir,
			Subdir: "USA",
			Skip:   30,
			Ext:    "jpg",
			Sets:   caltechUSASets,
		}
	case "usatrain":
		return CaltechSpec{
			Dir:    p.Dir,
			Subdir: "USA",
			Skip:   30,
			Ext:    "jpg",
			Sets:   caltechUSASets[0:6],
		}
	case "usatest":
		return CaltechSpec{
			Dir:    p.Dir,
			Subdir: "USA",
			Skip:   30,
			Ext:    "jpg",
			Sets:   caltechUSASets[6:11],
		}
	case "inriatrain":
		return CaltechSpec{
			Dir:    p.Dir,
			Subdir: "INRIA",
			Skip:   1,
			Ext:    "png",
			Sets:   []CaltechSet{{Index: 0, Videos: []int{0, 1}}},
		}
	case "inriatest":
		return CaltechSpec{
			Dir:    p.Dir,
			Subdir: "INRIA",
			Skip:   1,
			Ext:    "png",
			Sets:   []CaltechSet{{Index: 1, Videos: []int{0}}},
		}
	default:
		panic(fmt.Sprintf("unknown preset: %s", p.Name))
	}
}

func interval(a, b int) []int {
	x := make([]int, b-a)
	for i := a; i < b; i++ {
		x[i-a] = i
	}
	return x
}
