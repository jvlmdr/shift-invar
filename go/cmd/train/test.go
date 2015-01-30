package main

//	import (
//		"image"
//		"log"
//		"math"
//		"math/rand"
//		"path"
//
//		"github.com/jvlmdr/go-cv/dataset/inria"
//		"github.com/jvlmdr/go-cv/detect"
//		"github.com/jvlmdr/go-cv/feat"
//		"github.com/jvlmdr/go-cv/featpyr"
//		"github.com/jvlmdr/go-cv/imgpyr"
//		"github.com/jvlmdr/go-cv/imsamp"
//		"github.com/jvlmdr/go-file/fileutil"
//		"github.com/jvlmdr/go-pbs-pro/dstrfn"
//	)
//
//	func test(inriaDir, set string, tmpl *detect.FeatTmpl, opts DetectOpts, mininter float64) (map[string]*detect.ResultSet, error) {
//		// Load and merge positive and negative test sets.
//		log.Println("load test data")
//		annots, err := loadTestAnnots(inriaDir, set)
//		if err != nil {
//			return nil, err
//		}
//
//		// Execute in parallel.
//		var vals []*detect.ResultSet
//		conf := ValidateArgs{
//			Tmpl:       tmpl,
//			Dir:        inriaDir,
//			DetectOpts: opts,
//			MinInter:   mininter,
//		}
//		if err := dstrfn.MapFunc("test", &vals, annots, conf); err != nil {
//			log.Fatalln("test:", err)
//		}
//		m := make(map[string]*detect.ResultSet, len(vals))
//		for i := range annots {
//			m[annots[i].Image] = vals[i]
//		}
//		return m, nil
//	}
//
//	func mergeDets(m map[string]*detect.ResultSet) *detect.ResultSet {
//		vs := make([]*detect.ResultSet, 0, len(m))
//		for _, v := range m {
//			vs = append(vs, v)
//		}
//		return detect.MergeResults(vs...)
//	}
//
//	func loadTestAnnots(inriaDir, set string) ([]inria.Annot, error) {
//		// Load list of positive test image annotations.
//		posAnnots, err := loadPosAnnots(inriaDir, set)
//		if err != nil {
//			return nil, err
//		}
//
//		// Load list of negative test images.
//		negIms, err := fileutil.LoadLines(path.Join(inriaDir, set, "neg.lst"))
//		if err != nil {
//			return nil, err
//		}
//		negAnnots := imsToAnnots(negIms)
//
//		// Combine positive and negative images.
//		annots := append(posAnnots, negAnnots...)
//		// Shuffle. http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
//		for i := range annots {
//			j := rand.Intn(len(annots)-i) + i
//			annots[i], annots[j] = annots[j], annots[i]
//		}
//		return annots, nil
//	}
//
//	// Converts a list of images to a list of annotations with no rectangles.
//	// Used to test the negative set.
//	func imsToAnnots(ims []string) []inria.Annot {
//		annots := make([]inria.Annot, len(ims))
//		for i, im := range ims {
//			annots[i] = inria.Annot{im, nil}
//		}
//		return annots
//	}
//
//	type ValidateArgs struct {
//		Tmpl *detect.FeatTmpl
//		Dir  string
//		DetectOpts
//		MinInter float64
//	}
//
//	func init() {
//		dstrfn.RegisterMap("test", true, dstrfn.ConfigFunc(
//			func(annot inria.Annot, p ValidateArgs) (*detect.ResultSet, error) {
//				return testImage(p.Tmpl, annot, p.Dir, p.DetectOpts, p.MinInter)
//			},
//		))
//	}
//
//	type DetectOpts struct {
//		FeatName string
//		FeatSpec string
//		PyrStep  float64
//		MaxIOU   float64
//		Margin   int
//		LocalMax bool
//	}
//
//	// Runs detector across a single image and validates results.
//	func testImage(tmpl *detect.FeatTmpl, annot inria.Annot, dir string, opts DetectOpts, mininter float64) (*detect.ResultSet, error) {
//		im, err := loadImage(path.Join(dir, annot.Image))
//		if err != nil {
//			return nil, err
//		}
//		// Get detections.
//		phi, err := featByName(opts.FeatName, opts.FeatSpec)
//		if err != nil {
//			return nil, err
//		}
//		dets := detectImage(tmpl, im, opts.Margin, opts.PyrStep, phi, opts.LocalMax, opts.MaxIOU)
//		val := detect.ValidateMatch(dets, annot.Rects, mininter)
//		return val, nil
//	}
//
//	// Runs a single detector across a single image and returns results.
//	func detectImage(tmpl *detect.FeatTmpl, im image.Image, margin int, step float64, phi feat.Transform, localmax bool, maxiou float64) []detect.Det {
//		// Construct pyramid.
//		// Get range of scales.
//		scales := imgpyr.Scales(im.Bounds().Size(), tmpl.Size, step)
//		// Define amount and type of padding.
//		pad := feat.Pad{feat.Margin{margin, margin, margin, margin}, imsamp.Continue}
//		pyr := featpyr.NewPad(imgpyr.New(im, scales), phi, pad)
//
//		// Search feature pyramid.
//		// Options for running detector on each level.
//		detopts := detect.DetFilter{LocalMax: localmax, MinScore: math.Inf(-1)}
//		// Use intersection-over-union criteria for non-max suppression.
//		overlap := func(a, b image.Rectangle) bool {
//			return detect.IOU(a, b) > maxiou
//		}
//		// Options for non-max suppression.
//		suppropts := detect.SupprFilter{MaxNum: 0, Overlap: overlap}
//		dets := detect.Pyramid(pyr, tmpl, detopts, suppropts)
//		return dets
//	}
