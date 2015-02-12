package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"math"
	"os"
	"time"

	"github.com/gonum/floats"
	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/imsamp"
	"github.com/jvlmdr/go-file/fileutil"
	"github.com/jvlmdr/go-pbs-pro/dstrfn"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/nfnt/resize"
)

func init() {
	dstrfn.RegisterMap("test", false, dstrfn.ConfigFunc(test))
}

type TestInput struct {
	Fold int
	Param
}

func (x TestInput) Hash() string {
	return fmt.Sprintf("param-%s-fold-%d", x.Param.Hash(), x.Fold)
}

func (x TestInput) TmplFile() string {
	return fmt.Sprintf("tmpl-%s.gob", x.Hash())
}

func (x TestInput) PerfFile() string {
	return fmt.Sprintf("perf-%s.json", x.Hash())
}

// Remove feat.Pad since feat.Pad.Extend cannot be marshaled.
type MultiScaleOpts struct {
	MaxScale float64
	PyrStep  float64
	Interp   resize.InterpolationFunction
	// Replace Pad with PadMargin due to functional member.
	PadMargin feat.Margin
	detect.DetFilter
	// Replace SupprFilter with SupprMaxNum due to functional member.
	SupprMaxNum int
	// Override DetFilter.MinScore.
	ExpMinScore float64
}

func test(x TestInput, foldIms [][]string, datasetName, datasetSpec string, pad int, optsMsg MultiScaleOpts, minMatchIOU, minIgnoreCover float64, fppis []float64) (float64, error) {
	fmt.Printf("%s\t%s\n", x.Param.Hash(), x.Param.ID())
	opts := detect.MultiScaleOpts{
		MaxScale:    optsMsg.MaxScale,
		PyrStep:     optsMsg.PyrStep,
		Interp:      optsMsg.Interp,
		Pad:         feat.Pad{optsMsg.PadMargin, imsamp.Continue},
		DetFilter:   optsMsg.DetFilter,
		SupprFilter: detect.SupprFilter{optsMsg.SupprMaxNum, nil},
	}
	opts.DetFilter.MinScore = math.Log(optsMsg.ExpMinScore)
	// Use Feat and Overlap from Param.
	opts.Transform = x.Feat.Transform()
	opts.Overlap = x.Overlap.Spec.Eval
	opts.Pad.Extend = imsamp.Continue

	// Load template from disk.
	tmpl := new(detect.FeatTmpl)
	if err := fileutil.LoadExt(x.TmplFile(), tmpl); err != nil {
		return 0, err
	}

	ims := foldIms[x.Fold]
	// Re-load dataset on execution host.
	dataset, err := data.Load(datasetName, datasetSpec)
	if err != nil {
		return 0, err
	}
	// Remove images from list which should not be used for testing.
	var subset []string
	for _, name := range ims {
		if dataset.CanTest(name) {
			subset = append(subset, name)
		}
	}
	ims = subset

	// Cache validated detections.
	var imvals []*detect.ValSet
	err = fileutil.Cache(&imvals, fmt.Sprintf("val-dets-%s.json", x.Hash()), func() ([]*detect.ValSet, error) {
		var imvals []*detect.ValSet // Shadow variable in parent scope.
		for i, name := range ims {
			log.Printf("test image %d / %d: %s", i+1, len(ims), name)
			// Load image.
			file := dataset.File(name)
			t := time.Now()
			im, err := loadImage(file)
			if err != nil {
				log.Printf("load test image: %s, error: %v", file, err)
				continue
			}
			durLoad := time.Since(t)
			dets, durSearch, err := detect.MultiScale(im, tmpl, opts)
			if err != nil {
				return nil, err
			}
			annot := dataset.Annot(name)
			imval := detect.Validate(dets, annot.Instances, annot.Ignore, minMatchIOU, minIgnoreCover)
			imvals = append(imvals, imval.Set())
			log.Printf(
				"load %v, resize %v, feat %v, slide %v, suppr %v",
				durLoad, durSearch.Resize, durSearch.Feat, durSearch.Slide, durSearch.Suppr,
			)
		}
		return imvals, nil
	})
	if err != nil {
		return 0, err
	}

	valset := detect.MergeValSets(imvals...)
	// Get average miss rate.
	rates := detect.MissRateAtFPPIs(valset, fppis)
	for i := range rates {
		log.Printf("fppi %g, miss rate %g", fppis[i], rates[i])
	}
	perf := floats.Sum(rates) / float64(len(rates))
	if err := fileutil.SaveExt(x.PerfFile(), perf); err != nil {
		return 0, err
	}
	return perf, nil
}

func loadImage(fname string) (image.Image, error) {
	file, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	im, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	return im, nil
}
