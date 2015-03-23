package main

import (
	"log"
	"time"

	"github.com/gonum/floats"
	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/shift-invar/go/data"
)

func test(tmpl *detect.FeatTmpl, ims []string, dataset data.ImageSet, phi feat.Image, opts detect.MultiScaleOpts, minMatchIOU, minIgnoreCover float64, fppis []float64) (float64, error) {
	var subset []string
	for _, name := range ims {
		if dataset.CanTest(name) {
			subset = append(subset, name)
		}
	}
	ims = subset

	var imvals []*detect.ValSet
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
			return 0, err
		}
		annot := dataset.Annot(name)
		imval := detect.Validate(dets, annot.Instances, annot.Ignore, minMatchIOU, minIgnoreCover)
		imvals = append(imvals, imval.Set())
		log.Printf(
			"load %v, resize %v, feat %v, slide %v, suppr %v",
			durLoad, durSearch.Resize, durSearch.Feat, durSearch.Slide, durSearch.Suppr,
		)
	}
	valset := detect.MergeValSets(imvals...)
	// Get average miss rate.
	rates := detect.MissRateAtFPPIs(valset, fppis)
	for i := range rates {
		log.Printf("fppi %g, miss rate %g", fppis[i], rates[i])
	}
	perf := floats.Sum(rates) / float64(len(rates))
	return perf, nil
}
