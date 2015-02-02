package main

import (
	"log"

	"github.com/gonum/floats"
	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/shift-invar/go/data"
)

func test(tmpl *detect.FeatTmpl, ims []string, dataset data.ImageSet, phi feat.Image, opts detect.MultiScaleOpts, minMatchIOU, minIgnoreCover float64, fppis []float64) (float64, error) {
	var imvals []*detect.ValSet
	for _, name := range ims {
		// Load image.
		file := dataset.File(name)
		im, err := loadImage(file)
		if err != nil {
			log.Printf("load test image: %s, error: %v", file, err)
			continue
		}
		dets, err := detect.MultiScale(im, tmpl, opts)
		if err != nil {
			return 0, err
		}
		annot := dataset.Annot(name)
		imval := detect.Validate(dets, annot.Instances, annot.Ignore, minMatchIOU, minIgnoreCover)
		imvals = append(imvals, imval.Set())
	}
	valset := detect.MergeValSets(imvals...)
	// Get average miss rate.
	rates, err := detect.MissRateAtFPPIs(valset, fppis)
	if err != nil {
		return 0, err
	}
	perf := floats.Sum(rates) / float64(len(rates))
	return perf, nil
}
