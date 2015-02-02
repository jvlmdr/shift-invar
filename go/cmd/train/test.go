package main

import (
	"log"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
)

func test(tmpl *detect.FeatTmpl, ims []string, dataset Dataset, phi feat.Image, opts detect.MultiScaleOpts, minMatchIOU, minIgnoreCover float64, fppis []float64) (float64, error) {
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
	// Reduce set of validations to a number.
	perf := avgMissRate(valset, fppis)
	return perf, nil
}
