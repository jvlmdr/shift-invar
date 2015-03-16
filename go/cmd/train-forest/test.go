package main

import (
	"log"
	"time"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/slide"
	"github.com/jvlmdr/shift-invar/go/data"
)

func canTestSubset(dataset data.ImageSet, ims []string) []string {
	var subset []string
	for _, name := range ims {
		if dataset.CanTest(name) {
			subset = append(subset, name)
		}
	}
	return subset
}

func test(dataset data.ImageSet, ims []string, scorer slide.Scorer, region detect.PadRect, opts detect.MultiScaleOpts, minMatchIOU, minIgnoreCover float64) ([]*detect.ValSet, error) {
	imvals := make([]*detect.ValSet, len(ims))
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
		dets, durSearch, err := detect.MultiScale(im, scorer, region, opts)
		if err != nil {
			return nil, err
		}
		annot := dataset.Annot(name)
		imval := detect.Validate(dets, annot.Instances, annot.Ignore, minMatchIOU, minIgnoreCover)
		imvals[i] = imval.Set()
		log.Printf(
			"load %v, resize %v, feat %v, slide %v, suppr %v",
			durLoad, durSearch.Resize, durSearch.Feat, durSearch.Slide, durSearch.Suppr,
		)
	}
	return imvals, nil
}
