package main

import (
	"fmt"
	"log"
	"sort"

	"github.com/gonum/floats"
	"github.com/jvlmdr/go-cv/detect"
)

func avgMissRate(valset *detect.ValSet, fppis []float64) float64 {
	// Construct cumulative count of true positives
	// as threshold decreases.
	n := len(valset.Dets)
	falsePos := make([]int, n+1)
	for i, det := range valset.Dets {
		if det.True {
			falsePos[i+1] = falsePos[i]
		} else {
			falsePos[i+1] = falsePos[i] + 1
		}
	}
	// Number of detections that were positive at any threshold.
	var posDets int
	for _, det := range valset.Dets {
		if det.True {
			posDets++
		}
	}

	missRates := make([]float64, len(fppis))
	for i, fppi := range fppis {
		// Find operating point at which false positives
		// are not more than number given.
		// Obtain absolute number of false positives.
		// Largest integer such that maxFalsePos / numImages <= fppi.
		maxFalsePos := int(fppi * float64(valset.Images))
		// This may be zero, in which case there are
		// not enough images to measure miss rate at this FPPI.
		if maxFalsePos == 0 {
			panic(fmt.Sprintf("not enough images: fppi %g, images %d", fppi, valset.Images))
		}
		// Note that if the false positive is followed by true positives,
		// then there will be several points with the same FPPI.
		// falsePos[i] is the number of false positives in Dets[0, ..., i-1].
		// Want to find the largest i such that
		//   falsePos[i] <= maxFalsePos
		// and then use detections from Dets[0, ..., i-1].
		// This is equivalent to the smallest i such that
		//   falsePos[i+1] > maxFalsePos.
		// Check upper boundary:
		// The greatest i that Search() will test is i = n-1:
		//   falsePos[n] > maxFalsePos.
		// We should use Dets[0, ..., n-1] if
		//   falsePos[n] <= maxFalsePos.
		l := sort.Search(n, func(i int) bool { return falsePos[i+1] > maxFalsePos })

		// Number of true detections in Dets[0, ..., l-1].
		truePos := l - falsePos[l]
		// Number of true detections in Dets[l, ..., n-1].
		falseNeg := posDets - truePos
		// Miss rate is false negatives / number of actual positives.
		missRate := float64(falseNeg+valset.Misses) / float64(posDets+valset.Misses)
		log.Printf("fppi %.3g, false pos %d, num dets %d, miss rate %.3g", fppi, maxFalsePos, l, missRate)
		missRates[i] = missRate
	}
	return floats.Sum(missRates) / float64(len(missRates))
}
