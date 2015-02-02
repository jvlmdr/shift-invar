package main

import (
	"fmt"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-svm/svm"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/jvlmdr/shift-invar/go/vecset"
	"github.com/nfnt/resize"
)

type SVMOpts struct {
}

func trainSVM(trainData *data.TrainingSet, dataset data.ImageSet, phi feat.Image, bias float64, region detect.PadRect, addFlip bool, cpos, cneg, lambda float64, interp resize.InterpolationFunction) (*rimg64.Multi, float64, error) {
	// Load training data.
	// Size of feature image.
	size := phi.Size(region.Size)
	// Positive examples are extracted and stored as vectors.
	pos, err := data.PosExamples(trainData.PosImages, trainData.PosRects, dataset, phi, bias, region, addFlip, interp)
	if err != nil {
		return nil, 0, err
	}
	if len(pos) == 0 {
		return nil, 0, fmt.Errorf("empty positive set")
	}
	// Negative examples are represented as indices into an image.
	neg, err := data.NegWindowSets(trainData.NegImages, dataset, phi, bias, region, interp)
	if err != nil {
		return nil, 0, err
	}
	if len(neg) == 0 {
		return nil, 0, fmt.Errorf("empty negative set")
	}
	var numNeg int
	for i := range neg {
		numNeg += neg[i].Len()
	}

	var (
		x []vecset.Set
		y []float64
		c []float64
	)
	// Add positive examples as a set of vectors.
	x = append(x, vecset.Slice(pos))
	for _ = range pos {
		y = append(y, 1)
		c = append(c, cpos/lambda/float64(len(pos)))
	}
	// Add each set of negative vectors.
	for i := range neg {
		x = append(x, neg[i])
		ni := neg[i].Len()
		// Labels and costs for every positive and negative example.
		for j := 0; j < ni; j++ {
			y = append(y, -1)
			c = append(c, cneg/lambda/float64(numNeg))
		}
	}

	w, err := svm.Train(vecset.NewUnion(x), y, c,
		func(epoch int, f, fPrev, g, gPrev float64, w, wPrev []float64, a, aPrev map[int]float64) (bool, error) {
			if epoch < 4 {
				return false, nil
			}
			return true, nil
		},
	)
	if err != nil {
		return nil, 0, err
	}

	channels := phi.Channels()
	tmpl := &rimg64.Multi{
		Width:    size.X,
		Height:   size.Y,
		Channels: channels,
		// Exclude bias term if present.
		Elems: w[:size.X*size.Y*channels],
	}
	var b float64
	if bias != 0 {
		b = bias * w[size.X*size.Y*channels]
	}
	return tmpl, b, nil
}
