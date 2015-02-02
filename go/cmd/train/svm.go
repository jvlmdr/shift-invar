package main

import (
	"fmt"
	"image"
	"log"
	"time"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/imsamp"
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-svm/svm"
	"github.com/nfnt/resize"
)

type SVMOpts struct {
}

func trainSVM(trainData *TrainData, dataset Dataset, phi feat.Image, bias float64, region detect.PadRect, addFlip bool, cpos, cneg, lambda float64, interp resize.InterpolationFunction) (*rimg64.Multi, float64, error) {
	// Load training data.
	// Positive examples are extracted and stored as vectors.
	// Negative examples are represented as indices into an image.
	var pos [][]float64
	for _, name := range trainData.PosImages {
		log.Println("load positive image:", name)
		t := time.Now()
		file := dataset.File(name)
		im, err := loadImage(file)
		if err != nil {
			log.Printf("load positive image: %s, error: %v", file, err)
			continue
		}
		durLoad := time.Since(t)
		rects := trainData.PosRects[name]
		var durSamp, durResize, durFlip, durFeat time.Duration
		for _, rect := range rects {
			// Extract and resize window.
			t = time.Now()
			subim := imsamp.Rect(im, rect, imsamp.Continue)
			durSamp += time.Since(t)
			t = time.Now()
			subim = resize.Resize(uint(region.Size.X), uint(region.Size.Y), subim, interp)
			durResize += time.Since(t)
			// Add flip if desired.
			flips := []bool{false}
			if addFlip {
				flips = []bool{false, true}
			}
			for _, flip := range flips {
				pix := subim
				t = time.Now()
				if flip {
					pix = flipImageX(subim)
				}
				durFlip += time.Since(t)
				t = time.Now()
				x, err := phi.Apply(pix)
				if err != nil {
					return nil, 0, err
				}
				durFeat += time.Since(t)
				vec := x.Elems
				if bias != 0 {
					vec = append(vec, bias)
				}
				pos = append(pos, vec)
			}
		}
		log.Printf(
			"load %.3gms, sample %.3gms, resize %.3gms, flip %.3gms, feat %.3gms",
			durLoad.Seconds()*1000, durSamp.Seconds()*1000, durResize.Seconds()*1000,
			durFlip.Seconds()*1000, durFeat.Seconds()*1000,
		)
	}
	if len(pos) == 0 {
		return nil, 0, fmt.Errorf("empty positive set")
	}

	// Size of feature image.
	size := phi.Size(region.Size)
	var neg []*WindowSet
	for _, name := range trainData.NegImages {
		log.Println("load negative image:", name)
		t := time.Now()
		file := dataset.File(name)
		im, err := loadImage(file)
		if err != nil {
			log.Printf("load negative image: %s, error: %v", file, err)
			continue
		}
		durLoad := time.Since(t)
		t = time.Now()
		// Take transform of entire image.
		x, err := phi.Apply(im)
		if err != nil {
			return nil, 0, err
		}
		durFeat := time.Since(t)
		set := new(WindowSet)
		set.Image = x
		set.Size = size
		for u := 0; u < x.Width-size.X+1; u++ {
			for v := 0; v < x.Height-size.Y+1; v++ {
				set.Windows = append(set.Windows, image.Pt(u, v))
			}
		}
		set.Bias = bias
		neg = append(neg, set)
		log.Printf("load %.3gms, feat %.3gms", durLoad.Seconds()*1000, durFeat.Seconds()*1000)
	}
	if len(neg) == 0 {
		return nil, 0, fmt.Errorf("empty negative set")
	}

	var numNeg int
	for i := range neg {
		numNeg += neg[i].Len()
	}

	var (
		x []svm.Set
		y []float64
		c []float64
	)
	// Add positive examples as a set of vectors.
	x = append(x, svm.Slice(pos))
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

	w, err := svm.Train(NewUnion(x), y, c,
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
