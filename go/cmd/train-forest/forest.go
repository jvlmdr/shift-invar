package main

import (
	"image"
	"math"
	"sort"

	"github.com/jvlmdr/go-cv/rimg64"
)

type StumpForest struct {
	Stumps    []Stump
	ImageSize image.Point
}

func (f StumpForest) Score(x *rimg64.Multi) (float64, error) {
	var score float64
	for _, stump := range f.Stumps {
		score += stump.Score(x)
	}
	score /= float64(len(f.Stumps))
	return score, nil
}

func (f StumpForest) Size() image.Point {
	return f.ImageSize
}

type Stump struct {
	Feature Feature
	Thresh  float64
	Left    float64
	Right   float64
}

func (f Stump) Score(x *rimg64.Multi) float64 {
	if f.Feature.Eval(x) <= f.Thresh {
		return f.Left
	}
	return f.Right
}

// scalar with a label
type scalar struct {
	x, y float64
}

type byValue []scalar

func (a byValue) Len() int           { return len(a) }
func (a byValue) Less(i, j int) bool { return a[i].x < a[j].x }
func (a byValue) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func TrainStumpForest(x []*rimg64.Multi, y []float64, distr FeatureDistribution, size image.Point, numTrees, numCands int) (*StumpForest, error) {
	forest := &StumpForest{ImageSize: size}
	for i := 0; i < numTrees; i++ {
		// Sample candidate features.
		cands := make([]Feature, numCands)
		for j := range cands {
			cands[j] = distr.Sample()
		}
		var stump Stump
		minCost := math.Inf(1)
		// Find the best feature by its optimal split.
		for j := range cands {
			// Evaluate feature on every example.
			vals := make([]scalar, len(x))
			for k := range x {
				vals[k] = scalar{cands[j].Eval(x[k]), y[k]}
			}
			// Sort examples based on their feature value.
			sort.Sort(byValue(vals))
			index, cost := minCostSplit(vals)
			if cost >= minCost {
				continue
			}

			thresh := (vals[index].x + vals[index+1].x) / 2
			var leftPos, leftNeg, rightPos, rightNeg int
			for k, val := range vals {
				if k <= index {
					if val.y > 0 {
						leftPos++
					} else {
						leftNeg++
					}
				} else {
					if val.y > 0 {
						rightPos++
					} else {
						rightNeg++
					}
				}
			}
			left := float64(leftPos) / float64(leftPos+leftNeg)
			right := float64(rightPos) / float64(rightPos+rightNeg)
			//	var left, right float64
			//	if leftPos > leftNeg {
			//		left = 1
			//	} else {
			//		left = -1
			//	}
			//	if rightPos > rightNeg {
			//		right = 1
			//	} else {
			//		right = -1
			//	}
			stump = Stump{Feature: cands[j], Thresh: thresh, Left: left, Right: right}
			minCost = cost
		}
		forest.Stumps = append(forest.Stumps, stump)
	}
	return forest, nil
}

// vals must be sorted from smallest to largest.
func minCostSplit(vals []scalar) (arg int, min float64) {
	if !sort.IsSorted(byValue(vals)) {
		panic("not sorted")
	}
	n := len(vals)
	min = math.Inf(1)
	// Determine number of positive and negative examples.
	var numPos, numNeg int
	for _, val := range vals {
		if val.y > 0 {
			numPos++
		} else {
			numNeg++
		}
	}
	// Initially threshold is below lowest number and
	// all examples return true.
	var leftPos, leftNeg int
	for i := 0; i < n-1; i++ {
		// Threshold is between samples i and i+1.
		// Check if value greater than threshold.
		// Output changes from true to false as threshold increases.
		if vals[i].y > 0 {
			// Positive example goes below the threshold.
			leftPos++
		} else {
			leftNeg++
		}
		// Examples up to and including i are on the left.
		numLeft, numRight := i+1, n-(i+1)
		rightPos := numPos - leftPos
		fracLeft := float64(numLeft) / float64(n)
		costLeft := entropy(float64(leftPos) / float64(numLeft))
		costRight := entropy(float64(rightPos) / float64(numRight))
		cost := fracLeft*costLeft + (1-fracLeft)*costRight
		if cost < min {
			arg, min = i, cost
		}
	}
	//fmt.Print("\n\n")
	return arg, min
}

func entropy(p float64) float64 {
	return halfEntropy(p) + halfEntropy(1-p)
}

func halfEntropy(p float64) float64 {
	if p <= 0 {
		return 0
	}
	return -p * math.Log2(p)
}
