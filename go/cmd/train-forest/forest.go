package main

import (
	"image"
	"math"
	"sort"

	"github.com/jvlmdr/go-cv/rimg64"
)

import "log"

type Forest struct {
	Trees     []*Node
	InputSize image.Point // Size of feature image input.
}

func (f Forest) Score(x *rimg64.Multi) (float64, error) {
	var score float64
	for _, tree := range f.Trees {
		score += tree.Score(x)
	}
	score /= float64(len(f.Trees))
	return score, nil
}

func (f Forest) Size() image.Point {
	return f.InputSize
}

// scalar with a label
type scalar struct {
	x, y float64
	i    int
}

type byValue []scalar

func (a byValue) Len() int           { return len(a) }
func (a byValue) Less(i, j int) bool { return a[i].x < a[j].x }
func (a byValue) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func TrainForest(x []*rimg64.Multi, y []float64, distr FeatureDistribution, size image.Point, numTrees, depth, numCands int) (*Forest, error) {
	forest := &Forest{InputSize: size}
	subset := make([]int, len(x))
	for i := range subset {
		subset[i] = i
	}
	for i := 0; i < numTrees; i++ {
		log.Println("train tree", i+1)
		tree, err := trainTree(x, y, subset, distr, depth, numCands)
		if err != nil {
			return nil, err
		}
		forest.Trees = append(forest.Trees, tree)
	}
	return forest, nil
}

func trainTree(x []*rimg64.Multi, y []float64, subset []int, distr FeatureDistribution, depth, numCands int) (*Node, error) {
	if len(subset) == 0 {
		panic("empty list")
	}
	if len(subset) == 1 {
		// Forced leaf node.
		return &Node{Value: leafValue(y, subset)}, nil
	}
	if depth == 0 {
		// Leaf node.
		return &Node{Value: leafValue(y, subset)}, nil
	}
	// Sample candidate features.
	cands := make([]Feature, numCands)
	for j := range cands {
		cands[j] = distr.Sample()
	}
	var (
		scores    = make([]scalar, len(subset)) // Re-use memory.
		opt       int
		minCost   = math.Inf(1)
		optScores = make([]scalar, len(subset))
		optSplit  int
	)
	// Find the best feature by its optimal split.
	for j, cand := range cands {
		// Evaluate feature on every example.
		// Over-write.
		for k, i := range subset {
			scores[k] = scalar{cand.Eval(x[i]), y[i], i}
		}
		// Sort examples based on their feature value.
		sort.Sort(byValue(scores))
		// Find best split for this feature.
		split, cost := minCostSplit(scores)
		if cost >= minCost {
			continue
		}
		// This is the best candidate so far.
		opt = j
		minCost = cost
		copy(optScores, scores)
		optSplit = split
	}

	thresh := (optScores[optSplit-1].x + optScores[optSplit].x) / 2
	order := make([]int, len(subset))
	for k, score := range optScores {
		order[k] = score.i
	}
	left, err := trainTree(x, y, order[:optSplit], distr, depth-1, numCands)
	if err != nil {
		return nil, err
	}
	right, err := trainTree(x, y, order[optSplit:], distr, depth-1, numCands)
	if err != nil {
		return nil, err
	}
	return &Node{Feature: cands[opt], Thresh: thresh, Left: left, Right: right}, nil
}

func leafValue(y []float64, subset []int) float64 {
	var pos int
	for _, i := range subset {
		if y[i] > 0 {
			pos++
		}
	}
	return float64(pos) / float64(len(subset))
}

// scores must be sorted from smallest to largest.
// The best split is {0, ..., arg-1}, {arg, ..., n-1}.
// There should be at least two elements.
// The result will satisfy 0 < arg < n.
func minCostSplit(scores []scalar) (arg int, min float64) {
	if !sort.IsSorted(byValue(scores)) {
		panic("not sorted")
	}
	n := len(scores)
	min = math.Inf(1)
	// Determine number of positive and negative examples.
	var numPos, numNeg int
	for _, score := range scores {
		if score.y > 0 {
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
		if scores[i].y > 0 {
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
			arg, min = i+1, cost
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
