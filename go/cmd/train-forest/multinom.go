package main

import (
	"math/rand"
	"sort"
)

// CumSum computes the cumulative sum.
// Returns an array whose length is len(p)+1 and
// whose i-th element is the sum from p[0] to p[i-1] inclusive.
// The first element is zero.
func CumSum(p []float64) []float64 {
	n := len(p)
	sum := make([]float64, n+1)
	for i, x := range p {
		sum[i+1] = sum[i] + x
	}
	return sum
}

// Multinom returns an index from 0 to len(ws)-1.
// The mass of index i is ws[i].
func Multinom(ws []float64) int {
	var total float64
	for _, w := range ws {
		total += w
	}
	x := rand.Float64() * total
	// Find i such that sum([:i]) <= x < sum([:i+1]).
	// Equivalent to smallest index i such that x < s[i+1].
	var sum float64
	for i, w := range ws {
		sum += w
		if x < sum {
			return i
		}
	}
	return len(ws) - 1
}

// MultinomSum returns an index from 0 to len(sum)-2.
// The mass of index i is sum[i+1] - sum[i].
// Uses binary search.
func MultinomSum(sum []float64) int {
	n := len(sum) - 1
	x := rand.Float64() * sum[n]
	// Find i such that sum[i] <= x < sum[i+1].
	// Equivalent to smallest index i such that x < sum[i+1].
	return sort.Search(n-1, func(i int) bool { return x < sum[i+1] })
}
