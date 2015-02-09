package main

import (
	"image"
	"math/rand"
	"sort"
)

func min(a, b int) int {
	if b < a {
		return b
	}
	return a
}

func max(a, b int) int {
	if b > a {
		return b
	}
	return a
}

func area(r image.Rectangle) int {
	return r.Dx() * r.Dy()
}

// Split divides x randomly into n groups.
// It is possible that one of the groups is empty.
func split(x []string, n int) [][]string {
	y := make([][]string, n)
	for _, xi := range x {
		r := rand.Intn(n)
		y[r] = append(y[r], xi)
	}
	return y
}

func mergeExcept(x [][]string, j int) []string {
	var y []string
	for i, xi := range x {
		if i == j {
			continue
		}
		y = append(y, xi...)
	}
	return y
}

// Subset returns a random length-m subset of 0..n-1.
// Elements are ordered.
func randSubset(n, m int) []int {
	p := rand.Perm(n)[:m]
	sort.Ints(p)
	return p
}

func selectSubset(x []string, ind []int) []string {
	y := make([]string, len(ind))
	for i, j := range ind {
		y[i] = x[j]
	}
	return y
}
