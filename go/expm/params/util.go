package main

import (
	"image"
	"math/rand"
	"os"
	"sort"
)

func min64(a, b int64) int64 {
	if b < a {
		return b
	}
	return a
}

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

func mod(x, n int) int {
	return ((x % n) + n) % n
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

func union(x ...[]string) []string {
	m := make(map[string]int)
	for _, xi := range x {
		for _, xij := range xi {
			m[xij] += 1
		}
	}
	var s []string
	for k := range m {
		s = append(s, k)
	}
	sort.Strings(s)
	return s
}

func loadImage(fname string) (image.Image, error) {
	file, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	im, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	return im, nil
}

func loadImageSize(name string) (image.Point, error) {
	file, err := os.Open(name)
	if err != nil {
		return image.ZP, err
	}
	defer file.Close()
	cfg, _, err := image.DecodeConfig(file)
	if err != nil {
		return image.ZP, err
	}
	return image.Pt(cfg.Width, cfg.Height), nil
}
