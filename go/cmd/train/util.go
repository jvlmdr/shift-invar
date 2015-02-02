package main

import (
	"image"
	"math/rand"
)

func rectArea(r image.Rectangle) int {
	return r.Dx() * r.Dy()
}

func flipImageX(src image.Image) image.Image {
	r := src.Bounds()
	dst := image.NewRGBA64(r)
	q := dst.Bounds()
	for j := 0; j < q.Dy(); j++ {
		for i := 0; i < q.Dx(); i++ {
			dst.Set(q.Min.X+i, q.Min.Y+j, src.At(r.Max.X-1-i, r.Min.Y+j))
		}
	}
	return dst
}

// Split divides x randomly into n groups.
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
