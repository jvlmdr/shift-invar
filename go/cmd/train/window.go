package main

import (
	"image"

	"github.com/jvlmdr/go-cv/rimg64"
)

// WindowSet defines a set of windows in an image.
// If Bias is zero then no extra dimension is appended,
// otherwise Bias is appended to each vector.
type WindowSet struct {
	Image   *rimg64.Multi
	Size    image.Point
	Windows []image.Point
	Bias    float64
}

func (set *WindowSet) Len() int {
	return len(set.Windows)
}

func (set *WindowSet) AddBias() bool {
	return set.Bias != 0
}

func (set *WindowSet) Dim() int {
	n := set.Size.X * set.Size.Y * set.Image.Channels
	if set.AddBias() {
		n += 1
	}
	return n
}

// At returns a copy of the window.
func (set *WindowSet) At(i int) []float64 {
	w := set.Windows[i]
	x := make([]float64, 0, set.Dim())
	for u := 0; u < set.Size.X; u++ {
		for v := 0; v < set.Size.Y; v++ {
			for p := 0; p < set.Image.Channels; p++ {
				x = append(x, set.Image.At(w.X+u, w.Y+v, p))
			}
		}
	}
	if set.AddBias() {
		x = append(x, set.Bias)
	}
	return x
}
