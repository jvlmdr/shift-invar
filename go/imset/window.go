package imset

import (
	"image"

	"github.com/jvlmdr/go-cv/rimg64"
)

// WindowSet defines a set of windows in an image.
type WindowSet struct {
	Image   *rimg64.Multi // The source image.
	Size    image.Point   // The size of the windows.
	Windows []image.Point // The top-left corner of each window.
}

func (set *WindowSet) Len() int {
	return len(set.Windows)
}

func (set *WindowSet) ImageSize() image.Point {
	return set.Size
}

func (set *WindowSet) ImageChannels() int {
	return set.Image.Channels
}

// At returns a copy of the window.
func (set *WindowSet) At(i int) *rimg64.Multi {
	w := set.Windows[i]
	x := rimg64.NewMulti(set.Size.X, set.Size.Y, set.Image.Channels)
	for u := 0; u < set.Size.X; u++ {
		for v := 0; v < set.Size.Y; v++ {
			for p := 0; p < set.Image.Channels; p++ {
				x.Set(u, v, p, set.Image.At(w.X+u, w.Y+v, p))
			}
		}
	}
	return x
}
