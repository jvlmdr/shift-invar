package data

import (
	"image"
	"math/rand"

	"github.com/jvlmdr/go-cv/feat"
)

// RandomWindows returns a set of rectangles in the listed images.
// The windows are distributed uniformly over images.
// If an image is smaller than the window, then it is skipped and
// the number of windows returned may be less than n.
// The size is specified in pixels.
func RandomWindows(n int, ims []string, dataset ImageSet, margin feat.Margin, size image.Point) (map[string][]image.Rectangle, error) {
	// Count number of rectangles to take from each image.
	// Avoid opening same image twice.
	counts := make([]int, len(ims))
	for i := 0; i < n; i++ {
		// Uniform distribution over set of images.
		counts[rand.Intn(len(ims))]++
	}

	rects := make(map[string][]image.Rectangle)
	for i, count := range counts {
		if count == 0 {
			continue
		}
		im := ims[i]
		imsize, err := loadImageSize(dataset.File(im))
		if err != nil {
			return nil, err
		}
		lims := margin.AddTo(image.Rect(0, 0, imsize.X, imsize.Y))
		if lims.Dx() < size.X || lims.Dy() < size.Y {
			// Image cannot contain window.
			continue
		}
		for j := 0; j < count; j++ {
			x := rand.Intn(lims.Dx() - size.X + 1)
			y := rand.Intn(lims.Dy() - size.Y + 1)
			r := image.Rect(x, y, x+size.X, y+size.Y)
			rects[im] = append(rects[im], r)
		}
	}
	return rects, nil
}
