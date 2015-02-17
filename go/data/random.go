package data

import (
	"image"
	"math/rand"
)

// RandomWindows returns a set of rectangles in the listed images.
// The windows are distributed uniformly over images.
// If an image is smaller than the window, then it is skipped and
// the number of windows returned may be less than n.
func RandomWindows(n int, ims []string, dataset ImageSet, size image.Point) (map[string][]image.Rectangle, error) {
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
		if imsize.X < size.X || imsize.Y < size.Y {
			// Image cannot contain window.
			continue
		}
		for j := 0; j < count; j++ {
			x := rand.Intn(imsize.X - size.X + 1)
			y := rand.Intn(imsize.Y - size.Y + 1)
			r := image.Rect(x, y, x+size.X, y+size.Y)
			rects[im] = append(rects[im], r)
		}
	}
	return rects, nil
}
