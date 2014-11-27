package toepcov

import (
	"fmt"
	"image"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-cv/slide"
)

// Multiplies a stationary covariance matrix by an image via the Fourier domain.
func MulFFT(g *Covar, f *rimg64.Multi) *rimg64.Multi {
	if g.Channels != f.Channels {
		panic(fmt.Sprintf("different number of channels: covar %d, image %d", g.Channels, f.Channels))
	}

	var (
		// Input and output dimension.
		w = f.Width
		h = f.Height
		// Limit bandwidth to size of image.
		b  = g.Bandwidth
		bx = min(b, w-1)
		by = min(b, h-1)
	)
	// Working dimension in Fourier domain.
	work, _ := slide.FFT2Size(image.Pt(w+bx, h+by))
	m, n := work.X, work.Y
	// Constant for re-scaling transforms.
	N := float64(m) * float64(n)

	// Allocate output image.
	z := rimg64.NewMulti(w, h, f.Channels)

	for q := 0; q < f.Channels; q++ {
		// Take Fourier transform of channel q of f.
		fHat := dftImage(f.Channel(q), m, n)
		for p := 0; p < f.Channels; p++ {
			// Take Fourier transform of channel pair (q, p) of g.
			gHat := dftCovar(g, m, n, q, p, bx, by)
			// Multiply in Fourier domain.
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					gij := complex(1/N, 0) * gHat.At(i, j) * fHat.At(i, j)
					gHat.Set(i, j, gij)
				}
			}
			// Take inverse transform.
			z_pq := idftImage(gHat, w, h)
			// Add to result.
			for i := 0; i < w; i++ {
				for j := 0; j < h; j++ {
					z.Set(i, j, p, z.At(i, j, p)+z_pq.At(i, j))
				}
			}
		}
	}
	return z
}
