package toepcov

import (
	"github.com/jvlmdr/go-cv/rimg64"

	"fmt"
)

// Multiplies a stationary covariance matrix by an image.
// Uses naive summation but does not instantiate matrix.
func MulNaive(g *Covar, f *rimg64.Multi) *rimg64.Multi {
	if g.Channels != f.Channels {
		m := fmt.Sprintf("Covariance has different number of channels (%d) to image (%d)",
			g.Channels, f.Channels)
		panic(m)
	}

	h := rimg64.NewMulti(f.Width, f.Height, f.Channels)
	for i := 0; i < f.Width; i++ {
		for j := 0; j < f.Height; j++ {
			for k := 0; k < f.Channels; k++ {
				for u := 0; u < f.Width; u++ {
					for v := 0; v < f.Height; v++ {
						for w := 0; w < f.Channels; w++ {
							if abs(u-i) <= g.Bandwidth && abs(v-j) <= g.Bandwidth {
								// Add to h.
								d := g.At(u-i, v-j, k, w) * f.At(u, v, w)
								h.Set(i, j, k, d+h.At(i, j, k))
							}
						}
					}
				}
			}
		}
	}
	return h
}
