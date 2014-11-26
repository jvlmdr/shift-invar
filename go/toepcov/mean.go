package whog

import (
	"github.com/gonum/floats"
	"github.com/jvlmdr/go-cv/rimg64"
)

func clonePixel(x []float64) []float64 {
	y := make([]float64, len(x))
	copy(y, x)
	return y
}

// Panics if x and y have different number of channels.
func addPixel(x, y []float64) []float64 {
	// Ensure number of channels is consistent.
	if err := errIfNumChansNotEq(len(x), len(y)); err != nil {
		panic(err)
	}
	z := make([]float64, len(x))
	return floats.AddTo(z, x, y)
}

// Subtracts stationary mean (constant per channel) from an image.
// Panics if numbers of channels do not match.
func SubMean(x *rimg64.Multi, mu []float64) *rimg64.Multi {
	if err := errIfNumChansNotEq(x.Channels, len(mu)); err != nil {
		panic(err)
	}
	y := rimg64.NewMulti(x.Width, x.Height, x.Channels)
	for i := 0; i < x.Width; i++ {
		for j := 0; j < x.Height; j++ {
			for k := 0; k < x.Channels; k++ {
				y.Set(i, j, k, x.At(i, j, k)-mu[k])
			}
		}
	}
	return y
}

func ConstImage(width, height int, x []float64) *rimg64.Multi {
	f := rimg64.NewMulti(width, height, len(x))
	for i := 0; i < f.Width; i++ {
		for j := 0; j < f.Height; j++ {
			for k := 0; k < f.Channels; k++ {
				f.Set(i, j, k, x[k])
			}
		}
	}
	return f
}
