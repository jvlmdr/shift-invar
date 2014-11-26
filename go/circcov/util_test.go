package circcov

import (
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-whog/whog"
	"github.com/jvlmdr/lin-go/mat"

	"fmt"
	"math"
	"math/rand"
)

const eps = 1e-9

func epsEq(want, got, eps float64) bool {
	return math.Abs(want-got) <= eps
}

func imageDimsEq(want, got *rimg64.Multi) (bool, string) {
	if want.Width != got.Width || want.Height != got.Height {
		msg := fmt.Sprintf(
			"image sizes differ: want %dx%d, got %dx%d",
			want.Width, want.Height, got.Width, got.Height,
		)
		return false, msg
	}
	if want.Channels != got.Channels {
		msg := fmt.Sprintf("image channels differ: want %d, got %d", want.Channels, got.Channels)
		return false, msg
	}
	return true, ""
}

func imagesEq(want, got *rimg64.Multi) (bool, string) {
	if eq, msg := imageDimsEq(want, got); !eq {
		return eq, msg
	}

	for i := 0; i < want.Width; i++ {
		for j := 0; j < want.Height; j++ {
			for k := 0; k < want.Channels; k++ {
				x := want.At(i, j, k)
				y := got.At(i, j, k)
				if !epsEq(x, y, eps) {
					msg := fmt.Sprintf("at (%d, %d, %d): want %.4g, got %.4g", i, j, k, x, y)
					return false, msg
				}
			}
		}
	}
	return true, ""
}

func isSymm(a mat.Const) bool {
	m, n := a.Dims()
	for i := 0; i < m; i++ {
		for j := i + 1; j < n; j++ {
			if !epsEq(a.At(i, j), a.At(j, i), eps) {
				return false
			}
		}
	}
	return true
}

// Generates random stationary covariance with appropriate symmetry.
func randCovar(channels, bandwidth int) *whog.Covar {
	// Make a random image.
	f := randImage(4*bandwidth, 4*bandwidth, channels)
	// Estimate covariance from image.
	total := whog.Stats(f, bandwidth)
	// Do not remove mean to ensure semidefinite.
	return whog.Normalize(total, false).Covar
}

func randImage(width, height, channels int) *rimg64.Multi {
	f := rimg64.NewMulti(width, height, channels)
	for i := 0; i < width; i++ {
		for j := 0; j < height; j++ {
			for k := 0; k < channels; k++ {
				f.Set(i, j, k, rand.NormFloat64())
			}
		}
	}
	return f
}
