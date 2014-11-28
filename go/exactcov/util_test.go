package exactcov

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/shift-invar/go/imcov"
)

const eps = 1e-9

func epsEq(want, got, eps float64) bool {
	return math.Abs(want-got) <= eps
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

func testImageDimsEq(t *testing.T, want, got *rimg64.Multi) {
	if want.Width != got.Width || want.Height != got.Height {
		t.Fatalf("image sizes differ: want %dx%d, got %dx%d", want.Width, want.Height, got.Width, got.Height)
	}
	if want.Channels != got.Channels {
		t.Fatalf("image channels differ: want %d, got %d", want.Channels, got.Channels)
	}
}

func testImageEq(t *testing.T, want, got *rimg64.Multi) {
	testImageDimsEq(t, want, got)

	for i := 0; i < want.Width; i++ {
		for j := 0; j < want.Height; j++ {
			for k := 0; k < want.Channels; k++ {
				x := want.At(i, j, k)
				y := got.At(i, j, k)
				if !epsEq(x, y, eps) {
					t.Errorf("at (%d, %d, %d): want %.4g, got %.4g", i, j, k, x, y)
				}
			}
		}
	}
}

func timeFunc(f func()) time.Duration {
	t := time.Now()
	f()
	return time.Since(t)
}

func testFullCovarEq(t *testing.T, want, got *imcov.Covar) {
	if want.Width != got.Width || want.Height != got.Height {
		t.Fatalf("sizes differ: want %dx%d, got %dx%d", want.Width, want.Height, got.Width, got.Height)
	}
	if want.Channels != got.Channels {
		t.Fatalf("numbers of channels differ: want %d, got %d", want.Channels, got.Channels)
	}

	m, n, k := want.Width, want.Height, want.Channels
	for u := 0; u < m; u++ {
		for v := 0; v < n; v++ {
			for p := 0; p < k; p++ {
				for i := 0; i < m; i++ {
					for j := 0; j < n; j++ {
						for q := 0; q < k; q++ {
							x := want.At(u, v, p, i, j, q)
							y := got.At(u, v, p, i, j, q)
							if math.Abs(y-x) > eps {
								t.Errorf(
									"different: at (%d, %d, %d), (%d, %d, %d): want %.6g, got %.6g",
									u, v, p, i, j, q, x, y,
								)
							}
						}
					}
				}
			}
		}
	}
}

func sliceEq(t *testing.T, want, got []float64, eps float64) bool {
	if len(want) != len(got) {
		t.Errorf("lengths differ: want %d, got %d", len(want), len(got))
		return false
	}
	equal := true
	for i := range want {
		if !epsEq(want[i], got[i], eps) {
			t.Errorf("at %d: want %.4g, got %.4g", i, want[i], got[i])
			equal = false
		}
	}
	return equal
}
