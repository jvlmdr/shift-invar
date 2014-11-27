package toepcov

import (
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/lin-go/lapack"
	"github.com/jvlmdr/lin-go/mat"

	"math"
	"math/rand"
	"testing"
	"time"
)

const eps = 1e-9

func testMatDimsEq(t *testing.T, want, got mat.Const) {
	if !eqMatDims(want, got) {
		m, n := want.Dims()
		p, q := got.Dims()
		t.Fatalf("matrix sizes differ: want %dx%d, got %dx%d", m, n, p, q)
	}
}

func epsEq(want, got, eps float64) bool {
	return math.Abs(want-got) <= eps
}

func eqMatDims(a, b mat.Const) bool {
	m, n := a.Dims()
	p, q := b.Dims()
	return m == p && n == q
}

func testMatEq(t *testing.T, want, got mat.Const) {
	testMatDimsEq(t, want, got)

	m, n := want.Dims()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			u := want.At(i, j)
			v := got.At(i, j)
			if !epsEq(u, v, eps) {
				t.Errorf("at (%d, %d): want %.4g, got %.4g", i, j, u, v)
			}
		}
	}
}

// Generates random stationary covariance with appropriate symmetry.
func randCovar(channels, bandwidth int) *Covar {
	f := randImage(4*bandwidth, 4*bandwidth, channels)
	return covarFFT(f, bandwidth)
}

func TestRandomCovar(t *testing.T) {
	const (
		width     = 8
		height    = 16
		bandwidth = 12
		channels  = 3
		lambda    = 1e-2
	)
	cov := randCovar(channels, bandwidth)

	// Confirm that random covariance matrix is positive semi-definite.
	_, eigs, err := lapack.EigSymm(cov.Matrix(width, height))
	if err != nil {
		t.Fatal(err)
	}

	// Find max positive eigenvalue.
	var maxeig float64
	for _, eig := range eigs {
		if eig > maxeig {
			maxeig = eig
		}
	}

	const eps = 1e-6
	// Check for negative eigenvalues.
	for _, eig := range eigs {
		if eig < -eps*maxeig {
			t.Error("large negative eigenvalue:", eig)
		}
	}
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

func testFullCovarEq(t *testing.T, want, got *FullCovar) {
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

func covarEq(t *testing.T, want, got *Covar, eps float64) bool {
	if want.Channels != got.Channels {
		t.Errorf("different channels: want %d, got %d", want.Channels, got.Channels)
		return false
	}
	if want.Bandwidth != got.Bandwidth {
		t.Errorf("different bandwidth: want %d, got %d", want.Bandwidth, got.Bandwidth)
		return false
	}

	band := want.Bandwidth
	equal := true
	for du := -band; du <= band; du++ {
		for dv := -band; dv <= band; dv++ {
			for p := 0; p < want.Channels; p++ {
				for q := 0; q < want.Channels; q++ {
					x := want.At(du, dv, p, q)
					y := got.At(du, dv, p, q)
					if !epsEq(x, y, eps) {
						t.Errorf("at du %d, dv %d, p %d, q %d: want %.4g, got %.4g", du, dv, p, q, x, y)
						equal = false
					}
				}
			}
		}
	}
	return equal
}

func countEq(t *testing.T, want, got *Count) bool {
	if want.Band != got.Band {
		t.Errorf("different bandwidth: want %d, got %d", want.Band, got.Band)
		return false
	}
	equal := true
	for du := -want.Band; du <= want.Band; du++ {
		for dv := -want.Band; dv <= want.Band; dv++ {
			x := want.At(du, dv)
			y := got.At(du, dv)
			if x != y {
				t.Errorf("at du %d, dv %d: want %.4g, got %.4g", du, dv, x, y)
				equal = false
			}
		}
	}
	return equal
}
