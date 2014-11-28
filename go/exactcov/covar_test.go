package exactcov

import (
	"math"
	"testing"

	"github.com/jvlmdr/shift-invar/go/imcov"
)

func TestCovarStats_vsNaive(t *testing.T) {
	// Dimension of image.
	M, N := 20, 15
	// Dimension of template.
	m, n := 6, 9
	// Number of channels.
	k := 3
	im := randImage(M, N, k)

	naive := imcov.CovarSum(im, m, n)
	fast := CovarSum(im, m, n, max(m, n)-1).Export()

	testFullCovarEq(t, naive, fast)
}

func TestCovar_Subset(t *testing.T) {
	// Dimension of image.
	width, height, k := 20, 15, 3
	// Dimension of larger template.
	M, N := 6, 9
	// Dimension of subset.
	m, n := 5, 3

	im := randImage(width, height, k)
	naive := imcov.CovarSum(im, m, n)
	fast := CovarSum(im, M, N, max(M, N)-1)
	subset := fast.Subset(m, n, max(m, n)-1).Export()

	for u := 0; u < m; u++ {
		for v := 0; v < n; v++ {
			for p := 0; p < k; p++ {
				for i := 0; i < m; i++ {
					for j := 0; j < n; j++ {
						for q := 0; q < k; q++ {
							want := naive.At(u, v, p, i, j, q)
							got := subset.At(u, v, p, i, j, q)
							if math.Abs(got-want) > 1e-6 {
								t.Errorf(
									"different: at (%d, %d, %d), (%d, %d, %d): want %.6g, got %.6g",
									u, v, p, i, j, q, want, got,
								)
							}
						}
					}
				}
			}
		}
	}
}

func TestCovar_Plus_vsNaive(t *testing.T) {
	// Dimension of images.
	M1, N1 := 20, 15
	M2, N2 := 22, 14
	k := 3
	// Dimension of template.
	m, n := 6, 9

	im1 := randImage(M1, N1, k)
	im2 := randImage(M2, N2, k)

	naive1 := imcov.CovarSum(im1, m, n)
	naive2 := imcov.CovarSum(im2, m, n)
	naive := naive1.Plus(naive2)

	band := max(m, n) - 1
	fast1 := CovarSum(im1, m, n, band)
	fast2 := CovarSum(im2, m, n, band)
	fast := fast1.Plus(fast2).Export()

	for u := 0; u < m; u++ {
		for v := 0; v < n; v++ {
			for p := 0; p < k; p++ {
				for i := 0; i < m; i++ {
					for j := 0; j < n; j++ {
						for q := 0; q < k; q++ {
							want := naive.At(u, v, p, i, j, q)
							got := fast.At(u, v, p, i, j, q)
							if math.Abs(got-want) > 1e-6 {
								t.Errorf(
									"different: at (%d, %d, %d), (%d, %d, %d): want %.6g, got %.6g",
									u, v, p, i, j, q, want, got,
								)
							}
						}
					}
				}
			}
		}
	}
}
