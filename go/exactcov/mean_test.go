package exactcov

import (
	"math"
	"testing"

	"github.com/jvlmdr/shift-invar/go/imcov"
)

func TestMeanSum_vsNaive(t *testing.T) {
	// Dimension of image.
	M, N, k := 20, 15, 3
	// Dimension of template.
	m, n := 6, 9

	im := randImage(M, N, k)
	naive := imcov.MeanSum(im, m, n)
	fast := MeanSum(im, m, n).Export()

	for u := 0; u < m; u++ {
		for v := 0; v < n; v++ {
			for p := 0; p < k; p++ {
				want := naive.At(u, v, p)
				got := fast.At(u, v, p)
				if math.Abs(got-want) > 1e-6 {
					t.Errorf(
						"different: at (%d, %d, %d), (%d, %d, %d): want %.6g, got %.6g",
						u, v, p, want, got,
					)
				}
			}
		}
	}
}

func TestMean_Subset(t *testing.T) {
	// Dimension of image.
	width, height, k := 20, 15, 3
	// Dimension of larger template.
	M, N := 6, 9
	// Dimension of subset.
	m, n := 5, 3

	im := randImage(width, height, k)
	naive := imcov.MeanSum(im, m, n)
	fast := MeanSum(im, M, N)
	subset := fast.Subset(m, n).Export()

	for u := 0; u < m; u++ {
		for v := 0; v < n; v++ {
			for p := 0; p < k; p++ {
				want := naive.At(u, v, p)
				got := subset.At(u, v, p)
				if math.Abs(got-want) > 1e-6 {
					t.Errorf(
						"different: at (%d, %d, %d): want %.6g, got %.6g",
						u, v, p, want, got,
					)
				}
			}
		}
	}
}

func TestMean_Plus_vsNaive(t *testing.T) {
	// Dimension of images.
	M1, N1 := 20, 15
	M2, N2 := 22, 14
	k := 3
	// Dimension of template.
	m, n := 6, 9

	im1 := randImage(M1, N1, k)
	im2 := randImage(M2, N2, k)

	naive1 := imcov.MeanSum(im1, m, n)
	naive2 := imcov.MeanSum(im2, m, n)
	naive := naive1.Plus(naive2)

	fast1 := MeanSum(im1, m, n)
	fast2 := MeanSum(im2, m, n)
	fast := fast1.Plus(fast2).Export()

	for u := 0; u < m; u++ {
		for v := 0; v < n; v++ {
			for p := 0; p < k; p++ {
				want := naive.At(u, v, p)
				got := fast.At(u, v, p)
				if math.Abs(got-want) > 1e-6 {
					t.Errorf(
						"different: at (%d, %d, %d), (%d, %d, %d): want %.6g, got %.6g",
						u, v, p, want, got,
					)
				}
			}
		}
	}
}
