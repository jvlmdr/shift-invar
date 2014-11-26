package whog

import (
	"math"
	"testing"

	"github.com/jvlmdr/go-cv/rimg64"
)

func TestExactMeanNaive(t *testing.T) {
	M, N, k := 5, 3, 2
	m, n := 3, 2

	im := rimg64.NewMulti(M, N, k)
	// Row 0, channel 0.
	im.Set(0, 0, 0, 1)
	im.Set(1, 0, 0, 0)
	im.Set(2, 0, 0, 2)
	im.Set(3, 0, 0, 1)
	im.Set(4, 0, 0, 1)
	// Row 1, channel 0.
	im.Set(0, 1, 0, 0)
	im.Set(1, 1, 0, 3)
	im.Set(2, 1, 0, 5)
	im.Set(3, 1, 0, 0)
	im.Set(4, 1, 0, 0)
	// Row 2, channel 0.
	im.Set(0, 2, 0, 2)
	im.Set(1, 2, 0, 1)
	im.Set(2, 2, 0, 0)
	im.Set(3, 2, 0, 3)
	im.Set(4, 2, 0, 2)

	// Row 0, channel 1.
	im.Set(0, 0, 1, 2)
	im.Set(1, 0, 1, 0)
	im.Set(2, 0, 1, 3)
	im.Set(3, 0, 1, 5)
	im.Set(4, 0, 1, 5)
	// Row 1, channel 1.
	im.Set(0, 1, 1, 1)
	im.Set(1, 1, 1, 2)
	im.Set(2, 1, 1, 4)
	im.Set(3, 1, 1, 0)
	im.Set(4, 1, 1, 1)
	// Row 2, channel 1.
	im.Set(0, 2, 1, 4)
	im.Set(1, 2, 1, 0)
	im.Set(2, 2, 1, 5)
	im.Set(3, 2, 1, 5)
	im.Set(4, 2, 1, 4)

	cov := ExactMeanNaive(im, m, n)
	cases := []struct {
		U, V, P int
		Want    float64
	}{
		// Channel 0
		{0, 0, 0, (1 + 0 + 2) + (0 + 3 + 5)},
		{1, 0, 0, (0 + 2 + 1) + (3 + 5 + 0)},
		{2, 0, 0, (2 + 1 + 1) + (5 + 0 + 0)},
		{0, 1, 0, (0 + 3 + 5) + (2 + 1 + 0)},
		{1, 1, 0, (3 + 5 + 0) + (1 + 0 + 3)},
		{2, 1, 0, (5 + 0 + 0) + (0 + 3 + 2)},
		// Channel 1
		{0, 0, 1, (2 + 0 + 3) + (1 + 2 + 4)},
		{1, 0, 1, (0 + 3 + 5) + (2 + 4 + 0)},
		{2, 0, 1, (3 + 5 + 5) + (4 + 0 + 1)},
		{0, 1, 1, (1 + 2 + 4) + (4 + 0 + 5)},
		{1, 1, 1, (2 + 4 + 0) + (0 + 5 + 5)},
		{2, 1, 1, (4 + 0 + 1) + (5 + 5 + 4)},
	}

	for _, e := range cases {
		got := cov.At(e.U, e.V, e.P)
		if math.Abs(got-e.Want) > 1e-6 {
			t.Errorf(
				"different: at (%d, %d, %d): want %.6g, got %.6g",
				e.U, e.V, e.P, e.Want, got,
			)
		}
	}
}

func TestExactMeanOf_vsNaive(t *testing.T) {
	// Dimension of image.
	M, N, k := 20, 15, 3
	// Dimension of template.
	m, n := 6, 9

	im := randImage(M, N, k)
	naive := ExactMeanNaive(im, m, n)
	fast := ExactMeanOf(im, m, n).Export()

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

func TestExactMean_Subset(t *testing.T) {
	// Dimension of image.
	width, height, k := 20, 15, 3
	// Dimension of larger template.
	M, N := 6, 9
	// Dimension of subset.
	m, n := 5, 3

	im := randImage(width, height, k)
	naive := ExactMeanNaive(im, m, n)
	fast := ExactMeanOf(im, M, N)
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

func TestExactMean_Plus_vsNaive(t *testing.T) {
	// Dimension of images.
	M1, N1 := 20, 15
	M2, N2 := 22, 14
	k := 3
	// Dimension of template.
	m, n := 6, 9

	im1 := randImage(M1, N1, k)
	im2 := randImage(M2, N2, k)

	naive1 := ExactMeanNaive(im1, m, n)
	naive2 := ExactMeanNaive(im2, m, n)
	naive := plus(naive1, naive2)

	fast1 := ExactMeanOf(im1, m, n)
	fast2 := ExactMeanOf(im2, m, n)
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
