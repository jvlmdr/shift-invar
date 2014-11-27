package toepcov

import (
	"math"
	"testing"

	"github.com/jvlmdr/go-cv/rimg64"
)

func TestExactCovarNaive(t *testing.T) {
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

	cov := ExactCovarNaive(im, m, n)
	cases := []struct {
		U, V, P int
		I, J, Q int
		Want    float64
	}{
		// Channel 0 to 0, row 0 in (u, v)
		// (u, v) = (0, 0)
		{0, 0, 0, 0, 0, 0, (1*1 + 0*0 + 2*2) + (0*0 + 3*3 + 5*5)},
		{0, 0, 0, 1, 0, 0, (1*0 + 0*2 + 2*1) + (0*3 + 3*5 + 5*0)},
		{0, 0, 0, 2, 0, 0, (1*2 + 0*1 + 2*1) + (0*5 + 3*0 + 5*0)},
		{0, 0, 0, 0, 1, 0, (1*0 + 0*3 + 2*5) + (0*2 + 3*1 + 5*0)},
		{0, 0, 0, 1, 1, 0, (1*3 + 0*5 + 2*0) + (0*1 + 3*0 + 5*3)},
		{0, 0, 0, 2, 1, 0, (1*5 + 0*0 + 2*0) + (0*0 + 3*3 + 5*2)},
		// (u, v) = (1, 0)
		{1, 0, 0, 0, 0, 0, (0*1 + 2*0 + 1*2) + (3*0 + 5*3 + 0*5)},
		{1, 0, 0, 1, 0, 0, (0*0 + 2*2 + 1*1) + (3*3 + 5*5 + 0*0)},
		{1, 0, 0, 2, 0, 0, (0*2 + 2*1 + 1*1) + (3*5 + 5*0 + 0*0)},
		{1, 0, 0, 0, 1, 0, (0*0 + 2*3 + 1*5) + (3*2 + 5*1 + 0*0)},
		{1, 0, 0, 1, 1, 0, (0*3 + 2*5 + 1*0) + (3*1 + 5*0 + 0*3)},
		{1, 0, 0, 2, 1, 0, (0*5 + 2*0 + 1*0) + (3*0 + 5*3 + 0*2)},
		// (u, v) = (2, 0)
		{2, 0, 0, 0, 0, 0, (2*1 + 1*0 + 1*2) + (5*0 + 0*3 + 0*5)},
		{2, 0, 0, 1, 0, 0, (2*0 + 1*2 + 1*1) + (5*3 + 0*5 + 0*0)},
		{2, 0, 0, 2, 0, 0, (2*2 + 1*1 + 1*1) + (5*5 + 0*0 + 0*0)},
		{2, 0, 0, 0, 1, 0, (2*0 + 1*3 + 1*5) + (5*2 + 0*1 + 0*0)},
		{2, 0, 0, 1, 1, 0, (2*3 + 1*5 + 1*0) + (5*1 + 0*0 + 0*3)},
		{2, 0, 0, 2, 1, 0, (2*5 + 1*0 + 1*0) + (5*0 + 0*3 + 0*2)},

		// Channel 0 to 0, row 1 in (u, v)
		// (u, v) = (0, 1)
		{0, 1, 0, 0, 0, 0, (0*1 + 3*0 + 5*2) + (2*0 + 1*3 + 0*5)},
		{0, 1, 0, 1, 0, 0, (0*0 + 3*2 + 5*1) + (2*3 + 1*5 + 0*0)},
		{0, 1, 0, 2, 0, 0, (0*2 + 3*1 + 5*1) + (2*5 + 1*0 + 0*0)},
		{0, 1, 0, 0, 1, 0, (0*0 + 3*3 + 5*5) + (2*2 + 1*1 + 0*0)},
		{0, 1, 0, 1, 1, 0, (0*3 + 3*5 + 5*0) + (2*1 + 1*0 + 0*3)},
		{0, 1, 0, 2, 1, 0, (0*5 + 3*0 + 5*0) + (2*0 + 1*3 + 0*2)},
		// (u, v) = (1, 1)
		{1, 1, 0, 0, 0, 0, (3*1 + 5*0 + 0*2) + (1*0 + 0*3 + 3*5)},
		{1, 1, 0, 1, 0, 0, (3*0 + 5*2 + 0*1) + (1*3 + 0*5 + 3*0)},
		{1, 1, 0, 2, 0, 0, (3*2 + 5*1 + 0*1) + (1*5 + 0*0 + 3*0)},
		{1, 1, 0, 0, 1, 0, (3*0 + 5*3 + 0*5) + (1*2 + 0*1 + 3*0)},
		{1, 1, 0, 1, 1, 0, (3*3 + 5*5 + 0*0) + (1*1 + 0*0 + 3*3)},
		{1, 1, 0, 2, 1, 0, (3*5 + 5*0 + 0*0) + (1*0 + 0*3 + 3*2)},
		// (u, v) = (2, 1)
		{2, 1, 0, 0, 0, 0, (5*1 + 0*0 + 0*2) + (0*0 + 3*3 + 2*5)},
		{2, 1, 0, 1, 0, 0, (5*0 + 0*2 + 0*1) + (0*3 + 3*5 + 2*0)},
		{2, 1, 0, 2, 0, 0, (5*2 + 0*1 + 0*1) + (0*5 + 3*0 + 2*0)},
		{2, 1, 0, 0, 1, 0, (5*0 + 0*3 + 0*5) + (0*2 + 3*1 + 2*0)},
		{2, 1, 0, 1, 1, 0, (5*3 + 0*5 + 0*0) + (0*1 + 3*0 + 2*3)},
		{2, 1, 0, 2, 1, 0, (5*5 + 0*0 + 0*0) + (0*0 + 3*3 + 2*2)},

		// TODO: Channel pairs (0, 1) and (1, 1).
	}

	for _, e := range cases {
		got := cov.At(e.U, e.V, e.P, e.I, e.J, e.Q)
		if math.Abs(got-e.Want) > 1e-6 {
			t.Errorf(
				"different: at (%d, %d, %d), (%d, %d, %d): want %.6g, got %.6g",
				e.U, e.V, e.P, e.I, e.J, e.Q, e.Want, got,
			)
		}
	}
}

func TestExactCovarOf_vsNaive(t *testing.T) {
	// Dimension of image.
	M, N := 20, 15
	// Dimension of template.
	m, n := 6, 9
	// Number of channels.
	k := 3
	im := randImage(M, N, k)

	naive := ExactCovarNaive(im, m, n)
	fast := ExactCovarOf(im, m, n, max(m, n)-1).Export()

	testFullCovarEq(t, naive, fast)
}

func TestExactCovar_Subset(t *testing.T) {
	// Dimension of image.
	width, height, k := 20, 15, 3
	// Dimension of larger template.
	M, N := 6, 9
	// Dimension of subset.
	m, n := 5, 3

	im := randImage(width, height, k)
	naive := ExactCovarNaive(im, m, n)
	fast := ExactCovarOf(im, M, N, max(M, N)-1)
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

func TestExactCovar_Plus_vsNaive(t *testing.T) {
	// Dimension of images.
	M1, N1 := 20, 15
	M2, N2 := 22, 14
	k := 3
	// Dimension of template.
	m, n := 6, 9

	im1 := randImage(M1, N1, k)
	im2 := randImage(M2, N2, k)

	naive1 := ExactCovarNaive(im1, m, n)
	naive2 := ExactCovarNaive(im2, m, n)
	naive := naive1.Plus(naive2)

	band := max(m, n) - 1
	fast1 := ExactCovarOf(im1, m, n, band)
	fast2 := ExactCovarOf(im2, m, n, band)
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
