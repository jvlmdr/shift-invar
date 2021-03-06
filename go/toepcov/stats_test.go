package toepcov

import (
	"image"
	"testing"

	"github.com/jvlmdr/go-cv/rimg64"
)

func TestStats(t *testing.T) {
	const (
		band = 4
		eps  = 1e-9
	)

	f := rimg64.NewMulti(3, 2, 2)
	f.SetChannel(0, rimg64.FromRows([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}))
	f.SetChannel(1, rimg64.FromRows([][]float64{
		{7, 8, 9},
		{2, 3, 5},
	}))

	stats := Stats(f, band)
	distr := Normalize(stats, false)
	cov := distr.Covar
	cases := []struct {
		Du, Dv, P, Q int
		Want         float64
	}{
		// (du, dv) = (0, 0)
		{0, 0, 0, 0, float64(1*1+2*2+3*3+4*4+5*5+6*6) / 6},
		{0, 0, 1, 1, float64(7*7+8*8+9*9+2*2+3*3+5*5) / 6},
		{0, 0, 0, 1, float64(1*7+2*8+3*9+4*2+5*3+6*5) / 6},
		{0, 0, 1, 0, float64(1*7+2*8+3*9+4*2+5*3+6*5) / 6},
		// (1, 0)
		{1, 0, 0, 0, float64(1*2+2*3+4*5+5*6) / 4},
		{1, 0, 1, 1, float64(7*8+8*9+2*3+3*5) / 4},
		{1, 0, 0, 1, float64(1*8+2*9+4*3+5*5) / 4},
		{1, 0, 1, 0, float64(7*2+8*3+2*5+3*6) / 4},
		// (2, 0)
		{2, 0, 0, 0, float64(1*3+4*6) / 2},
		{2, 0, 1, 1, float64(7*9+2*5) / 2},
		{2, 0, 0, 1, float64(1*9+4*5) / 2},
		{2, 0, 1, 0, float64(7*3+2*6) / 2},
		// (0, 1)
		{0, 1, 0, 0, float64(1*4+2*5+3*6) / 3},
		{0, 1, 1, 1, float64(7*2+8*3+9*5) / 3},
		{0, 1, 0, 1, float64(1*2+2*3+3*5) / 3},
		{0, 1, 1, 0, float64(7*4+8*5+9*6) / 3},
		// (1, 1)
		{1, 1, 0, 0, float64(1*5+2*6) / 2},
		{1, 1, 1, 1, float64(7*3+8*5) / 2},
		{1, 1, 0, 1, float64(1*3+2*5) / 2},
		{1, 1, 1, 0, float64(7*5+8*6) / 2},
		// (2, 1)
		{2, 1, 0, 0, float64(1*6) / 1},
		{2, 1, 1, 1, float64(7*5) / 1},
		{2, 1, 0, 1, float64(1*5) / 1},
		{2, 1, 1, 0, float64(7*6) / 1},
	}
	for _, c := range cases {
		if got := cov.At(c.Du, c.Dv, c.P, c.Q); !epsEq(c.Want, got, eps) {
			t.Errorf(
				"not equal: (du, dv, p, q) = (%d, %d, %d, %d): want %.5g, got %.5g",
				c.Du, c.Dv, c.P, c.Q, c.Want, got,
			)
		}
		if got := cov.At(-c.Du, -c.Dv, c.Q, c.P); !epsEq(c.Want, got, eps) {
			t.Errorf(
				"not symmetric: (du, dv, p, q) = (%d, %d, %d, %d): want %.5g, got %.5g",
				-c.Du, -c.Dv, c.Q, c.P, c.Want, got,
			)
		}
	}
}

// Compute covariance using FFT and compare to the naive method.
func TestCovarSumFFT_vsNaive(t *testing.T) {
	const eps = 1e-9
	var (
		width     = 200
		height    = 150
		bandwidth = 40
		channels  = 4
	)
	if testing.Short() {
		t.Log("reduce size in short mode")
		width = 40
		height = 30
		bandwidth = 8
	}

	im := randImage(width, height, channels)

	var naive, fft *Covar
	durNaive := timeFunc(func() {
		naive = CovarSumNaive(im, bandwidth)
	})
	durFFT := timeFunc(func() {
		fft = CovarSumFFT(im, bandwidth)
	})

	t.Log("naive:", durNaive)
	t.Log("fft:", durFFT)

	if naive.Bandwidth != fft.Bandwidth {
		t.Fatalf("bandwidth not equal: naive %d, fft %d", naive.Bandwidth, fft.Bandwidth)
	}

	for u := -bandwidth; u <= bandwidth; u++ {
		for v := -bandwidth; v <= bandwidth; v++ {
			for p := 0; p < channels; p++ {
				for q := 0; q < channels; q++ {
					want := naive.At(u, v, p, q)
					got := fft.At(u, v, p, q)
					if !epsEq(want, got, eps) {
						t.Errorf("not equal: at (%d, %d), (%d, %d): naive %g, fft %g", u, v, p, q, want, got)
					}
				}
			}
		}
	}
}

// Compute counts without looping over every pair
// and compare to the naive method.
func TestCovarCount_vsNaive(t *testing.T) {
	var (
		width     = 200
		height    = 150
		bandwidth = 40
		channels  = 4
	)
	if testing.Short() {
		t.Log("reduce size in short mode")
		width = 40
		height = 30
		bandwidth = 8
	}

	im := rimg64.NewMulti(width, height, channels)

	naive := covarCountNaive(im, bandwidth)
	clever := CovarCount(im.Width, im.Height, bandwidth)

	if naive.Band != clever.Band {
		t.Fatalf("bandwidth not equal: naive %d, clever %d", naive.Band, clever.Band)
	}

	for u := -bandwidth; u <= bandwidth; u++ {
		for v := -bandwidth; v <= bandwidth; v++ {
			want := naive.At(u, v)
			got := clever.At(u, v)
			if want != got {
				t.Errorf("not equal: at %d, %d: naive %d, clever %d", u, v, want, got)
			}
		}
	}
}

// b is the bandwidth.
func covarCountNaive(f *rimg64.Multi, b int) *Count {
	cnt := NewCount(b)
	bnds := image.Rect(0, 0, f.Width, f.Height)
	for u := 0; u < f.Width; u++ {
		for v := 0; v < f.Height; v++ {
			near := image.Rect(u-b, v-b, u+b+1, v+b+1)
			r := bnds.Intersect(near)

			for i := r.Min.X; i < r.Max.X; i++ {
				du := i - u
				for j := r.Min.Y; j < r.Max.Y; j++ {
					dv := j - v
					cnt.Set(du, dv, cnt.At(du, dv)+1)
				}
			}
		}
	}
	return cnt
}
