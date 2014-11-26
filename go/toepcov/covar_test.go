package whog

import (
	"testing"

	"github.com/jvlmdr/go-cv/rimg64"
)

func TestCovar_Matrix(t *testing.T) {
	const (
		width     = 8
		height    = 16
		bandwidth = 2
		channels  = 3
		lambda    = 1e-2
	)

	cov := randCovar(channels, bandwidth)
	want := cov.Matrix(width, height)
	got := CovarMatrix{cov, width, height}
	testMatEq(t, want, got)
}

func TestCovar_Downsample(t *testing.T) {
	const (
		channels = 4
		band     = 12
		rate     = 5
		width    = 600 // highly divisible
		height   = 479 // prime
	)

	// Make a random image.
	im := randImage(width, height, channels)

	// Estimate covariance then downsample.
	got := covarFFT(im, band).Downsample(rate)

	// Downsample and the estimate covariance.
	// Subsample image at every offset and combine them.
	var (
		sum *Covar
		cnt *Count
	)
	for i := 0; i < rate; i++ {
		for j := 0; j < rate; j++ {
			subim := downsampleImage(im, rate, i, j)
			t := covarStatsFFT(subim, band/rate)
			n := covarCounts(subim.Width, subim.Height, band/rate)
			if sum == nil {
				sum = t
				cnt = n
			} else {
				sum = AddCovar(sum, t)
				cnt = AddCount(cnt, n)
			}
		}
	}
	want := normCovar(sum, cnt)

	covarEq(t, want, got, 1e-6)
}

func downsampleImage(f *rimg64.Multi, k int, offx, offy int) *rimg64.Multi {
	// offx + k*(width-1) = f.Width - 1
	// k*width = f.Width - offx + k - 1
	width, height := (f.Width-offx+k-1)/k, (f.Height-offy+k-1)/k
	g := rimg64.NewMulti(width, height, f.Channels)
	for i := 0; i < g.Width; i++ {
		for j := 0; j < g.Height; j++ {
			for p := 0; p < f.Channels; p++ {
				g.Set(i, j, p, f.At(offx+k*i, offy+k*j, p))
			}
		}
	}
	return g
}
