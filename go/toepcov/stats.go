package toepcov

import (
	"image"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-cv/slide"
	"github.com/jvlmdr/go-fftw/fftw"
)

// Stats accumulates stationary statistics over a single image.
func Stats(f *rimg64.Multi, band int) *Total {
	mean := MeanSum(f)
	cov := CovarSumFFT(f, band)
	cnt := CovarCount(f.Width, f.Height, band)
	return &Total{mean, cov, cnt, 1}
}

// MeanSum computes the un-normalized mean (the sum) over all pixels in an image.
func MeanSum(f *rimg64.Multi) []float64 {
	x := make([]float64, f.Channels)
	for i := 0; i < f.Width; i++ {
		for j := 0; j < f.Height; j++ {
			for k := 0; k < f.Channels; k++ {
				x[k] += f.At(i, j, k)
			}
		}
	}
	return x
}

// CovarSumNaive computes the un-normalized covariance sum.
// b is the maximum bandwidth.
func CovarSumNaive(f *rimg64.Multi, b int) *Covar {
	cov := NewCovar(f.Channels, b)
	bnds := image.Rect(0, 0, f.Width, f.Height)
	for u := 0; u < f.Width; u++ {
		for v := 0; v < f.Height; v++ {
			near := image.Rect(u-b, v-b, u+b+1, v+b+1)
			r := bnds.Intersect(near)

			for i := r.Min.X; i < r.Max.X; i++ {
				du := i - u
				for j := r.Min.Y; j < r.Max.Y; j++ {
					dv := j - v

					for p := 0; p < f.Channels; p++ {
						for q := 0; q < f.Channels; q++ {
							ff := f.At(u, v, p) * f.At(i, j, q)
							cov.Set(du, dv, p, q, cov.At(du, dv, p, q)+ff)
						}
					}
				}
			}
		}
	}
	return cov
}

// CovarSumFFT computes the un-normalized covariance sum.
// b is the maximum bandwidth.
func CovarSumFFT(f *rimg64.Multi, b int) *Covar {
	// Cap x- and y-bandwidth at the max relative displacement.
	bx := min(b, f.Width-1)
	by := min(b, f.Height-1)
	// Set padded dimensions.
	min := image.Pt(f.Width+bx, f.Height+by)
	work, _ := slide.FFT2Size(min)

	fHat := make([]*fftw.Array2, f.Channels)
	for p := 0; p < f.Channels; p++ {
		fHat[p] = dftChannel(f, p, work.X, work.Y)
	}
	gHat := fftw.NewArray2(work.X, work.Y)
	gInv := fftw.NewPlan2(gHat, gHat, fftw.Backward, fftw.Estimate)
	defer gInv.Destroy()

	cov := NewCovar(f.Channels, b)
	// Iterate over ordered pairs (p, q).
	for p := 0; p < f.Channels; p++ {
		for q := p; q < f.Channels; q++ {
			// Compute cross-correlation and put result in gHat.
			crossCorr(gHat, fHat[p], fHat[q])
			// Take inverse transform in-place.
			gInv.Execute()
			setCovarCrossCorr(cov, gHat, p, q, bx, by)
		}
	}
	return cov
}

// cov(u, v, p, q) <- real(g(u, v)) for all u, v.
func setCovarCrossCorr(cov *Covar, g *fftw.Array2, p, q, bx, by int) {
	m, n := g.Dims()
	for du := -bx; du <= bx; du++ {
		for dv := -by; dv <= by; dv++ {
			// Wrap around boundary.
			x := (du + m) % m
			y := (dv + n) % n
			val := real(g.At(x, y))
			cov.Set(du, dv, p, q, val)
			if p != q {
				cov.Set(-du, -dv, q, p, val)
			}
		}
	}
}

// CovarCount computes the number of occurrences of
// each displacement up to the given bandwidth in an image of size w x h.
func CovarCount(w, h, b int) *Count {
	cnt := NewCount(b)
	// Cap at max observed rel displacement.
	bx := min(b, w-1)
	by := min(b, h-1)
	// Count number of times bandwidth occurs.
	for dx := 0; dx <= bx; dx++ {
		for dy := 0; dy <= by; dy++ {
			n := int64(w-dx) * int64(h-dy)
			// Number depends on absolute dx, dy.
			cnt.Set(dx, dy, n)
			cnt.Set(dx, -dy, n)
			cnt.Set(-dx, dy, n)
			cnt.Set(-dx, -dy, n)
		}
	}
	return cnt
}
