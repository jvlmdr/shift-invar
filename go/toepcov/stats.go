package whog

import (
	"image"
	"math/cmplx"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-cv/slide"
	"github.com/jvlmdr/go-fftw/fftw"
)

// Accumulate stationary statistics over a single image.
func Stats(f *rimg64.Multi, band int) *Total {
	mean := meanPixel(f)
	cov := covarStatsFFT(f, band)
	cnt := covarCounts(f.Width, f.Height, band)
	return &Total{mean, cov, cnt, 1}
}

// Sum over all pixels.
func meanPixel(f *rimg64.Multi) []float64 {
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

// b is the bandwidth.
func covarStatsNaive(f *rimg64.Multi, b int) *Covar {
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

// b is the bandwidth.
func covarCountsNaive(f *rimg64.Multi, b int) *Count {
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

func covarFFT(f *rimg64.Multi, b int) *Covar {
	sum := covarStatsFFT(f, b)
	count := covarCounts(f.Width, f.Height, b)
	return normCovar(sum, count)
}

// Computes the number of pixels at each bandwidth.
func covarCounts(w, h, b int) *Count {
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

// b is the bandwidth.
func covarStatsFFT(f *rimg64.Multi, b int) *Covar {
	// Cap x- and y-bandwidth at the max relative displacement.
	bx := min(b, f.Width-1)
	by := min(b, f.Height-1)
	// Set padded dimensions.
	n := image.Pt(f.Width+bx, f.Height+by)
	n, _ = slide.FFT2Size(n)
	cov := NewCovar(f.Channels, b)

	// We could take the FFT of all k channels and store them
	// rather than re-compute for each pair.
	// But we have to compute k^2 inverse transforms anyway,
	// so it doesn't affect the asymptotic time complexity.

	// Re-use the same pair to avoid creating garbage.
	fp := fftw.NewArray2(n.X, n.Y)
	fq := fftw.NewArray2(n.X, n.Y)

	fftP := fftw.NewPlan2(fp, fp, fftw.Forward, fftw.Measure)
	defer fftP.Destroy()
	fftQ := fftw.NewPlan2(fq, fq, fftw.Forward, fftw.Measure)
	defer fftQ.Destroy()
	ifftQ := fftw.NewPlan2(fq, fq, fftw.Backward, fftw.Measure)
	defer ifftQ.Destroy()

	for p := 0; p < f.Channels; p++ {
		// Take the FFT of channel p.
		copyImageToArray(fp, f, p)
		fftP.Execute()
		// Iterate over ordered pairs (p, q).
		for q := p; q < f.Channels; q++ {
			copyImageToArray(fq, f, q)
			fftQ.Execute()
			// Compute cross-correlation and put result in fq.
			crossCorr(fq, fp, fq)
			// Take inverse transform in-place.
			ifftQ.Execute()
			setCovarCrossCorr(cov, fq, p, q, bx, by)
		}
	}
	return cov
}

// z(u, v) <- conj(x(u, v)) * y(u, v) for all u, v.
func crossCorr(z, x, y *fftw.Array2) {
	m, n := x.Dims()
	div := float64(m) * float64(n)
	for u := 0; u < m; u++ {
		for v := 0; v < n; v++ {
			xy := cmplx.Conj(x.At(u, v)) * y.At(u, v) / complex(div, 0)
			z.Set(u, v, xy)
		}
	}
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
