package circcov

import (
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-fftw/fftw"
	"github.com/jvlmdr/go-whog/whog"

	"log"
	"math"
)

// Copies an image into an FFT array and computes the forward transform.
//
// The image is copied into the top-left corner.
// Any extra space is filled with zeros.
func dftImage(src *rimg64.Image, m, n int) *fftw.Array2 {
	w, h := src.Size().X, src.Size().Y
	dst := fftw.NewArray2(m, n)
	// Copy.
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i < w && j < h {
				dst.Set(i, j, complex(src.At(i, j), 0))
			}
		}
	}
	// Forward transform in-place.
	plan := fftw.NewPlan2(dst, dst, fftw.Forward, fftw.Estimate)
	defer plan.Destroy()
	plan.Execute()
	return dst
}

// Takes the 2D inverse FFT and copies the result out to an image.
//
// The image is copied from the top-left corner.
func idftImage(src *fftw.Array2, w, h int) *rimg64.Image {
	dst := rimg64.New(w, h)
	// Inverse transform in-place.
	plan := fftw.NewPlan2(src, src, fftw.Backward, fftw.Estimate)
	defer plan.Destroy()
	plan.Execute()
	// Accumulate total real and imaginary components to check.
	var re, im float64
	for i := 0; i < w; i++ {
		for j := 0; j < h; j++ {
			a, b := real(src.At(i, j)), imag(src.At(i, j))
			re, im = re+a*a, im+b*b
			dst.Set(i, j, a)
		}
	}
	re, im = math.Sqrt(re), math.Sqrt(im)
	const eps = 1e-6
	if (re > eps && im/re > 1e-12) || (re <= eps && im > 1e-6) {
		log.Printf("significant imaginary component (real %g, imag %g)", re, im)
	}
	return dst
}

func dftCovarCirc(g *whog.Covar, m, n, p, q int, coeffs CoeffsFunc) *fftw.Array2 {
	at := func(g *whog.Covar, du, dv, p, q int) float64 {
		if abs(du) > g.Bandwidth {
			return 0
		}
		if abs(dv) > g.Bandwidth {
			return 0
		}
		return g.At(du, dv, p, q)
	}

	dst := fftw.NewArray2(m, n)
	for du := 0; du < m; du++ {
		for dv := 0; dv < n; dv++ {
			a, b := coeffs(du, dv, m, n)
			var h float64
			h += (1 - a) * (1 - b) * at(g, mod(du, m), mod(dv, n), p, q)
			h += (1 - a) * b * at(g, mod(du, m), -mod(-dv, n), p, q)
			h += a * (1 - b) * at(g, -mod(-du, m), mod(dv, n), p, q)
			h += a * b * at(g, -mod(-du, m), -mod(-dv, n), p, q)
			dst.Set(du, dv, complex(h, 0))
		}
	}
	plan := fftw.NewPlan2(dst, dst, fftw.Forward, fftw.Estimate)
	defer plan.Destroy()
	plan.Execute()
	return dst
}
