package circcov

import (
	"log"
	"math"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-fftw/fftw"
	"github.com/jvlmdr/shift-invar/go/toepcov"
)

// Copies an image into an FFT array and computes the forward transform.
//
// The image is copied into the top-left corner.
// Any extra space is filled with zeros.
func dftChannel(src *rimg64.Multi, channel int, m, n int) *fftw.Array2 {
	dst := fftw.NewArray2(m, n)
	// Copy.
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i < src.Width && j < src.Height {
				dst.Set(i, j, complex(src.At(i, j, channel), 0))
			}
		}
	}
	// Forward transform in-place.
	fftw.FFT2To(dst, dst)
	return dst
}

// Takes the 2D inverse FFT and copies the result out to an image.
//
// The image is copied from the top-left corner.
func idftToChannel(dst *rimg64.Multi, channel int, src *fftw.Array2) {
	// Inverse transform in-place.
	fftw.IFFT2To(src, src)
	// Accumulate total real and imaginary components to check.
	var re, im float64
	for i := 0; i < dst.Width; i++ {
		for j := 0; j < dst.Height; j++ {
			a, b := real(src.At(i, j)), imag(src.At(i, j))
			re, im = re+a*a, im+b*b
			dst.Set(i, j, channel, a)
		}
	}
	re, im = math.Sqrt(re), math.Sqrt(im)
	const eps = 1e-6
	if (re > eps && im/re > 1e-12) || (re <= eps && im > 1e-6) {
		log.Printf("significant imaginary component (real %g, imag %g)", re, im)
	}
}

func dftCovarCirc(g *toepcov.Covar, m, n, p, q int, coeffs CoeffsFunc) *fftw.Array2 {
	at := func(g *toepcov.Covar, du, dv, p, q int) float64 {
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
	fftw.FFT2To(dst, dst)
	return dst
}
