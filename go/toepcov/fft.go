package whog

import (
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-fftw/fftw"

	"fmt"
	"log"
	"math"
)

func copyImageToArray(dst *fftw.Array2, src *rimg64.Multi, p int) {
	m, n := dst.Dims()
	w, h := src.Width, src.Height
	// Copy.
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i < w && j < h {
				dst.Set(i, j, complex(src.At(i, j, p), 0))
			} else {
				dst.Set(i, j, 0)
			}
		}
	}
}

// Copies an image into an FFT array and computes the forward transform.
//
// The image is copied into the top-left corner.
// Any extra space is filled with zeros.
func dftImage(src *rimg64.Image, m, n int) *fftw.Array2 {
	dst := fftw.NewArray2(m, n)
	dftImageTo(dst, src)
	return dst
}

// Copies an image into an FFT array and computes the forward transform.
//
// The image is copied into the top-left corner.
// Any extra space is filled with zeros.
func dftImageTo(dst *fftw.Array2, src *rimg64.Image) {
	m, n := dst.Dims()
	w, h := src.Size().X, src.Size().Y
	// Copy.
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i < w && j < h {
				dst.Set(i, j, complex(src.At(i, j), 0))
			} else {
				dst.Set(i, j, 0)
			}
		}
	}
	// Forward transform in-place.
	plan := fftw.NewPlan2(dst, dst, fftw.Forward, fftw.Estimate)
	defer plan.Destroy()
	plan.Execute()
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

// Copies an image into an FFT array and computes the forward transform.
func dftCovar(g *Covar, m, n, p, q, bx, by int) *fftw.Array2 {
	dst := fftw.NewArray2(m, n)
	if 2*bx+1 > m || 2*by+1 > n {
		panic(fmt.Sprintf("bandwidth too large (bx %d, by %d, m %d, n %d)", bx, by, m, n))
	}
	for u := -bx; u <= bx; u++ {
		for v := -by; v <= by; v++ {
			i := (u + m) % m
			j := (v + n) % n
			dst.Set(i, j, complex(g.At(u, v, p, q), 0))
		}
	}
	plan := fftw.NewPlan2(dst, dst, fftw.Forward, fftw.Estimate)
	defer plan.Destroy()
	plan.Execute()
	return dst
}
