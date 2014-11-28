package toepcov

import (
	"fmt"
	"log"
	"math"
	"math/cmplx"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-fftw/fftw"
)

// Copies one channel of an image into an FFT array.
// The image is copied into the top-left corner.
// Any extra space is filled with zeros.
func copyChannelTo(dst *fftw.Array2, src *rimg64.Multi, channel int) {
	m, n := dst.Dims()
	w, h := src.Width, src.Height
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var val complex128
			if i < w && j < h {
				val = complex(src.At(i, j, channel), 0)
			}
			dst.Set(i, j, val)
		}
	}
}

// Copies the real part of an FFT array into one channel of an image.
// The image is copied into the top-left corner.
// Any extra space is filled with zeros.
func copyRealToChannel(dst *rimg64.Multi, channel int, src *fftw.Array2) {
	m, n := src.Dims()
	for i := 0; i < dst.Width; i++ {
		for j := 0; j < dst.Height; j++ {
			var val float64
			if i < m && j < n {
				val = real(src.At(i, j))
			}
			dst.Set(i, j, channel, val)
		}
	}
}

// Copies an image into an FFT array and computes the forward transform.
//
// The image is copied into the top-left corner.
// Any extra space is filled with zeros.
func dftChannel(src *rimg64.Multi, channel int, m, n int) *fftw.Array2 {
	dst := fftw.NewArray2(m, n)
	copyChannelTo(dst, src, channel)
	fftw.FFT2To(dst, dst)
	return dst
}

// Takes the 2D inverse FFT and copies the result out to an image.
//
// The image is copied from the top-left corner.
func idftToChannel(dst *rimg64.Multi, channel int, src *fftw.Array2) {
	// Inverse transform in-place.
	plan := fftw.NewPlan2(src, src, fftw.Backward, fftw.Estimate)
	defer plan.Destroy()
	plan.Execute()
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

// Copies a channel pair (p, q) into an FFT array.
func copyCovarTo(dst *fftw.Array2, g *Covar, p, q, bx, by int) {
	m, n := dst.Dims()
	if 2*bx+1 > m || 2*by+1 > n {
		panic(fmt.Sprintf("bandwidth too large (bx %d, by %d, m %d, n %d)", bx, by, m, n))
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			dst.Set(i, j, 0)
		}
	}
	for u := -bx; u <= bx; u++ {
		for v := -by; v <= by; v++ {
			i := (u + m) % m
			j := (v + n) % n
			dst.Set(i, j, complex(g.At(u, v, p, q), 0))
		}
	}
}

func scale(alpha complex128, x *fftw.Array2) {
	m, n := x.Dims()
	for u := 0; u < m; u++ {
		for v := 0; v < n; v++ {
			x.Set(u, v, alpha*x.At(u, v))
		}
	}
}

// z(u, v) <- x(u, v) * y(u, v) for all u, v.
func mul(z, x, y *fftw.Array2) {
	m, n := z.Dims()
	for u := 0; u < m; u++ {
		for v := 0; v < n; v++ {
			z.Set(u, v, x.At(u, v)*y.At(u, v))
		}
	}
}

// z(u, v) <- z(u, v) + x(u, v)*y(u, v) for all u, v.
func addMul(z, x, y *fftw.Array2) {
	m, n := z.Dims()
	for u := 0; u < m; u++ {
		for v := 0; v < n; v++ {
			xy := x.At(u, v) * y.At(u, v)
			z.Set(u, v, z.At(u, v)+xy)
		}
	}
}

// z(u, v) <- conj(x(u, v)) * y(u, v) / mn for all u, v.
func crossCorr(z, x, y *fftw.Array2) {
	m, n := x.Dims()
	mn := float64(m * n)
	for u := 0; u < m; u++ {
		for v := 0; v < n; v++ {
			xy := cmplx.Conj(x.At(u, v)) * y.At(u, v) / complex(mn, 0)
			z.Set(u, v, xy)
		}
	}
}
