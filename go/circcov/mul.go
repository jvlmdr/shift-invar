package circcov

import (
	"math/cmplx"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-fftw/fftw"
	"github.com/jvlmdr/shift-invar/go/toepcov"
)

// Mul calls MulMode using Convex to obtain the circulant matrix.
func Mul(g *toepcov.Covar, f *rimg64.Multi) *rimg64.Multi {
	return MulMode(g, f, Convex)
}

// MulMode computes the product of a circulant covariance matrix with an image.
// The mode is determined by the coefficients function.
func MulMode(g *toepcov.Covar, f *rimg64.Multi, coeffs CoeffsFunc) *rimg64.Multi {
	c := g.Channels

	gHat := make([][]*fftw.Array2, c)
	for p := 0; p < c; p++ {
		gHat[p] = make([]*fftw.Array2, c)
		for q := 0; q < c; q++ {
			// Take Fourier transform of channel pair of g.
			gHat[p][q] = dftCovarCirc(g, f.Width, f.Height, p, q, coeffs)
		}
	}
	fHat := make([]*fftw.Array2, c)
	for p := 0; p < c; p++ {
		fHat[p] = dftChannel(f, p, f.Width, f.Height)
	}
	xHat := make([]*fftw.Array2, c)
	for p := 0; p < c; p++ {
		xHat[p] = fftw.NewArray2(f.Width, f.Height)
	}

	n := float64(f.Width * f.Height)
	for u := 0; u < f.Width; u++ {
		for v := 0; v < f.Height; v++ {
			for p := 0; p < c; p++ {
				var total complex128
				for q := 0; q < c; q++ {
					total += cmplx.Conj(gHat[p][q].At(u, v)) * fHat[q].At(u, v)
				}
				xHat[p].Set(u, v, total/complex(n, 0))
			}
		}
	}

	x := rimg64.NewMulti(f.Width, f.Height, c)
	for p := 0; p < c; p++ {
		idftToChannel(x, p, xHat[p])
	}
	return x
}
