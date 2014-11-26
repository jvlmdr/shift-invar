package circcov

import (
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-fftw/fftw"
	"github.com/jvlmdr/go-whog/whog"

	"math/cmplx"
)

func Mul(g *whog.Covar, f *rimg64.Multi) *rimg64.Multi {
	return MulMode(g, f, Convex)
}

func MulMode(g *whog.Covar, f *rimg64.Multi, coeffs CoeffsFunc) *rimg64.Multi {
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
		fHat[p] = dftImage(f.Channel(p), f.Width, f.Height)
	}
	xHat := make([]*fftw.Array2, c)
	for p := 0; p < c; p++ {
		xHat[p] = fftw.NewArray2(f.Width, f.Height)
	}

	N := complex(float64(f.Width)*float64(f.Height), 0)
	for u := 0; u < f.Width; u++ {
		for v := 0; v < f.Height; v++ {
			for p := 0; p < c; p++ {
				for q := 0; q < c; q++ {
					delta := cmplx.Conj(gHat[p][q].At(u, v)) * fHat[q].At(u, v) / N
					xHat[p].Set(u, v, xHat[p].At(u, v)+delta)
				}
			}
		}
	}

	x := rimg64.NewMulti(f.Width, f.Height, c)
	for p := 0; p < c; p++ {
		x.SetChannel(p, idftImage(xHat[p], f.Width, f.Height))
	}
	return x
}
