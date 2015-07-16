package circcov

import (
	"math/cmplx"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-fftw/fftw"
	"github.com/jvlmdr/lin-go/clap"
	"github.com/jvlmdr/lin-go/cmat"
	"github.com/jvlmdr/shift-invar/go/toepcov"
)

// InvMul calls InvMulMode using Convex to obtain the circulant matrix.
func InvMul(g *toepcov.Covar, f *rimg64.Multi) (*rimg64.Multi, error) {
	return InvMulMode(g, f, Convex)
}

// InvMulMode solves for x in S x = f where S is the circulant covariance matrix.
// The manner in which the circulant matrix is constructed is determined
// by the coefficients function.
func InvMulMode(g *toepcov.Covar, f *rimg64.Multi, coeffs CoeffsFunc) (*rimg64.Multi, error) {
	// TODO: Why is this a separate implementation to InvMuler? Less memory?
	c := g.Channels
	n := float64(f.Width * f.Height)

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

	// Solve a channels x channels system per pixel.
	for u := 0; u < f.Width; u++ {
		for v := 0; v < f.Height; v++ {
			A := cmat.New(c, c)
			for p := 0; p < c; p++ {
				for q := 0; q < c; q++ {
					A.Set(p, q, cmplx.Conj(gHat[p][q].At(u, v)))
				}
			}
			y := make([]complex128, c)
			for p := 0; p < c; p++ {
				y[p] = fHat[p].At(u, v)
			}
			z, err := clap.SolvePosDef(A, y)
			if err != nil {
				return nil, err
			}
			for p := 0; p < c; p++ {
				xHat[p].Set(u, v, complex(1/n, 0)*z[p])
			}
		}
	}

	x := rimg64.NewMulti(f.Width, f.Height, c)
	for p := 0; p < c; p++ {
		idftToChannel(x, p, xHat[p])
	}
	return x, nil
}
