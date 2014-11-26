package circcov

import (
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-fftw/fftw"
	"github.com/jvlmdr/go-whog/whog"
	"github.com/jvlmdr/lin-go/clap"
	"github.com/jvlmdr/lin-go/cmat"

	"log"
	"math/cmplx"
)

func InvMul(g *whog.Covar, f *rimg64.Multi) *rimg64.Multi {
	return InvMulMode(g, f, Convex)
}

func InvMulMode(g *whog.Covar, f *rimg64.Multi, coeffs CoeffsFunc) *rimg64.Multi {
	c := g.Channels
	N := float64(f.Width) * float64(f.Height)

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
				log.Fatalln("could not solve channels x channels problem:", err)
			}
			for p := 0; p < c; p++ {
				xHat[p].Set(u, v, complex(1/N, 0)*z[p])
			}
		}
	}

	x := rimg64.NewMulti(f.Width, f.Height, c)
	for p := 0; p < c; p++ {
		x.SetChannel(p, idftImage(xHat[p], f.Width, f.Height))
	}
	return x
}
