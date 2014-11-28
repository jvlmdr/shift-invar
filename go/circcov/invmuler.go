package circcov

import (
	"fmt"
	"math/cmplx"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-fftw/fftw"
	"github.com/jvlmdr/lin-go/clap"
	"github.com/jvlmdr/lin-go/cmat"
	"github.com/jvlmdr/shift-invar/go/toepcov"
)

// InvMuler multiplies the inverse of a circulant covariance matrix by multiple images.
// Uses the FFT and complex Cholesky factorization.
// All possible pre-computation is performed in Init.
// Stores one k x k factorization per pixel, where k is the number of channels.
type InvMuler struct {
	// Factorized matries.
	// Fact[u][v] with 0 <= u < Width, 0 <= v < Height.
	Fact [][]*clap.CholFact
	// Dimensions of image.
	Width, Height, Channels int
}

// Init does pre-computation for multiplying by the inverse covariance matrix.
// It takes k^2 transforms in O(mnk^2 log(mn)) time and
// computes mn factorizations of size kxk in O(mnk^3) time.
// Total time is O(mnk^2 (k+log(mn))).
// Returns an error if any of the complex systems are not positive definite.
func (op *InvMuler) Init(g *toepcov.Covar, w, h int) error {
	op.Width = w
	op.Height = h
	op.Channels = g.Channels

	// Compute the Fourier transform of every channel pair.
	gHat := make([][]*fftw.Array2, g.Channels)
	for p := range gHat {
		gHat[p] = make([]*fftw.Array2, g.Channels)
		for q := range gHat[p] {
			gHat[p][q] = dftCovarCirc(g, w, h, p, q, Convex)
		}
	}

	// Compute factorizations.
	op.Fact = make([][]*clap.CholFact, w)
	for u := range op.Fact {
		op.Fact[u] = make([]*clap.CholFact, h)
		for v := range op.Fact[u] {
			a := cmat.New(g.Channels, g.Channels)
			for p := 0; p < g.Channels; p++ {
				for q := 0; q < g.Channels; q++ {
					a.Set(p, q, cmplx.Conj(gHat[p][q].At(u, v)))
				}
			}
			chol, err := clap.Chol(a)
			if err != nil {
				//	log.Print(cmat.Sprintf("%8.3g", a))
				//	// Get eigenvalues.
				//	_, eigs, eigerr := clap.EigHerm(a)
				//	if eigerr != nil {
				//		log.Print(eigerr)
				//	} else {
				//		log.Println("eigenvalues:", eigs)
				//	}
				return err
			}
			op.Fact[u][v] = chol
		}
	}
	return nil
}

// Mul performs mn back-substitutions of size kxk in O(mnk^2) time and
// takes k transforms and k inverse transforms in O(mnk log(mn)) time.
// Total time is O(mnk(k+log(mn))).
func (op *InvMuler) Mul(f *rimg64.Multi) *rimg64.Multi {
	if f.Channels != op.Channels {
		panic(fmt.Sprintf(
			"bad number of channels: covar %d, image %d",
			op.Channels, f.Channels,
		))
	}
	if f.Width != op.Width || f.Height != op.Height {
		panic(fmt.Sprintf(
			"bad dimensions: operator %dx%d, image %dx%d",
			op.Width, op.Height,
			f.Width, f.Height,
		))
	}

	fHat := make([]*fftw.Array2, op.Channels)
	for p := 0; p < op.Channels; p++ {
		fHat[p] = dftChannel(f, p, f.Width, f.Height)
	}
	xHat := make([]*fftw.Array2, op.Channels)
	for p := 0; p < op.Channels; p++ {
		xHat[p] = fftw.NewArray2(f.Width, f.Height)
	}

	// Solve a channels x channels system per pixel.
	N := float64(op.Width) * float64(op.Height)
	for u := 0; u < f.Width; u++ {
		for v := 0; v < f.Height; v++ {
			y := make([]complex128, op.Channels)
			for p := 0; p < op.Channels; p++ {
				y[p] = fHat[p].At(u, v)
			}
			z, err := op.Fact[u][v].Solve(y)
			if err != nil {
				panic(err)
			}
			for p := 0; p < op.Channels; p++ {
				xHat[p].Set(u, v, complex(1/N, 0)*z[p])
			}
		}
	}

	// Take inverse transform of each channel.
	x := rimg64.NewMulti(f.Width, f.Height, op.Channels)
	for p := 0; p < op.Channels; p++ {
		idftToChannel(x, p, xHat[p])
	}
	return x
}
