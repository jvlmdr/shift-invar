package circcov

import (
	"fmt"
	"math/cmplx"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-fftw/fftw"
	"github.com/jvlmdr/shift-invar/go/toepcov"
)

// Muler multiplies a circulant covariance matrix by multiple images.
// Uses the FFT.
// Pre-computes all that is possible.
//
// Stores O(c^2) FFTs (c is the number of channels).
type Muler struct {
	// Fourier transform of filters corresponding to cross-channel correlations.
	GHat [][]*fftw.Array2
	// Dimensions of image.
	Width, Height, Channels int
}

// Init does pre-computation for multiplying by the covariance matrix.
// It takes k^2 transforms in O(mnk^2 log(mn)) time.
func (op *Muler) Init(g *toepcov.Covar, w, h int) {
	op.Width = w
	op.Height = h
	op.Channels = g.Channels

	// Compute the Fourier transform of every channel pair.
	op.GHat = make([][]*fftw.Array2, g.Channels)
	for p := range op.GHat {
		op.GHat[p] = make([]*fftw.Array2, g.Channels)
		for q := range op.GHat[p] {
			op.GHat[p][q] = dftCovarCirc(g, w, h, p, q, Convex)
		}
	}
}

// Mul multiplies an image by the inverse circulant covariance matrix.
// It solves mn factorizations of size kxk in O(mnk^2) time,
// and takes k transforms and k inverse transforms in O(mnk log(mn)) time.
// Total time is O(mnk(k+log(mn))).
func (op *Muler) Mul(f *rimg64.Multi) *rimg64.Multi {
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

	n := float64(f.Width * f.Height)
	for u := 0; u < f.Width; u++ {
		for v := 0; v < f.Height; v++ {
			for p := 0; p < f.Channels; p++ {
				var total complex128
				for q := 0; q < f.Channels; q++ {
					total += cmplx.Conj(op.GHat[p][q].At(u, v)) * fHat[q].At(u, v)
				}
				xHat[p].Set(u, v, total/complex(n, 0))
			}
		}
	}

	x := rimg64.NewMulti(f.Width, f.Height, f.Channels)
	for p := 0; p < f.Channels; p++ {
		idftToChannel(x, p, xHat[p])
	}
	return x
}
