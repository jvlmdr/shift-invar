package circcov

import (
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-fftw/fftw"
	"github.com/jvlmdr/go-whog/whog"

	"fmt"
	"math/cmplx"
)

// Multiplies a circulant covariance matrix by an image.
// Uses the FFT.
//
// Stores O(c^2) FFTs (c is the number of channels).
type Muler struct {
	// Fourier transform of filters corresponding to cross-channel correlations.
	GHat [][]*fftw.Array2
	// Dimensions of image.
	Width, Height, Channels int
}

// Takes k^2 transforms. O(mnk^2 log(mn))
func (op *Muler) Init(g *whog.Covar, w, h int) {
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

// Solves mn factorizations of size kxk. O(mnk^2)
// Takes k transforms and k inverse transforms. O(mnk log(mn))
// Total time O(mnk(k+log(mn))).
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
		fHat[p] = dftImage(f.Channel(p), f.Width, f.Height)
	}
	xHat := make([]*fftw.Array2, op.Channels)
	for p := 0; p < op.Channels; p++ {
		xHat[p] = fftw.NewArray2(f.Width, f.Height)
	}

	N := complex(float64(f.Width)*float64(f.Height), 0)
	for u := 0; u < f.Width; u++ {
		for v := 0; v < f.Height; v++ {
			for p := 0; p < f.Channels; p++ {
				for q := 0; q < f.Channels; q++ {
					delta := cmplx.Conj(op.GHat[p][q].At(u, v)) * fHat[q].At(u, v) / N
					xHat[p].Set(u, v, xHat[p].At(u, v)+delta)
				}
			}
		}
	}

	x := rimg64.NewMulti(f.Width, f.Height, f.Channels)
	for p := 0; p < f.Channels; p++ {
		x.SetChannel(p, idftImage(xHat[p], f.Width, f.Height))
	}
	return x
}
