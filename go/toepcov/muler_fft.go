package toepcov

import (
	"fmt"
	"image"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-cv/slide"
	"github.com/jvlmdr/go-fftw/fftw"
)

// MulerFFT pre-computes all possible transforms,
// then can be used to multiply a covariance matrix
// by several images of the same size.
// Stores O(c^2) FFTs (c is the number of channels).
type MulerFFT struct {
	// Fourier transform of filters corresponding to cross-channel correlations.
	GHat [][]*fftw.Array2
	// Dimensions of image.
	Width, Height int
	// Dimensions of padded image.
	M, N int
}

// Init transforms each channel pair (p, q) of the covariance matrix.
func (op *MulerFFT) Init(g *Covar, w, h int) {
	op.Width = w
	op.Height = h
	// Limit bandwidth to size of image.
	var (
		b  = g.Bandwidth
		bx = min(b, w-1)
		by = min(b, h-1)
	)
	// Working dimension in Fourier domain.
	work, _ := slide.FFT2Size(image.Pt(w+bx, h+by))
	op.M, op.N = work.X, work.Y

	op.GHat = make([][]*fftw.Array2, g.Channels)
	for p := 0; p < g.Channels; p++ {
		op.GHat[p] = make([]*fftw.Array2, g.Channels)
		for q := 0; q < g.Channels; q++ {
			// Take Fourier transform of channel pair of g.
			op.GHat[p][q] = dftCovar(g, op.M, op.N, p, q, bx, by)
		}
	}
}

// Mul computes the product of the covariance matrix with the image f.
// Init must be called before Mul.
func (op *MulerFFT) Mul(f *rimg64.Multi) *rimg64.Multi {
	if f.Channels != len(op.GHat) {
		panic(fmt.Sprintf(
			"bad number of channels: covar %d, image %d",
			len(op.GHat), f.Channels,
		))
	}
	if f.Width != op.Width || f.Height != op.Height {
		panic(fmt.Sprintf(
			"bad dimensions: operator %dx%d, image %dx%d",
			op.Width, op.Height,
			f.Width, f.Height,
		))
	}

	// Normalization constant.
	n := float64(op.M) * float64(op.N)
	// Transform of input image.
	fHat := make([]*fftw.Array2, f.Channels)
	for p := 0; p < f.Channels; p++ {
		fHat[p] = dftChannel(f, p, op.M, op.N)
	}
	// Transform of result.
	zHat := make([]*fftw.Array2, f.Channels)
	for p := 0; p < f.Channels; p++ {
		zHat[p] = fftw.NewArray2(op.M, op.N)
	}
	for p := 0; p < f.Channels; p++ {
		for q := 0; q < f.Channels; q++ {
			// Multiply in Fourier domain and add to result.
			for i := 0; i < op.M; i++ {
				for j := 0; j < op.N; j++ {
					delta := op.GHat[q][p].At(i, j) * fHat[q].At(i, j)
					zHat[p].Set(i, j, zHat[p].At(i, j)+delta)
				}
			}
		}
	}
	// Take inverse transform of each channel.
	z := rimg64.NewMulti(f.Width, f.Height, f.Channels)
	for p := 0; p < f.Channels; p++ {
		for i := 0; i < op.M; i++ {
			for j := 0; j < op.N; j++ {
				zHat[p].Set(i, j, complex(1/n, 0)*zHat[p].At(i, j))
			}
		}
		idftToChannel(z, p, zHat[p])
	}
	return z
}
