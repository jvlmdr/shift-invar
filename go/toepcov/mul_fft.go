package toepcov

import (
	"fmt"
	"image"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-cv/slide"
	"github.com/jvlmdr/go-fftw/fftw"
)

// MulFFT computes the matrix-vector product in the Fourier domain.
func MulFFT(g *Covar, f *rimg64.Multi) *rimg64.Multi {
	if g.Channels != f.Channels {
		panic(fmt.Sprintf("different number of channels: covar %d, image %d", g.Channels, f.Channels))
	}
	var (
		b = g.Bandwidth
		// Limit bandwidth to size of image.
		bx = min(b, f.Width-1)
		by = min(b, f.Height-1)
	)
	// Working dimension in Fourier domain.
	work, _ := slide.FFT2Size(image.Pt(f.Width+bx, f.Height+by))

	fHat := fftw.NewArray2(work.X, work.Y)
	fFwd := fftw.NewPlan2(fHat, fHat, fftw.Forward, fftw.Estimate)
	defer fFwd.Destroy()
	gHat := fftw.NewArray2(work.X, work.Y)
	gFwd := fftw.NewPlan2(gHat, gHat, fftw.Forward, fftw.Estimate)
	defer gFwd.Destroy()
	hHat := make([]*fftw.Array2, f.Channels)
	for i := range hHat {
		hHat[i] = fftw.NewArray2(work.X, work.Y)
	}
	for q := 0; q < f.Channels; q++ {
		// Take Fourier transform of channel q of f.
		copyChannelTo(fHat, f, q)
		fFwd.Execute()
		for p := 0; p < f.Channels; p++ {
			// Take Fourier transform of channel pair (q, p) of g.
			copyCovarTo(gHat, g, q, p, bx, by)
			gFwd.Execute()
			// Multiply and accumulate in Fourier domain.
			addMul(hHat[p], gHat, fHat)
		}
	}
	// Take inverse transform of each channel to give result.
	h := rimg64.NewMulti(f.Width, f.Height, f.Channels)
	// Constant for re-scaling transforms.
	n := float64(work.X * work.Y)
	for p := 0; p < f.Channels; p++ {
		scale(complex(1/n, 0), hHat[p])
		idftToChannel(h, p, hHat[p])
	}
	return h
}
