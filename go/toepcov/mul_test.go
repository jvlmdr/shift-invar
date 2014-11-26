package whog

import (
	"testing"
	"time"

	"github.com/jvlmdr/go-cv/rimg64"
)

type mulAlgo struct {
	Name string
	Func MulFunc
}

func mulerFFTMul(cov *Covar, f *rimg64.Multi) *rimg64.Multi {
	var muler MulerFFT
	muler.Init(cov, f.Width, f.Height)
	return muler.Mul(f)
}

func testMul(t *testing.T, width, height, bandwidth, channels int, muls []mulAlgo) {
	cov := randCovar(channels, bandwidth)
	f := randImage(width, height, channels)

	// Treat first multiplier as reference.
	var want *rimg64.Multi

	for i, mul := range muls {
		start := time.Now()
		got := mul.Func(cov, f)
		duration := time.Since(start)
		t.Logf("%s: %v", mul.Name, duration)
		if i == 0 {
			want = got
		} else {
			testImageEq(t, want, got)
		}
	}
}

// bandwidth < width/2, height/2
func TestMul(t *testing.T) {
	muls := []mulAlgo{
		{"naive", MulNaive},
		{"fft", MulFFT},
		{"fft-muler", mulerFFTMul},
	}
	testMul(t, 40, 30, 8, 4, muls)
}

// Big problem.
func TestMul_large(t *testing.T) {
	if testing.Short() {
		t.Skip("skip in short mode")
	}
	muls := []mulAlgo{
		{"naive", MulNaive},
		{"fft", MulFFT},
		{"fft-muler", mulerFFTMul},
	}
	testMul(t, 150, 100, 40, 4, muls)
}

// width, height < bandwidth
func TestMul_bandwidthExceedsDims(t *testing.T) {
	muls := []mulAlgo{
		{"naive", MulNaive},
		{"fft", MulFFT},
		{"fft-muler", mulerFFTMul},
	}
	testMul(t, 20, 10, 25, 4, muls)
}

// width/2, height/2 < bandwidth < width, height
func TestMul_halfDimsBandwidthDims(t *testing.T) {
	muls := []mulAlgo{
		{"naive", MulNaive},
		{"fft", MulFFT},
		{"fft-muler", mulerFFTMul},
	}
	testMul(t, 20, 18, 13, 4, muls)
}

// height < bandwidth < width
func TestMul_heightBandwidthWidth(t *testing.T) {
	muls := []mulAlgo{
		{"naive", MulNaive},
		{"fft", MulFFT},
		{"fft-muler", mulerFFTMul},
	}
	testMul(t, 30, 15, 25, 4, muls)
}

// width < bandwidth < height
func TestMul_widthBandwidthHeight(t *testing.T) {
	muls := []mulAlgo{
		{"naive", MulNaive},
		{"fft", MulFFT},
		{"fft-muler", mulerFFTMul},
	}
	testMul(t, 10, 20, 15, 4, muls)
}
