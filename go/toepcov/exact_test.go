package whog

import (
	"testing"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/lin-go/lapack"
)

func TestExactCount(t *testing.T) {
	M, N := 5, 3
	m, n := 3, 2
	count := ExactCountOf(M, N, m, n)
	want := int64(M-m+1) * int64(N-n+1)
	got := count.Export()
	if want != got {
		t.Errorf("different: want %.6g, got %.6g", want, got)
	}
}

func TestExactCount_Subset(t *testing.T) {
	M, N := 5, 3
	m, n := 3, 2
	count := ExactCountOf(M, N, m, n)
	m, n = 1, 2
	count = count.Subset(m, n)
	want := int64(M-m+1) * int64(N-n+1)
	got := count.Export()
	if want != got {
		t.Errorf("different: want %.6g, got %.6g", want, got)
	}
}

func TestExactStats_Normalize(t *testing.T) {
	// Dimension of image.
	M, N := 20, 15
	// Dimension of template.
	m, n := 6, 9
	// Number of channels.
	k := 3
	im := randImage(M, N, k)

	// Compute mean and covariance naively.
	count := (M - m + 1) * (N - n + 1)
	naiveMean := ExactMeanNaive(im, m, n).Scale(1 / float64(count))
	naiveCovar := ExactCovarNaive(im, m, n).Scale(1 / float64(count))

	// Compute using Normalize().
	fastMean, fastCovar := ExactStatsOf(im, m, n, max(m, n)-1).Normalize()

	// Compare results.
	testImageEq(t, naiveMean, fastMean)
	testFullCovarEq(t, naiveCovar, fastCovar)
}

func TestFullCovar_Center(t *testing.T) {
	// Dimension of image.
	M, N := 20, 15
	// Dimension of template.
	m, n := 6, 9
	// Number of channels.
	k := 3
	im := randImage(M, N, k)

	count := (M - m + 1) * (N - n + 1)
	// Compute mean naively.
	mean := ExactMeanNaive(im, m, n).Scale(1 / float64(count))
	// Compute covar naively with mean subtracted, then normalize.
	pre := exactCovarNaiveMean(im, mean).Scale(1 / float64(count))

	// Compute covar without subtracting mean then normalize.
	post := ExactCovarNaive(im, m, n).Scale(1 / float64(count))
	//_, fast := ExactStatsOf(im, m, n, max(m, n)-1).Normalize()
	post = post.Center(mean)

	testFullCovarEq(t, pre, post)
}

func TestExactCovar_Center_PSD(t *testing.T) {
	// Dimension of image.
	M, N := 20, 15
	// Dimension of template.
	m, n := 6, 9
	// Number of channels.
	k := 3
	im := randImage(M, N, k)

	mean, covar := ExactStatsOf(im, m, n, max(m, n)-1).Normalize()
	// Subtract mu*mu'.
	centered := covar.Center(mean)
	// Give some tolerance.
	covar.AddLambdaI(1e-6)
	centered.AddLambdaI(1e-6)

	if _, err := lapack.Chol(covar.Matrix()); err != nil {
		t.Error("without mean removed:", err)
	}
	if _, err := lapack.Chol(centered.Matrix()); err != nil {
		t.Error("with mean removed:", err)
	}
}

// Computes covariance naively from all windows in one image with mean removed.
// Returns non-normalized covariance.
func exactCovarNaiveMean(im, mean *rimg64.Multi) *FullCovar {
	width, height := mean.Width, mean.Height
	if im.Width < width || im.Height < height {
		return nil
	}

	cov := NewFullCovar(width, height, im.Channels)
	for a := 0; a < im.Width-width+1; a++ {
		for b := 0; b < im.Height-height+1; b++ {
			for u := 0; u < width; u++ {
				for v := 0; v < height; v++ {
					for p := 0; p < im.Channels; p++ {
						for i := 0; i < width; i++ {
							for j := 0; j < height; j++ {
								for q := 0; q < im.Channels; q++ {
									uvp := im.At(a+u, b+v, p) - mean.At(u, v, p)
									ijq := im.At(a+i, b+j, q) - mean.At(i, j, q)
									cov.AddAt(u, v, p, i, j, q, uvp*ijq)
								}
							}
						}
					}
				}
			}
		}
	}
	return cov
}
