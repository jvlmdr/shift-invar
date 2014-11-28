package exactcov

import (
	"testing"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/lin-go/lapack"
	"github.com/jvlmdr/shift-invar/go/imcov"
)

func TestCount(t *testing.T) {
	M, N := 5, 3
	m, n := 3, 2
	count := CovarCount(M, N, m, n)
	want := int64(M-m+1) * int64(N-n+1)
	got := count.Export()
	if want != got {
		t.Errorf("different: want %.6g, got %.6g", want, got)
	}
}

func TestCount_Subset(t *testing.T) {
	M, N := 5, 3
	m, n := 3, 2
	count := CovarCount(M, N, m, n)
	m, n = 1, 2
	count = count.Subset(m, n)
	want := int64(M-m+1) * int64(N-n+1)
	got := count.Export()
	if want != got {
		t.Errorf("different: want %.6g, got %.6g", want, got)
	}
}

func TestStats_Normalize(t *testing.T) {
	// Dimension of image.
	M, N := 20, 15
	// Dimension of template.
	m, n := 6, 9
	// Number of channels.
	k := 3
	im := randImage(M, N, k)

	// Compute mean and covariance naively.
	count := (M - m + 1) * (N - n + 1)
	naiveMean := imcov.MeanSum(im, m, n).Scale(1 / float64(count))
	naiveCovar := imcov.CovarSum(im, m, n).Scale(1 / float64(count))

	// Compute using Normalize().
	fastMean, fastCovar := Stats(im, m, n, max(m, n)-1).Normalize()

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
	mean := imcov.MeanSum(im, m, n).Scale(1 / float64(count))
	// Compute covar naively with mean subtracted, then normalize.
	pre := covarSumCenter(im, mean).Scale(1 / float64(count))

	// Compute covar without subtracting mean then normalize.
	post := imcov.CovarSum(im, m, n).Scale(1 / float64(count))
	//_, fast := Stats(im, m, n, max(m, n)-1).Normalize()
	post = post.Center(mean)

	testFullCovarEq(t, pre, post)
}

func TestCovar_Center_PSD(t *testing.T) {
	// Dimension of image.
	M, N := 20, 15
	// Dimension of template.
	m, n := 6, 9
	// Number of channels.
	k := 3
	im := randImage(M, N, k)

	mean, covar := Stats(im, m, n, max(m, n)-1).Normalize()
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
func covarSumCenter(im, mean *rimg64.Multi) *imcov.Covar {
	width, height := mean.Width, mean.Height
	if im.Width < width || im.Height < height {
		return nil
	}

	cov := imcov.NewCovar(width, height, im.Channels)
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
