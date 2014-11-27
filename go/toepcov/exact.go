package toepcov

import (
	"log"

	"github.com/jvlmdr/go-cv/rimg64"
)

// Information gathered from image set to obtain mean and covariance.
// Provides ability to combine statistics from multiple images
// and to extract necessary statistics for sub-windows.
type ExactStats struct {
	Mean   *ExactMean
	Covar  *ExactCovar
	Count  *ExactCount
	Images int
}

// Accumulate stationary statistics over a single image.
func ExactStatsOf(f *rimg64.Multi, m, n, band int) *ExactStats {
	log.Println("compute stats: mean")
	mean := ExactMeanOf(f, m, n)
	log.Println("compute stats: covar")
	cov := ExactCovarOf(f, m, n, band)
	log.Println("compute stats: count")
	count := ExactCountOf(f.Width, f.Height, m, n)
	return &ExactStats{mean, cov, count, 1}
}

func (stats *ExactStats) Subset(m, n, band int) *ExactStats {
	return &ExactStats{
		stats.Mean.Subset(m, n),
		stats.Covar.Subset(m, n, band),
		stats.Count.Subset(m, n),
		stats.Images,
	}
}

func (a *ExactStats) Plus(b *ExactStats) *ExactStats {
	log.Println("add stats: mean")
	mean := a.Mean.Plus(b.Mean)
	log.Println("add stats: covar")
	covar := a.Covar.Plus(b.Covar)
	log.Println("add stats: count")
	count := a.Count.Plus(b.Count)
	return &ExactStats{mean, covar, count, a.Images + b.Images}
}

// Computes the mean and covariance from the summations.
// Divides both by the number of examples.
// Does not center the covariance matrix.
func (stats *ExactStats) Normalize() (mu *rimg64.Multi, cov *FullCovar) {
	alpha := 1 / float64(stats.Count.Export())
	mu = stats.Mean.Export().Scale(alpha)
	cov = stats.Covar.Export().Scale(alpha)
	return
}

type ExactCount struct {
	Width  int
	Height int
	// Elems[m-1][n-1] gives count(m, n) with 0 <= u < Width, 0 <= v < Height.
	Elems [][]int64
}

func NewExactCount(m, n int) *ExactCount {
	e := make([][]int64, m)
	for i := range e {
		e[i] = make([]int64, n)
	}
	return &ExactCount{m, n, e}
}

func ExactCountOf(M, N, m, n int) *ExactCount {
	count := NewExactCount(m, n)
	for u := 0; u < m && u < M; u++ {
		for v := 0; v < n && v < N; v++ {
			count.Elems[u][v] = int64(M-u) * int64(N-v)
		}
	}
	return count
}

func (count *ExactCount) Subset(m, n int) *ExactCount {
	if m > count.Width || n > count.Height {
		panic("not a subset")
	}

	dst := NewExactCount(m, n)
	for i := range dst.Elems {
		copy(dst.Elems[i], count.Elems[i])
	}
	return dst
}

// Returns the count for a Width x Height template.
func (count *ExactCount) Export() int64 {
	return count.Elems[count.Width-1][count.Height-1]
}

func (a *ExactCount) Plus(b *ExactCount) *ExactCount {
	if a.Width != b.Width || a.Height != b.Height {
		panic("dimensions are not the same")
	}

	dst := NewExactCount(a.Width, a.Height)
	for i := range dst.Elems {
		for j := range dst.Elems[i] {
			dst.Elems[i][j] = a.Elems[i][j] + b.Elems[i][j]
		}
	}
	return dst
}
