package exactcov

import (
	"log"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/shift-invar/go/imcov"
)

// Information gathered from image set to obtain mean and covariance.
// Provides ability to combine statistics from multiple images
// and to extract necessary statistics for sub-windows.
type Total struct {
	Mean   *Mean
	Covar  *Covar
	Count  *Count
	Images int
}

// Accumulate stationary statistics over a single image.
func Stats(f *rimg64.Multi, m, n, band int) *Total {
	log.Println("compute stats: mean")
	mean := MeanSum(f, m, n)
	log.Println("compute stats: covar")
	cov := CovarSum(f, m, n, band)
	log.Println("compute stats: count")
	count := CovarCount(f.Width, f.Height, m, n)
	return &Total{mean, cov, count, 1}
}

func (stats *Total) Subset(m, n, band int) *Total {
	return &Total{
		stats.Mean.Subset(m, n),
		stats.Covar.Subset(m, n, band),
		stats.Count.Subset(m, n),
		stats.Images,
	}
}

func (a *Total) Plus(b *Total) *Total {
	log.Println("add stats: mean")
	mean := a.Mean.Plus(b.Mean)
	log.Println("add stats: covar")
	covar := a.Covar.Plus(b.Covar)
	log.Println("add stats: count")
	count := a.Count.Plus(b.Count)
	return &Total{mean, covar, count, a.Images + b.Images}
}

// Computes the mean and covariance from the summations.
// Divides both by the number of examples.
// Does not center the covariance matrix.
func (stats *Total) Normalize() (mu *rimg64.Multi, cov *imcov.Covar) {
	alpha := 1 / float64(stats.Count.Export())
	mu = stats.Mean.Export().Scale(alpha)
	cov = stats.Covar.Export().Scale(alpha)
	return
}

type Count struct {
	Width  int
	Height int
	// Elems[m-1][n-1] gives count(m, n) with 0 <= u < Width, 0 <= v < Height.
	Elems [][]int64
}

func NewCount(m, n int) *Count {
	e := make([][]int64, m)
	for i := range e {
		e[i] = make([]int64, n)
	}
	return &Count{m, n, e}
}

func CovarCount(M, N, m, n int) *Count {
	count := NewCount(m, n)
	for u := 0; u < m && u < M; u++ {
		for v := 0; v < n && v < N; v++ {
			count.Elems[u][v] = int64(M-u) * int64(N-v)
		}
	}
	return count
}

func (count *Count) Subset(m, n int) *Count {
	if m > count.Width || n > count.Height {
		panic("not a subset")
	}

	dst := NewCount(m, n)
	for i := range dst.Elems {
		copy(dst.Elems[i], count.Elems[i])
	}
	return dst
}

// Returns the count for a Width x Height template.
func (count *Count) Export() int64 {
	return count.Elems[count.Width-1][count.Height-1]
}

func (a *Count) Plus(b *Count) *Count {
	if a.Width != b.Width || a.Height != b.Height {
		panic("dimensions are not the same")
	}

	dst := NewCount(a.Width, a.Height)
	for i := range dst.Elems {
		for j := range dst.Elems[i] {
			dst.Elems[i][j] = a.Elems[i][j] + b.Elems[i][j]
		}
	}
	return dst
}
