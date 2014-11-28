package imcov

import (
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/lin-go/mat"
)

// Covar describes a dense covariance matrix.
// Elements are indexed (u, v, w), (i, j, k)
// for channel w at position (u, v) and channel k at position (i, j).
type Covar struct {
	Width, Height, Channels int
	// Elems[u][v][w][i][j][k]
	Elems [][][][][][]float64
}

// NewCovar creates a new covariance matrix of the given dimension.
func NewCovar(width, height, channels int) *Covar {
	elems := make([][][][][][]float64, width)
	for u := range elems {
		elems[u] = make([][][][][]float64, height)
		for v := range elems[u] {
			elems[u][v] = make([][][][]float64, channels)
			for w := range elems[u][v] {
				elems[u][v][w] = make([][][]float64, width)
				for i := range elems[u][v][w] {
					elems[u][v][w][i] = make([][]float64, height)
					for j := range elems[u][v][w][i] {
						elems[u][v][w][i][j] = make([]float64, channels)
					}
				}
			}
		}
	}
	return &Covar{width, height, channels, elems}
}

// At accesses the element (u, v, w), (i, j, k).
func (cov *Covar) At(u, v, w, i, j, k int) float64 {
	return cov.Elems[u][v][w][i][j][k]
}

// Set modifies the element (u, v, w), (i, j, k).
func (cov *Covar) Set(u, v, w, i, j, k int, x float64) {
	cov.Elems[u][v][w][i][j][k] = x
}

// AddAt increments the element (u, v, w), (i, j, k).
func (cov *Covar) AddAt(u, v, w, i, j, k int, x float64) {
	cov.Elems[u][v][w][i][j][k] += x
}

// Clone creates a copy of the covariance matrix.
func (cov *Covar) Clone() *Covar {
	dst := NewCovar(cov.Width, cov.Height, cov.Channels)
	for u := range dst.Elems {
		for v := range dst.Elems[u] {
			for w := range dst.Elems[u][v] {
				for i := range dst.Elems[u][v][w] {
					for j := range dst.Elems[u][v][w][i] {
						for k := range dst.Elems[u][v][w][i][j] {
							dst.Elems[u][v][w][i][j][k] = cov.Elems[u][v][w][i][j][k]
						}
					}
				}
			}
		}
	}
	return dst
}

// Plus computes the sum of two covariances.
// Memory is allocated for the result.
func (a *Covar) Plus(b *Covar) *Covar {
	if a.Width != b.Width || a.Height != b.Height || a.Channels != b.Channels {
		panic("dimensions are not the same")
	}
	dst := NewCovar(a.Width, a.Height, a.Channels)
	for u := range dst.Elems {
		for v := range dst.Elems[u] {
			for w := range dst.Elems[u][v] {
				for i := range dst.Elems[u][v][w] {
					for j := range dst.Elems[u][v][w][i] {
						for k := range dst.Elems[u][v][w][i][j] {
							x := a.Elems[u][v][w][i][j][k]
							y := b.Elems[u][v][w][i][j][k]
							dst.Elems[u][v][w][i][j][k] = x + y
						}
					}
				}
			}
		}
	}
	return dst
}

// AddLambdaI adds a scaled identity matrix to the covariance.
// Modifies the current matrix.
func (a *Covar) AddLambdaI(lambda float64) {
	for u := range a.Elems {
		for v := range a.Elems[u] {
			for w := range a.Elems[u][v] {
				a.Elems[u][v][w][u][v][w] += lambda
			}
		}
	}
}

// Scale multiplies the covariance matrix by a constant.
// Memory is allocated for the result.
func (cov *Covar) Scale(alpha float64) *Covar {
	dst := NewCovar(cov.Width, cov.Height, cov.Channels)
	for u := range dst.Elems {
		for v := range dst.Elems[u] {
			for w := range dst.Elems[u][v] {
				for i := range dst.Elems[u][v][w] {
					for j := range dst.Elems[u][v][w][i] {
						for k := range dst.Elems[u][v][w][i][j] {
							dst.Elems[u][v][w][i][j][k] = alpha * cov.Elems[u][v][w][i][j][k]
						}
					}
				}
			}
		}
	}
	return dst
}

// Center subtracts the mean from the covariance matrix.
// It computes:
// 	cov - mu mu'
// Memory is allocated for the result.
func (cov *Covar) Center(mu *rimg64.Multi) *Covar {
	dst := NewCovar(cov.Width, cov.Height, cov.Channels)
	for u := range dst.Elems {
		for v := range dst.Elems[u] {
			for w := range dst.Elems[u][v] {
				uvw := mu.At(u, v, w)
				for i := range dst.Elems[u][v][w] {
					for j := range dst.Elems[u][v][w][i] {
						for k := range dst.Elems[u][v][w][i][j] {
							ijk := mu.At(i, j, k)
							dst.Elems[u][v][w][i][j][k] = cov.Elems[u][v][w][i][j][k] - uvw*ijk
						}
					}
				}
			}
		}
	}
	return dst
}

// Matrix instantiates the full whc x whc covariance matrix.
// w, h and c are the width, height and number of channels.
func (cov *Covar) Matrix() *mat.Mat {
	var (
		n  = cov.Channels * cov.Height * cov.Width
		s  = mat.New(n, n)
		ei = cov.Channels * cov.Height
		ej = cov.Channels
		ek = 1
	)
	// Function to vectorize indices.
	vec := func(i, j, k int) int { return i*ei + j*ej + k*ek }

	for u := 0; u < cov.Width; u++ {
		for v := 0; v < cov.Height; v++ {
			for w := 0; w < cov.Channels; w++ {
				uvw := vec(u, v, w)
				for i := 0; i < cov.Width; i++ {
					for j := 0; j < cov.Height; j++ {
						for k := 0; k < cov.Channels; k++ {
							ijk := vec(i, j, k)
							s.Set(uvw, ijk, cov.At(u, v, w, i, j, k))
						}
					}
				}
			}
		}
	}
	return s
}
