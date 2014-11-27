package toepcov

import (
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/lin-go/mat"
)

// Describes a dense covariance matrix
// whose elements are indexed (u, v, w), (i, j, k).
type FullCovar struct {
	Width, Height, Channels int
	// Elems[u][v][w][i][j][k]
	Elems [][][][][][]float64
}

func NewFullCovar(width, height, channels int) *FullCovar {
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
	return &FullCovar{width, height, channels, elems}
}

func (cov *FullCovar) At(u, v, w, i, j, k int) float64 {
	return cov.Elems[u][v][w][i][j][k]
}

func (cov *FullCovar) Set(u, v, w, i, j, k int, x float64) {
	cov.Elems[u][v][w][i][j][k] = x
}

func (cov *FullCovar) AddAt(u, v, w, i, j, k int, x float64) {
	cov.Elems[u][v][w][i][j][k] += x
}

func (cov *FullCovar) Clone() *FullCovar {
	dst := NewFullCovar(cov.Width, cov.Height, cov.Channels)
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

func (a *FullCovar) Plus(b *FullCovar) *FullCovar {
	if a.Width != b.Width || a.Height != b.Height || a.Channels != b.Channels {
		panic("dimensions are not the same")
	}
	dst := NewFullCovar(a.Width, a.Height, a.Channels)
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

// Adds a scaled identity matrix to the covariance.
// Modifies the current matrix.
func (a *FullCovar) AddLambdaI(lambda float64) {
	for u := range a.Elems {
		for v := range a.Elems[u] {
			for w := range a.Elems[u][v] {
				a.Elems[u][v][w][u][v][w] += lambda
			}
		}
	}
}

func (cov *FullCovar) Scale(alpha float64) *FullCovar {
	dst := NewFullCovar(cov.Width, cov.Height, cov.Channels)
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

func (cov *FullCovar) Center(mu *rimg64.Multi) *FullCovar {
	dst := NewFullCovar(cov.Width, cov.Height, cov.Channels)
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

// Instantiates the full whc x whc covariance matrix.
// w, h and c are the width, height and number of channels.
func (cov *FullCovar) Matrix() *mat.Mat {
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
