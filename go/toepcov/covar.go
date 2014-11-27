package toepcov

import (
	"errors"
	"log"

	"github.com/jvlmdr/lin-go/mat"
)

// Describes a stationary covariance matrix.
type Covar struct {
	Channels  int
	Bandwidth int
	// Gamma[dx][dy][p][q]
	Gamma [][][][]float64
}

func NewCovar(channels, bandwidth int) *Covar {
	n := 2*bandwidth + 1

	gamma := make([][][][]float64, n)
	for i := 0; i < n; i++ {
		gamma[i] = make([][][]float64, n)
		for j := 0; j < n; j++ {
			gamma[i][j] = make([][]float64, channels)
			for k := 0; k < channels; k++ {
				gamma[i][j][k] = make([]float64, channels)
			}
		}
	}
	return &Covar{channels, bandwidth, gamma}
}

// x, y in [-Bandwidth, Bandwidth]
// p, q in [0, Channels)
func (cov *Covar) At(x, y, p, q int) float64 {
	b := cov.Bandwidth
	return cov.Gamma[b+x][b+y][p][q]
}

// x, y in [-Bandwidth, Bandwidth]
// p, q in [0, Channels)
func (cov *Covar) Set(x, y, p, q int, v float64) {
	b := cov.Bandwidth
	cov.Gamma[b+x][b+y][p][q] = v
}

// Creates a copy.
func (src *Covar) Clone() *Covar {
	dst := NewCovar(src.Channels, src.Bandwidth)
	src.CopyTo(dst)
	return dst
}

// Creates a copy with the specified bandwidth.
func (src *Covar) CloneBandwidth(bandwidth int) *Covar {
	dst := NewCovar(src.Channels, bandwidth)
	src.CopyTo(dst)
	return dst
}

func (src *Covar) CopyTo(dst *Covar) error {
	if src.Channels != dst.Channels {
		return errors.New("Dimension of covariances differs")
	}

	// Copy minimum bandwidth.
	a := src.Bandwidth
	b := dst.Bandwidth
	c := min(src.Bandwidth, dst.Bandwidth)

	for i := -c; i <= c; i++ {
		for j := -c; j <= c; j++ {
			for p := 0; p < src.Channels; p++ {
				copy(dst.Gamma[b+i][b+j][p], src.Gamma[a+i][a+j][p])
			}
		}
	}
	return nil
}

// Instantiates the full whc x whc covariance matrix.
// w, h and c are the width, height and number of channels.
func (g *Covar) Matrix(width, height int) *mat.Mat {
	var (
		n  = g.Channels * height * width
		S  = mat.New(n, n)
		ei = g.Channels * height
		ej = g.Channels
		ek = 1
	)
	log.Printf("instantiate %dx%d covar matrix\n", n, n)
	for i := 0; i < width; i++ {
		for j := 0; j < height; j++ {
			for k := 0; k < g.Channels; k++ {
				r := i*ei + j*ej + k*ek
				for u := 0; u < width; u++ {
					for v := 0; v < height; v++ {
						for w := 0; w < g.Channels; w++ {
							c := u*ei + v*ej + w*ek
							// y(ijk) = sum_uvw s(ijk,uvw) x(uvw)
							//        = sum_uvw g(u-i,v-j,k,w) x(uvw)
							if abs(u-i) <= g.Bandwidth && abs(v-j) <= g.Bandwidth {
								S.Set(r, c, g.At(u-i, v-j, k, w))
							}
						}
					}
				}
			}
		}
	}
	return S
}

// Computes the trace of g.Matrix(width, height).
// Takes O(c) time, where c is the number of channels.
func (g *Covar) Trace(width, height int) float64 {
	var tr float64
	for k := 0; k < g.Channels; k++ {
		tr += g.At(0, 0, k, k)
	}
	return tr * float64(width) * float64(height)
}

// Adds scaled identity to covariance matrix.
func (g *Covar) AddLambdaI(lambda float64) {
	for k := 0; k < g.Channels; k++ {
		g.Set(0, 0, k, k, g.At(0, 0, k, k)+lambda)
	}
}

// Subtracts mu mu' from covariance matrix.
func (g *Covar) Center(mu []float64) {
	// Subtract mu mu'.
	for du := -g.Bandwidth; du <= g.Bandwidth; du++ {
		for dv := -g.Bandwidth; dv <= g.Bandwidth; dv++ {
			for p := 0; p < g.Channels; p++ {
				for q := 0; q < g.Channels; q++ {
					val := g.At(du, dv, p, q) - mu[p]*mu[q]
					g.Set(du, dv, p, q, val)
				}
			}
		}
	}
}

// Returns the sum of two covariance matrices.
// If one has greater bandwidth than the other,
// the larger bandwidth is adopted.
// Does not modify either input.
func AddCovar(lhs, rhs *Covar) *Covar {
	return addCovar(lhs, rhs, false)
}

// Returns the sum of two covariance matrices.
// If one has greater bandwidth than the other,
// the larger bandwidth is adopted.
// Could modify either input.
func AddCovarToEither(lhs, rhs *Covar) *Covar {
	return addCovar(lhs, rhs, true)
}

func addCovar(lhs, rhs *Covar, mutate bool) *Covar {
	// Ensure number of channels is consistent.
	if err := errIfNumChansNotEq(lhs.Channels, rhs.Channels); err != nil {
		panic(err)
	}

	// Swap pointers such that lhs.Bandwidth >= rhs.Bandwidth.
	if lhs.Bandwidth < rhs.Bandwidth {
		lhs, rhs = rhs, lhs
	}
	dst := lhs
	if !mutate {
		dst = dst.Clone()
	}

	// Add values and counts.
	for i := -rhs.Bandwidth; i <= rhs.Bandwidth; i++ {
		for j := -rhs.Bandwidth; j <= rhs.Bandwidth; j++ {
			for p := 0; p < dst.Channels; p++ {
				for q := 0; q < dst.Channels; q++ {
					dst.Set(i, j, p, q, dst.At(i, j, p, q)+rhs.At(i, j, p, q))
				}
			}
		}
	}
	return dst
}

// Downsample takes every n-th sample in x and y.
func (cov *Covar) Downsample(rate int) *Covar {
	// rate * newBand <= oldBand
	// newBand = floor(oldBand / rate)
	band := cov.Bandwidth / rate
	ret := NewCovar(cov.Channels, band)
	for du := -band; du <= band; du++ {
		for dv := -band; dv <= band; dv++ {
			for p := 0; p < cov.Channels; p++ {
				for q := 0; q < cov.Channels; q++ {
					ret.Set(du, dv, p, q, cov.At(rate*du, rate*dv, p, q))
				}
			}
		}
	}
	return ret
}
