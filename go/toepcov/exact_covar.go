package whog

import (
	"image"
	"log"
	"math/cmplx"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-fftw/fftw"
)

// Necessary data to construct a covariance matrix.
// Can also be sub-matrixed.
type ExactCovar struct {
	Width     int
	Height    int
	Channels  int
	Bandwidth int
	Toeplitz  *Covar
	Sides     *sideCovar
	Corners   *cornerCovar
}

func (cov *ExactCovar) Clone() *ExactCovar {
	return &ExactCovar{
		cov.Width, cov.Height, cov.Channels, cov.Bandwidth,
		cov.Toeplitz.Clone(),
		cov.Sides.Clone(),
		cov.Corners.Clone(),
	}
}

func (cov *ExactCovar) Subset(m, n, band int) *ExactCovar {
	if m > cov.Width || n > cov.Height || band > cov.Bandwidth {
		panic("not a subset")
	}

	k := cov.Channels
	dst := &ExactCovar{Width: m, Height: n, Channels: k, Bandwidth: band}
	dst.Toeplitz = cov.Toeplitz.CloneBandwidth(band)
	dst.Sides = cov.Sides.Subset(m, n)
	dst.Corners = cov.Corners.Subset(m, n)
	return dst
}

// Flattens the components into a single un-normalized matrix.
func (cov *ExactCovar) Export() *FullCovar {
	m, n, k := cov.Width, cov.Height, cov.Channels
	a := NewFullCovar(m, n, k)
	for u := 0; u < m; u++ {
		for v := 0; v < n; v++ {
			for p := 0; p < k; p++ {
				for i := 0; i < m; i++ {
					for j := 0; j < n; j++ {
						for q := 0; q < k; q++ {
							du, dv := i-u, j-v
							s := cov.Toeplitz.At(du, dv, p, q)
							s -= cov.Sides.At(u, v, du, dv, p, q)
							s += cov.Corners.At(u, v, p, i, j, q)
							a.Set(u, v, p, i, j, q, s)
						}
					}
				}
			}
		}
	}
	return a
}

func (a *ExactCovar) Plus(b *ExactCovar) *ExactCovar {
	if a.Width != b.Width || a.Height != b.Height || a.Channels != b.Channels || a.Bandwidth != b.Bandwidth {
		panic("dimensions are not the same")
	}
	log.Println("add covar: Toeplitz")
	tplz := AddCovar(a.Toeplitz, b.Toeplitz)
	log.Println("add covar: sides")
	sides := a.Sides.Plus(b.Sides)
	log.Println("add covar: corners")
	corners := a.Corners.Plus(b.Corners)
	return &ExactCovar{a.Width, a.Height, a.Channels, a.Bandwidth, tplz, sides, corners}
}

// Returns non-normalized covariance.
func ExactCovarOf(im *rimg64.Multi, m, n, band int) *ExactCovar {
	k := im.Channels
	// Get stationary part.
	log.Println("compute covar: Toeplitz")
	tplz := covarStatsFFT(im, band)
	// Get non-stationary part.
	log.Println("compute covar: sides")
	sides := sideCovarOf(im, m, n)
	log.Println("compute covar: corners")
	corners := cornerCovarOf(im, m, n)
	return &ExactCovar{m, n, k, band, tplz, sides, corners}
}

// Corners[i][j].At(u, v, du, dv, p, q) gives
// sum over a and b of x_p(a, b) + x_q(a+du, v+dv).
// i = 0 for a in 0 to u-1, i = 1 for a in M-m+u+1 to M-1.
// j = 0 for b in 0 to v-1, j = 1 for b in N-n+v+1 to N-1.
func cornerCovarOf(im *rimg64.Multi, m, n int) *cornerCovar {
	M, N, k := im.Width, im.Height, im.Channels
	cov := newCornerCovar(m, n, k)

	for p := 0; p < k; p++ {
		for q := 0; q < k; q++ {
			for du := -m + 1; du <= m-1; du++ {
				for dv := -n + 1; dv <= n-1; dv++ {
					// Zero pad images to make life easier.
					phi := func(u, v int) float64 {
						bnds := image.Rect(0, 0, M, N)
						if !image.Pt(u, v).In(bnds) {
							return 0
						}
						if !image.Pt(u+du, v+dv).In(bnds) {
							return 0
						}
						return im.At(u, v, p) * im.At(u+du, v+dv, q)
					}

					// f[u, v] = sum_{a = 0..u-1} sum_{b = 0..v-1} phi[u, v]
					//         = f[u, v-1] + sum_{a = 0..u-1} phi[u, v-1]
					//         = f[u, v-1] + g[u, v-1]
					// g[u, v] = sum_{a = 0..u-1} phi[u, v]
					//         = g[u-1, v] + phi[u-1, v]

					for u := max(0, -du) + 1; u < min(m, m-du); u++ {
						g := make([]float64, n)
						for v := max(0, -dv) + 1; v < min(n, n-dv); v++ {
							g[v] = g[v-1] + phi(u-1, v-1)
							i, j := u+du, v+dv
							cov.Elems[0][0].Set(u, v, p, i, j, q, cov.Elems[0][0].At(u-1, v, p, i-1, j, q)+g[v])
						}
					}

					for u := min(m-1, m-du-1) - 1; u >= max(0, -du); u-- {
						g := make([]float64, n)
						for v := max(0, -dv) + 1; v < min(n, n-dv); v++ {
							g[v] = g[v-1] + phi(M-m+u+1, v-1)
							i, j := u+du, v+dv
							cov.Elems[1][0].Set(u, v, p, i, j, q, cov.Elems[1][0].At(u+1, v, p, i+1, j, q)+g[v])
						}
					}

					for u := max(0, -du) + 1; u < min(m, m-du); u++ {
						g := make([]float64, n)
						for v := min(n-1, n-dv-1) - 1; v >= max(0, -dv); v-- {
							g[v] = g[v+1] + phi(u-1, N-n+v+1)
							i, j := u+du, v+dv
							cov.Elems[0][1].Set(u, v, p, i, j, q, cov.Elems[0][1].At(u-1, v, p, i-1, j, q)+g[v])
						}
					}

					for u := min(m-1, m-du-1) - 1; u >= max(0, -du); u-- {
						g := make([]float64, n)
						for v := min(n-1, n-dv-1) - 1; v >= max(0, -dv); v-- {
							g[v] = g[v+1] + phi(M-m+u+1, N-n+v+1)
							i, j := u+du, v+dv
							cov.Elems[1][1].Set(u, v, p, i, j, q, cov.Elems[1][1].At(u+1, v, p, i+1, j, q)+g[v])
						}
					}
				}
			}
		}
	}
	return cov
}

// Sides[0][i].At(v, du, dv, p, q) defined for v = 0, ..., n-1.
// Sides[1][i].At(u, du, dv, p, q) defined for u = 0, ..., m-1.
func sideCovarOf(im *rimg64.Multi, m, n int) *sideCovar {
	M, N, k := im.Width, im.Height, im.Channels
	cov := newSideCovar(m, n, k)

	for p := 0; p < k; p++ {
		for q := 0; q < k; q++ {
			phi := func(u, v, du, dv int) float64 {
				bnds := image.Rect(0, 0, M, N)
				if !image.Pt(u, v).In(bnds) {
					return 0
				}
				if !image.Pt(u+du, v+dv).In(bnds) {
					return 0
				}
				return im.At(u, v, p) * im.At(u+du, v+dv, q)
			}

			// sum_{a = 0..M-1} sum_{b = 0..v-1} x(a, b, p) x(a+du, b+dv, q)
			// v >= 0 and v <= n-1 and v-1 >= 0 and v-1 <= n-1
			// 1 <= v <= n-1
			for v := 1; v < n; v++ {
				f := fftw.NewArray(M + m - 1)
				for u := 0; u < M; u++ {
					f.Set(u, complex(im.At(u, v-1, p), 0))
				}
				f = fftw.FFT(f)
				alpha := complex(float64(f.Len()), 0)

				// v+dv >= 0 and v+dv <= n-1 and v-1+dv >= 0 and v-1+dv <= n-1
				// dv >= -v and dv <= n-1-v and dv >= 1-v and dv <= n-v
				// -v+1 <= v <= n-v-1
				for dv := -v + 1; dv < n-v; dv++ {
					// Compute sum over a for du = -m+1..m-1.
					// Extract signal and pad with m-1 zeros.
					g := fftw.NewArray(M + m - 1)
					for u := 0; u < M; u++ {
						g.Set(u, complex(im.At(u, v-1+dv, q), 0))
					}
					g = fftw.FFT(g)
					// Point-wise multiply in Fourier domain.
					for u := 0; u < f.Len(); u++ {
						g.Set(u, cmplx.Conj(f.At(u))*g.At(u)/alpha)
					}
					g = fftw.IFFT(g)
					// Assign result for -m+1 <= du <= m-1.
					for du := -m + 1; du <= m-1; du++ {
						prev := cov.Elems[0][0].At(v-1, du, dv, p, q)
						curr := real(g.At(mod(du, f.Len())))
						cov.Elems[0][0].Set(v, du, dv, p, q, prev+curr)
					}
				}
			}

			for v := n - 1; v >= 0; v-- {
				if v < n-1 {
					for dv := -n + 1; dv <= n-1; dv++ {
						for du := -m + 1; du <= m-1; du++ {
							var t float64
							for u := 0; u < M; u++ {
								t += phi(u, N-n+v+1, du, dv)
							}
							cov.Elems[0][1].Set(v, du, dv, p, q, t+cov.Elems[0][1].At(v+1, du, dv, p, q))
						}
					}
				}
			}

			for u := 0; u < m; u++ {
				if u > 0 {
					for du := -m + 1; du <= m-1; du++ {
						for dv := -n + 1; dv <= n-1; dv++ {
							var t float64
							for v := 0; v < N; v++ {
								t += phi(u-1, v, du, dv)
							}
							cov.Elems[1][0].Set(u, du, dv, p, q, t+cov.Elems[1][0].At(u-1, du, dv, p, q))
						}
					}
				}
			}

			for u := m - 1; u >= 0; u-- {
				if u < m-1 {
					for du := -m + 1; du <= m-1; du++ {
						for dv := -n + 1; dv <= n-1; dv++ {
							var t float64
							for v := 0; v < N; v++ {
								t += phi(M-m+u+1, v, du, dv)
							}
							cov.Elems[1][1].Set(u, du, dv, p, q, t+cov.Elems[1][1].At(u+1, du, dv, p, q))
						}
					}
				}
			}
		}
	}
	return cov
}

type oneCornerCovar struct {
	Width    int
	Height   int
	Channels int
	// Elems[u][v][u+du][v+dv][p][q]
	// 0 <= u < Width, 0 <= v < Height
	// 0 <= u+du < Width, 0 <= v+dv < Height
	// 0 <= p, q < Channels
	Elems [][][][][][]float64
}

type cornerCovar struct {
	Width    int
	Height   int
	Channels int
	Elems    [2][2]*FullCovar
}

func newCornerCovar(m, n, k int) *cornerCovar {
	cov := &cornerCovar{Width: m, Height: n, Channels: k}
	cov.Elems[0][0] = NewFullCovar(m, n, k)
	cov.Elems[1][0] = NewFullCovar(m, n, k)
	cov.Elems[0][1] = NewFullCovar(m, n, k)
	cov.Elems[1][1] = NewFullCovar(m, n, k)
	return cov
}

func (cov *cornerCovar) At(u, v, p, i, j, q int) float64 {
	var s float64
	s += cov.Elems[0][0].At(u, v, p, i, j, q)
	s += cov.Elems[1][0].At(u, v, p, i, j, q)
	s += cov.Elems[0][1].At(u, v, p, i, j, q)
	s += cov.Elems[1][1].At(u, v, p, i, j, q)
	return s
}

func (cov *cornerCovar) Clone() *cornerCovar {
	m, n, k := cov.Width, cov.Height, cov.Channels
	dst := newCornerCovar(m, n, k)
	for p := 0; p < k; p++ {
		for q := 0; q < k; q++ {
			for u := 0; u < m; u++ {
				for v := 0; v < n; v++ {
					for i := 0; i < m; i++ {
						for j := 0; j < n; j++ {
							dst.Elems[0][0].Set(u, v, p, i, j, q, cov.Elems[0][0].At(u, v, p, i, j, q))
							dst.Elems[1][0].Set(u, v, p, i, j, q, cov.Elems[1][0].At(u, v, p, i, j, q))
							dst.Elems[0][1].Set(u, v, p, i, j, q, cov.Elems[0][1].At(u, v, p, i, j, q))
							dst.Elems[1][1].Set(u, v, p, i, j, q, cov.Elems[1][1].At(u, v, p, i, j, q))
						}
					}
				}
			}
		}
	}
	return dst
}

func (cov *cornerCovar) Subset(m, n int) *cornerCovar {
	// Previous (larger) width and height.
	M, N := cov.Width, cov.Height
	if m > M || n > N {
		panic("not a subset")
	}

	k := cov.Channels
	dst := newCornerCovar(m, n, k)
	for p := 0; p < k; p++ {
		for q := 0; q < k; q++ {
			for u := 0; u < m; u++ {
				for v := 0; v < n; v++ {
					for i := 0; i < m; i++ {
						for j := 0; j < n; j++ {
							dst.Elems[0][0].Set(u, v, p, i, j, q, cov.Elems[0][0].At(u, v, p, i, j, q))
							dst.Elems[1][0].Set(u, v, p, i, j, q, cov.Elems[1][0].At(M-m+u, v, p, M-m+i, j, q))
							dst.Elems[0][1].Set(u, v, p, i, j, q, cov.Elems[0][1].At(u, N-n+v, p, i, N-n+j, q))
							dst.Elems[1][1].Set(u, v, p, i, j, q, cov.Elems[1][1].At(M-m+u, N-n+v, p, M-m+i, N-n+j, q))
						}
					}
				}
			}
		}
	}
	return dst
}

func (a *cornerCovar) Plus(b *cornerCovar) *cornerCovar {
	if a.Width != b.Width || a.Height != b.Height || a.Channels != b.Channels {
		panic("dimensions are not the same")
	}

	m, n, k := a.Width, a.Height, a.Channels
	dst := newCornerCovar(m, n, k)
	for p := 0; p < k; p++ {
		for q := 0; q < k; q++ {
			for u := 0; u < m; u++ {
				for v := 0; v < n; v++ {
					for i := 0; i < m; i++ {
						for j := 0; j < n; j++ {
							add := func(r, s int) {
								x := a.Elems[r][s].At(u, v, p, i, j, q)
								y := b.Elems[r][s].At(u, v, p, i, j, q)
								dst.Elems[r][s].Set(u, v, p, i, j, q, x+y)
							}
							add(0, 0)
							add(0, 1)
							add(1, 0)
							add(1, 1)
						}
					}
				}
			}
		}
	}
	return dst
}

// Describes the covariance for some (u, du, dv) or (v, du, dv)
// collected from x(a, b, p) x(a+du, b+dv, q) for either
// a = 0 to M-1 and b = 0 to v-1 or a = N-n+v+1 to N-1, or
// b = 0 to N-1 and a = 0 to u-1 or a = M-m+u+1 to M-1.
type oneSideCovar struct {
	Len      int
	Width    int
	Height   int
	Channels int
	Elems    [][][][][]float64
}

func newOneSideCovar(l, m, n, k int) *oneSideCovar {
	e := make([][][][][]float64, l)
	for i := range e {
		e[i] = make([][][][]float64, 2*m-1)
		for du := range e[i] {
			e[i][du] = make([][][]float64, 2*n-1)
			for dv := range e[i][du] {
				e[i][du][dv] = make([][]float64, k)
				for p := range e[i][du][dv] {
					e[i][du][dv][p] = make([]float64, k)
				}
			}
		}
	}
	return &oneSideCovar{l, m, n, k, e}
}

func (t *oneSideCovar) At(i, du, dv, p, q int) float64 {
	return t.Elems[i][t.Width-1+du][t.Height-1+dv][p][q]
}

func (t *oneSideCovar) Set(i, du, dv, p, q int, x float64) {
	t.Elems[i][t.Width-1+du][t.Height-1+dv][p][q] = x
}

type sideCovar struct {
	Width    int
	Height   int
	Channels int
	Elems    [2][2]*oneSideCovar
}

func newSideCovar(m, n, k int) *sideCovar {
	cov := &sideCovar{Width: m, Height: n, Channels: k}
	cov.Elems[0][0] = newOneSideCovar(n, m, n, k)
	cov.Elems[0][1] = newOneSideCovar(n, m, n, k)
	cov.Elems[1][0] = newOneSideCovar(m, m, n, k)
	cov.Elems[1][1] = newOneSideCovar(m, m, n, k)
	return cov
}

func (cov *sideCovar) At(u, v, du, dv, p, q int) float64 {
	var s float64
	s += cov.Elems[0][0].At(v, du, dv, p, q)
	s += cov.Elems[0][1].At(v, du, dv, p, q)
	s += cov.Elems[1][0].At(u, du, dv, p, q)
	s += cov.Elems[1][1].At(u, du, dv, p, q)
	return s
}

func (cov *sideCovar) Clone() *sideCovar {
	m, n, k := cov.Width, cov.Height, cov.Channels
	dst := newSideCovar(m, n, k)
	for p := 0; p < k; p++ {
		for q := 0; q < k; q++ {
			for du := -m + 1; du <= m-1; du++ {
				for dv := -n + 1; dv <= n-1; dv++ {
					for v := 0; v < n; v++ {
						dst.Elems[0][0].Set(v, du, dv, p, q, cov.Elems[0][0].At(v, du, dv, p, q))
						dst.Elems[0][1].Set(v, du, dv, p, q, cov.Elems[0][1].At(v, du, dv, p, q))
					}
					for u := 0; u < m; u++ {
						dst.Elems[1][0].Set(u, du, dv, p, q, cov.Elems[1][0].At(u, du, dv, p, q))
						dst.Elems[1][1].Set(u, du, dv, p, q, cov.Elems[1][1].At(u, du, dv, p, q))
					}
				}
			}
		}
	}
	return dst
}

func (cov *sideCovar) Subset(m, n int) *sideCovar {
	// Previous (larger) width and height.
	M, N := cov.Width, cov.Height
	if m > M || n > N {
		panic("not a subset")
	}

	k := cov.Channels
	dst := newSideCovar(m, n, k)
	for p := 0; p < k; p++ {
		for q := 0; q < k; q++ {
			for du := -m + 1; du <= m-1; du++ {
				for dv := -n + 1; dv <= n-1; dv++ {
					for v := 0; v < n; v++ {
						dst.Elems[0][0].Set(v, du, dv, p, q, cov.Elems[0][0].At(v, du, dv, p, q))
						dst.Elems[0][1].Set(n-1-v, du, dv, p, q, cov.Elems[0][1].At(N-1-v, du, dv, p, q))
					}
					for u := 0; u < m; u++ {
						dst.Elems[1][0].Set(u, du, dv, p, q, cov.Elems[1][0].At(u, du, dv, p, q))
						dst.Elems[1][1].Set(m-1-u, du, dv, p, q, cov.Elems[1][1].At(M-1-u, du, dv, p, q))
					}
				}
			}
		}
	}
	return dst
}

func (a *sideCovar) Plus(b *sideCovar) *sideCovar {
	if a.Width != b.Width || a.Height != b.Height || a.Channels != b.Channels {
		panic("dimensions are not the same")
	}

	m, n, k := a.Width, a.Height, a.Channels
	dst := newSideCovar(m, n, k)
	for p := 0; p < k; p++ {
		for q := 0; q < k; q++ {
			for du := -m + 1; du <= m-1; du++ {
				for dv := -n + 1; dv <= n-1; dv++ {
					for v := 0; v < n; v++ {
						add := func(j int) {
							const i = 0
							x := a.Elems[i][j].At(v, du, dv, p, q)
							y := b.Elems[i][j].At(v, du, dv, p, q)
							dst.Elems[i][j].Set(v, du, dv, p, q, x+y)
						}
						add(0)
						add(1)
					}

					for u := 0; u < m; u++ {
						add := func(j int) {
							const i = 1
							x := a.Elems[i][j].At(u, du, dv, p, q)
							y := b.Elems[i][j].At(u, du, dv, p, q)
							dst.Elems[i][j].Set(u, du, dv, p, q, x+y)
						}
						add(0)
						add(1)
					}
				}
			}
		}
	}
	return dst
}

// Computes covariance naively from all windows in one image.
// Covariance is not normalized.
func ExactCovarNaive(im *rimg64.Multi, width, height int) *FullCovar {
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
									uvp := im.At(a+u, b+v, p)
									ijq := im.At(a+i, b+j, q)
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
