package toepcov

import (
	"image"

	"github.com/jvlmdr/go-cv/rimg64"
)

// Necessary data to construct a covariance matrix.
// Can also be sub-matrixed.
type ExactMean struct {
	Width     int
	Height    int
	Channels  int
	MeanPixel []float64
	Sides     *sideMean
	Corners   *cornerMean
}

func (cov *ExactMean) Subset(m, n int) *ExactMean {
	if m > cov.Width || n > cov.Height {
		panic("not a subset")
	}

	k := cov.Channels
	dst := &ExactMean{Width: m, Height: n, Channels: k}
	dst.MeanPixel = clonePixel(cov.MeanPixel)
	dst.Sides = cov.Sides.Subset(m, n)
	dst.Corners = cov.Corners.Subset(m, n)
	return dst
}

// Flattens the components into a single un-normalized vector (image).
func (cov *ExactMean) Export() *rimg64.Multi {
	m, n, k := cov.Width, cov.Height, cov.Channels
	a := rimg64.NewMulti(m, n, k)
	for u := 0; u < m; u++ {
		for v := 0; v < n; v++ {
			for p := 0; p < k; p++ {
				s := cov.MeanPixel[p]
				s -= cov.Sides.At(u, v, p)
				s += cov.Corners.At(u, v, p)
				a.Set(u, v, p, s)
			}
		}
	}
	return a
}

func (a *ExactMean) Plus(b *ExactMean) *ExactMean {
	if a.Width != b.Width || a.Height != b.Height || a.Channels != b.Channels {
		panic("dimensions are not the same")
	}
	return &ExactMean{
		a.Width, a.Height, a.Channels,
		addPixel(a.MeanPixel, b.MeanPixel),
		a.Sides.Plus(b.Sides),
		a.Corners.Plus(b.Corners),
	}
}

// Returns non-normalized covariance.
func ExactMeanOf(im *rimg64.Multi, m, n int) *ExactMean {
	k := im.Channels
	// Get stationary part.
	tplz := meanPixel(im)
	// Get non-stationary part.
	sides := sideMeanOf(im, m, n)
	corners := cornerMeanOf(im, m, n)
	return &ExactMean{m, n, k, tplz, sides, corners}
}

// Corners[i][j].At(u, v, p) gives
// sum over a and b of x_p(a, b).
// i = 0 for a in 0 to u-1, i = 1 for a in M-m+u+1 to M-1.
// j = 0 for b in 0 to v-1, j = 1 for b in N-n+v+1 to N-1.
func cornerMeanOf(im *rimg64.Multi, m, n int) *cornerMean {
	M, N, k := im.Width, im.Height, im.Channels
	mu := newCornerMean(m, n, k)

	for p := 0; p < k; p++ {
		// Zero pad images to make life easier.
		phi := func(u, v int) float64 {
			bnds := image.Rect(0, 0, M, N)
			if !image.Pt(u, v).In(bnds) {
				return 0
			}
			return im.At(u, v, p)
		}

		// f[u, v] = sum_{a = 0..u-1} sum_{b = 0..v-1} phi[u, v]
		//         = f[u, v-1] + sum_{a = 0..u-1} phi[u, v-1]
		//         = f[u, v-1] + g[u, v-1]
		// g[u, v] = sum_{a = 0..u-1} phi[u, v]
		//         = g[u-1, v] + phi[u-1, v]

		for u := 0; u < m; u++ {
			g := make([]float64, n)
			for v := 0; v < n; v++ {
				if u > 0 && v > 0 {
					g[v] = g[v-1] + phi(u-1, v-1)
					mu.Elems[0][0].Set(u, v, p, mu.Elems[0][0].At(u-1, v, p)+g[v])
				}
			}
		}

		for u := m - 1; u >= 0; u-- {
			g := make([]float64, n)
			for v := 0; v < n; v++ {
				if u < m-1 && v > 0 {
					g[v] = g[v-1] + phi(M-m+u+1, v-1)
					mu.Elems[1][0].Set(u, v, p, mu.Elems[1][0].At(u+1, v, p)+g[v])
				}
			}
		}

		for u := 0; u < m; u++ {
			g := make([]float64, n)
			for v := n - 1; v >= 0; v-- {
				if u > 0 && v < n-1 {
					g[v] = g[v+1] + phi(u-1, N-n+v+1)
					mu.Elems[0][1].Set(u, v, p, mu.Elems[0][1].At(u-1, v, p)+g[v])
				}
			}
		}

		for u := m - 1; u >= 0; u-- {
			g := make([]float64, n)
			for v := n - 1; v >= 0; v-- {
				if u < m-1 && v < n-1 {
					g[v] = g[v+1] + phi(M-m+u+1, N-n+v+1)
					mu.Elems[1][1].Set(u, v, p, mu.Elems[1][1].At(u+1, v, p)+g[v])
				}
			}
		}
	}
	return mu
}

// Sides[0][i].At(v, p) defined for v = 0, ..., n-1.
// Sides[1][i].At(u, p) defined for u = 0, ..., m-1.
func sideMeanOf(im *rimg64.Multi, m, n int) *sideMean {
	M, N, k := im.Width, im.Height, im.Channels
	mu := newSideMean(m, n, k)

	for p := 0; p < k; p++ {
		phi := func(u, v int) float64 {
			bnds := image.Rect(0, 0, M, N)
			if !image.Pt(u, v).In(bnds) {
				return 0
			}
			return im.At(u, v, p)
		}

		for v := 0; v < n; v++ {
			if v > 0 {
				var t float64
				for a := 0; a < M; a++ {
					t += phi(a, v-1)
				}
				mu.Elems[0][0].Set(v, p, t+mu.Elems[0][0].At(v-1, p))
			}
		}

		for v := n - 1; v >= 0; v-- {
			if v < n-1 {
				var t float64
				for a := 0; a < M; a++ {
					t += phi(a, N-n+v+1)
				}
				mu.Elems[0][1].Set(v, p, t+mu.Elems[0][1].At(v+1, p))
			}
		}

		for u := 0; u < m; u++ {
			if u > 0 {
				var t float64
				for b := 0; b < N; b++ {
					t += phi(u-1, b)
				}
				mu.Elems[1][0].Set(u, p, t+mu.Elems[1][0].At(u-1, p))
			}
		}

		for u := m - 1; u >= 0; u-- {
			if u < m-1 {
				var t float64
				for b := 0; b < N; b++ {
					t += phi(M-m+u+1, b)
				}
				mu.Elems[1][1].Set(u, p, t+mu.Elems[1][1].At(u+1, p))
			}
		}
	}
	return mu
}

type oneCornerMean struct {
	Width    int
	Height   int
	Channels int
	// Elems[u][v][p]
	// 0 <= u < Width, 0 <= v < Height
	// 0 <= p < Channels
	Elems [][][]float64
}

func newOneCornerMean(width, height, channels int) *oneCornerMean {
	e := make([][][]float64, width)
	for u := range e {
		e[u] = make([][]float64, height)
		for v := range e[u] {
			e[u][v] = make([]float64, channels)
		}
	}
	return &oneCornerMean{width, height, channels, e}
}

func (t *oneCornerMean) At(u, v, p int) float64 {
	return t.Elems[u][v][p]
}

func (t *oneCornerMean) Set(u, v, p int, x float64) {
	t.Elems[u][v][p] = x
}

type cornerMean struct {
	Width    int
	Height   int
	Channels int
	Elems    [2][2]*oneCornerMean
}

func newCornerMean(m, n, k int) *cornerMean {
	cov := &cornerMean{Width: m, Height: n, Channels: k}
	cov.Elems[0][0] = newOneCornerMean(m, n, k)
	cov.Elems[1][0] = newOneCornerMean(m, n, k)
	cov.Elems[0][1] = newOneCornerMean(m, n, k)
	cov.Elems[1][1] = newOneCornerMean(m, n, k)
	return cov
}

func (cov *cornerMean) At(u, v, p int) float64 {
	var s float64
	s += cov.Elems[0][0].At(u, v, p)
	s += cov.Elems[1][0].At(u, v, p)
	s += cov.Elems[0][1].At(u, v, p)
	s += cov.Elems[1][1].At(u, v, p)
	return s
}

func (cov *cornerMean) Clone() *cornerMean {
	m, n, k := cov.Width, cov.Height, cov.Channels
	dst := newCornerMean(m, n, k)
	for p := 0; p < k; p++ {
		for u := 0; u < m; u++ {
			for v := 0; v < n; v++ {
				dst.Elems[0][0].Set(u, v, p, cov.Elems[0][0].At(u, v, p))
				dst.Elems[1][0].Set(u, v, p, cov.Elems[1][0].At(u, v, p))
				dst.Elems[0][1].Set(u, v, p, cov.Elems[0][1].At(u, v, p))
				dst.Elems[1][1].Set(u, v, p, cov.Elems[1][1].At(u, v, p))
			}
		}
	}
	return dst
}

func (cov *cornerMean) Subset(m, n int) *cornerMean {
	// Previous (larger) width and height.
	M, N := cov.Width, cov.Height
	if m > M || n > N {
		panic("not a subset")
	}

	k := cov.Channels
	dst := newCornerMean(m, n, k)
	for p := 0; p < k; p++ {
		for u := 0; u < m; u++ {
			for v := 0; v < n; v++ {
				dst.Elems[0][0].Set(u, v, p, cov.Elems[0][0].At(u, v, p))
				dst.Elems[1][0].Set(m-1-u, v, p, cov.Elems[1][0].At(M-1-u, v, p))
				dst.Elems[0][1].Set(u, n-1-v, p, cov.Elems[0][1].At(u, N-1-v, p))
				dst.Elems[1][1].Set(m-1-u, n-1-v, p, cov.Elems[1][1].At(M-1-u, N-1-v, p))
			}
		}
	}
	return dst
}

func (a *cornerMean) Plus(b *cornerMean) *cornerMean {
	if a.Width != b.Width || a.Height != b.Height || a.Channels != b.Channels {
		panic("dimensions are not the same")
	}

	m, n, k := a.Width, a.Height, a.Channels
	dst := newCornerMean(m, n, k)
	for p := 0; p < k; p++ {
		for u := 0; u < m; u++ {
			for v := 0; v < n; v++ {
				add := func(i, j int) {
					x := a.Elems[i][j].At(u, v, p)
					y := b.Elems[i][j].At(u, v, p)
					dst.Elems[i][j].Set(u, v, p, x+y)
				}
				add(0, 0)
				add(0, 1)
				add(1, 0)
				add(1, 1)
			}
		}
	}
	return dst
}

// Describes the covariance for some u or v
// collected from x(a, b, p) for either
// a = 0 to M-1 and b = 0 to v-1 or a = N-n+v+1 to N-1, or
// b = 0 to N-1 and a = 0 to u-1 or a = M-m+u+1 to M-1.
type oneSideMean struct {
	Len      int
	Channels int
	Elems    [][]float64
}

func newSide(n, k int) *oneSideMean {
	e := make([][]float64, n)
	for i := range e {
		e[i] = make([]float64, k)
	}
	return &oneSideMean{n, k, e}
}

func (t *oneSideMean) At(i, p int) float64 {
	return t.Elems[i][p]
}

func (t *oneSideMean) Set(i, p int, x float64) {
	t.Elems[i][p] = x
}

type sideMean struct {
	Width    int
	Height   int
	Channels int
	Elems    [2][2]*oneSideMean
}

func newSideMean(m, n, k int) *sideMean {
	cov := &sideMean{Width: m, Height: n, Channels: k}
	cov.Elems[0][0] = newSide(n, k)
	cov.Elems[0][1] = newSide(n, k)
	cov.Elems[1][0] = newSide(m, k)
	cov.Elems[1][1] = newSide(m, k)
	return cov
}

func (cov *sideMean) At(u, v, p int) float64 {
	var s float64
	s += cov.Elems[0][0].At(v, p)
	s += cov.Elems[0][1].At(v, p)
	s += cov.Elems[1][0].At(u, p)
	s += cov.Elems[1][1].At(u, p)
	return s
}

func (cov *sideMean) Clone() *sideMean {
	m, n, k := cov.Width, cov.Height, cov.Channels
	dst := newSideMean(m, n, k)
	for p := 0; p < k; p++ {
		for v := 0; v < n; v++ {
			dst.Elems[0][0].Set(v, p, cov.Elems[0][0].At(v, p))
			dst.Elems[0][1].Set(v, p, cov.Elems[0][1].At(v, p))
		}
		for u := 0; u < m; u++ {
			dst.Elems[1][0].Set(u, p, cov.Elems[1][0].At(u, p))
			dst.Elems[1][1].Set(u, p, cov.Elems[1][1].At(u, p))
		}
	}
	return dst
}

func (cov *sideMean) Subset(m, n int) *sideMean {
	// Previous (larger) width and height.
	M, N := cov.Width, cov.Height
	if m > M || n > N {
		panic("not a subset")
	}

	k := cov.Channels
	dst := newSideMean(m, n, k)
	for p := 0; p < k; p++ {
		for v := 0; v < n; v++ {
			dst.Elems[0][0].Set(v, p, cov.Elems[0][0].At(v, p))
			dst.Elems[0][1].Set(n-1-v, p, cov.Elems[0][1].At(N-1-v, p))
		}
		for u := 0; u < m; u++ {
			dst.Elems[1][0].Set(u, p, cov.Elems[1][0].At(u, p))
			dst.Elems[1][1].Set(m-1-u, p, cov.Elems[1][1].At(M-1-u, p))
		}
	}
	return dst
}

func (a *sideMean) Plus(b *sideMean) *sideMean {
	if a.Width != b.Width || a.Height != b.Height || a.Channels != b.Channels {
		panic("dimensions are not the same")
	}

	m, n, k := a.Width, a.Height, a.Channels
	dst := newSideMean(m, n, k)
	for p := 0; p < k; p++ {
		for v := 0; v < n; v++ {
			add := func(j int) {
				const i = 0
				x := a.Elems[i][j].At(v, p)
				y := b.Elems[i][j].At(v, p)
				dst.Elems[i][j].Set(v, p, x+y)
			}
			add(0)
			add(1)
		}

		for u := 0; u < m; u++ {
			add := func(j int) {
				const i = 1
				x := a.Elems[i][j].At(u, p)
				y := b.Elems[i][j].At(u, p)
				dst.Elems[i][j].Set(u, p, x+y)
			}
			add(0)
			add(1)
		}
	}
	return dst
}

// Computes mean naively from all windows in one image.
// Mean is not normalized.
func ExactMeanNaive(im *rimg64.Multi, width, height int) *rimg64.Multi {
	if im.Width < width || im.Height < height {
		return nil
	}

	mu := rimg64.NewMulti(width, height, im.Channels)
	for a := 0; a < im.Width-width+1; a++ {
		for b := 0; b < im.Height-height+1; b++ {
			for u := 0; u < width; u++ {
				for v := 0; v < height; v++ {
					for w := 0; w < im.Channels; w++ {
						prev := mu.At(u, v, w)
						curr := im.At(a+u, b+v, w)
						mu.Set(u, v, w, prev+curr)
					}
				}
			}
		}
	}
	return mu
}
