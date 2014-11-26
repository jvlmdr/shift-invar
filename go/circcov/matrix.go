package circcov

import (
	"github.com/jvlmdr/go-whog/whog"
	"github.com/jvlmdr/lin-go/mat"
)

// g(du, dv) = sum_{a, b} x(a, b) x(a+du, a+dv)
//
// h(du, dv) = sum_{a, b} sum_{u, v} x(a+u, b+v) x(a+(u+du mod m), b+(v+dv mod n))

func Matrix(g *whog.Covar, m, n int) *mat.Mat {
	return MatrixMode(g, m, n, Convex)
}

func MatrixMode(g *whog.Covar, m, n int, coeffs CoeffsFunc) *mat.Mat {
	c := g.Channels
	A := mat.New(m*n*c, m*n*c)
	// Populate matrix.
	for u := 0; u < m; u++ {
		for v := 0; v < n; v++ {
			for p := 0; p < c; p++ {
				for i := 0; i < m; i++ {
					for j := 0; j < n; j++ {
						for q := 0; q < c; q++ {
							du, dv := i-u, j-v
							a, b := coeffs(du, dv, m, n)

							var h float64
							h += (1 - a) * (1 - b) * g.At(mod(du, m), mod(dv, n), p, q)
							h += (1 - a) * b * g.At(mod(du, m), -mod(-dv, n), p, q)
							h += a * (1 - b) * g.At(-mod(-du, m), mod(dv, n), p, q)
							h += a * b * g.At(-mod(-du, m), -mod(-dv, n), p, q)

							row := (u*n+v)*c + p
							col := (i*n+j)*c + q
							A.Set(row, col, h)
						}
					}
				}
			}
		}
	}
	return A
}
