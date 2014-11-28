package circcov

// CoeffsFunc is the type of a function which
// specifies x and y mixture coefficients for constructing
// a circulant covariance matrix from a Toeplitz matrix.
//
// Elements are combined according to the following expression,
// where g is the Toeplitz matrix and h is the circulant matrix.
// 	var t float64
// 	t += (1 - a) * (1 - b) * g.At(mod(du, m), mod(dv, n), p, q)
// 	t += (1 - a) * b * g.At(mod(du, m), -mod(-dv, n), p, q)
// 	t += a * (1 - b) * g.At(-mod(-du, m), mod(dv, n), p, q)
// 	t += a * b * g.At(-mod(-du, m), -mod(-dv, n), p, q)
// 	h.Set(du, dv, p, q)
type CoeffsFunc func(du, dv, m, n int) (a, b float64)

// Convex returns mixture coefficients for convex combination.
func Convex(du, dv, m, n int) (float64, float64) {
	a := float64(mod(du, m)) / float64(m)
	b := float64(mod(dv, n)) / float64(n)
	return a, b
}

// Mean returns mixture coefficients for equal combination.
func Mean(du, dv, m, n int) (float64, float64) {
	return 0.5, 0.5
}

// Nearest returns mixture coefficients to select
// the element of the Toeplitz matrix with the nearest boundary.
func Nearest(du, dv, m, n int) (float64, float64) {
	a := stepFrac(du, m)
	b := stepFrac(dv, n)
	return a, b
}

// Returns
//	0   if du/m < 0.5
//	1   if du/m > 0.5
//	0.5 if du/m = 0.5
func stepFrac(du, m int) float64 {
	du = mod(du, m)
	switch {
	case du < m/2:
		return 0
	case du > m/2:
		return 1
	default:
		return 0.5
	}
}
