package circcov

type CoeffsFunc func(du, dv, m, n int) (float64, float64)

// Returns mixture coefficients for convex combination.
func Convex(du, dv, m, n int) (float64, float64) {
	a := float64(mod(du, m)) / float64(m)
	b := float64(mod(dv, n)) / float64(n)
	return a, b
}

// Returns mixture coefficients for equal combination.
func Mean(du, dv, m, n int) (float64, float64) {
	return 0.5, 0.5
}

// Returns mixture coefficients to simply select the element of the Toeplitz matrix with the nearest boundary.
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
