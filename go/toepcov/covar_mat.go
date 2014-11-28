package toepcov

// CovarMatrix treats the Toeplitz covariance as a simple matrix
// without instantiating it.
type CovarMatrix struct {
	Gamma  *Covar
	Width  int
	Height int
}

// Dims returns the dimensions of the matrix.
func (s CovarMatrix) Dims() (rows, cols int) {
	n := s.Width * s.Height * s.Gamma.Channels
	return n, n
}

// At accesses the element at row i and column j.
func (s CovarMatrix) At(i, j int) float64 {
	height := s.Height
	channels := s.Gamma.Channels

	x := i / (height * channels)
	y := (i / channels) % height
	p := i % channels

	u := (j / height) / channels
	v := (j / channels) % height
	q := j % channels

	if abs(x-u) > s.Gamma.Bandwidth {
		return 0
	}
	if abs(y-v) > s.Gamma.Bandwidth {
		return 0
	}
	return s.Gamma.At(u-x, v-y, p, q)
}

// SqrNorm is an efficient method for computing
// the squared Frobenius norm of the matrix.
func (s CovarMatrix) SqrNorm() float64 {
	bandwidth := s.Gamma.Bandwidth
	channels := s.Gamma.Channels
	var acc float64
	for u := -bandwidth; u <= bandwidth; u++ {
		for v := -bandwidth; v <= bandwidth; v++ {
			n := max(s.Width-abs(u), 0) * max(s.Height-abs(v), 0)
			for p := 0; p < channels; p++ {
				for q := 0; q < channels; q++ {
					a := s.Gamma.At(u, v, p, q)
					acc += a * a * float64(n)
				}
			}
		}
	}
	return acc
}

// Trace is an efficient method for computing
// the trace of the matrix.
func (s CovarMatrix) Trace() float64 {
	return s.Gamma.Trace(s.Width, s.Height)
}
