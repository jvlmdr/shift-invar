package toepcov

import "fmt"

// Summation over images so far.
type Total struct {
	// Sum over pixel values per channel.
	MeanTotal []float64
	// Sum over second-order interactions
	// per relative offset per channel pair.
	CovarTotal *Covar
	// Number of observations of each relative position.
	Count *Count
	// Number of images visited.
	Images int
}

// Combine two totals.
// Neither operand can be nil.
func AddTotal(lhs, rhs *Total) *Total {
	pixel := addPixel(lhs.MeanTotal, rhs.MeanTotal)
	covar := AddCovar(lhs.CovarTotal, rhs.CovarTotal)
	count := AddCount(lhs.Count, rhs.Count)
	n := lhs.Images + rhs.Images
	return &Total{pixel, covar, count, n}
}

// Combine two totals.
// One of the inputs will be modified.
// If either operand is nil, then the other is returned.
func AddTotalToEither(lhs, rhs *Total) *Total {
	if rhs == nil {
		return lhs
	}
	if lhs == nil {
		return rhs
	}
	pixel := addPixel(lhs.MeanTotal, rhs.MeanTotal)
	covar := AddCovarToEither(lhs.CovarTotal, rhs.CovarTotal)
	count := AddCount(lhs.Count, rhs.Count)
	n := lhs.Images + rhs.Images
	return &Total{pixel, covar, count, n}
}

// Normalizes the sums to obtain expected mean and covariance.
// Normalization is performed per-pixel,
// giving expectation per relative displacement
// but not guaranteeing a positive semidefinite matrix.
//
// It is not recommended to disable centering of the covariance matrix.
// This corresponds to the assumption that the mean is zero.
func Normalize(total *Total, center bool) *Distr {
	mean := scaleMean(1/float64(total.Count.At(0, 0)), total.MeanTotal)
	cov := normCovar(total.CovarTotal, total.Count)
	if center {
		cov.Center(mean)
	}
	return &Distr{mean, cov}
}

// Normalizes the sums to obtain expected mean and covariance.
// Normalization is performed uniformly, giving the exactly-stationary covariance matrix
// which would be obtained by computing the expectation
// over all translated windows in the periodic extension of the zero-padded image.
// Therefore the size must be specified,
// although this may differ from that used for the actual template.
// This matrix should be positive semidefinite and
// is recommended for a small number of small images,
// where Normalize() is more likely to produce an indefinite matrix.
//
// It is not recommended to disable centering of the covariance matrix.
// This corresponds to the assumption that the mean is zero.
func NormalizeUniform(total *Total, center bool, width, height int) *Distr {
	// Obtain sum over all image widths
	// by considering vertical shift of one pixel.
	n11 := total.Count.At(0, 0)
	M := n11 - total.Count.At(0, 1)
	N := n11 - total.Count.At(1, 0)
	n12 := int64(width-1) * N
	n21 := M * int64(height-1)
	union := n11 - total.Count.At(1, 1)
	p := int(M + N - union)
	if p != total.Images {
		err := fmt.Errorf("num images not supported by (1, 1): %d, %d", total.Images, p)
		panic(err)
	}
	n22 := int64(width-1) * int64(height-1) * int64(p)
	n := n11 + n12 + n21 + n22

	mean := scaleMean(1/float64(n), total.MeanTotal)
	cov := scaleCovar(1/float64(n), total.CovarTotal)
	if center {
		cov.Center(mean)
	}
	return &Distr{mean, cov}
}

func normCovar(total *Covar, count *Count) *Covar {
	cov := total.Clone()
	for i := -cov.Bandwidth; i <= cov.Bandwidth; i++ {
		for j := -cov.Bandwidth; j <= cov.Bandwidth; j++ {
			n := float64(count.At(i, j))
			for p := 0; p < cov.Channels; p++ {
				for q := 0; q < cov.Channels; q++ {
					cov.Set(i, j, p, q, cov.At(i, j, p, q)/n)
				}
			}
		}
	}
	return cov
}

func scaleCovar(alpha float64, cov *Covar) *Covar {
	cov = cov.Clone()
	for i := -cov.Bandwidth; i <= cov.Bandwidth; i++ {
		for j := -cov.Bandwidth; j <= cov.Bandwidth; j++ {
			for p := 0; p < cov.Channels; p++ {
				for q := 0; q < cov.Channels; q++ {
					cov.Set(i, j, p, q, alpha*cov.At(i, j, p, q))
				}
			}
		}
	}
	return cov
}

func scaleMean(alpha float64, x []float64) []float64 {
	y := make([]float64, len(x))
	for i := range x {
		y[i] = alpha * x[i]
	}
	return y
}
