package toepcov

// Distr specifies a stationary distribution in terms of
// its mean and covariance.
type Distr struct {
	Mean  []float64
	Covar *Covar
}

// Center subtracts the mean from the covariance matrix.
// Modifies the covariance.
func (p *Distr) Center() {
	for k := 0; k < p.Covar.Channels; k++ {
		for w := 0; w < p.Covar.Channels; w++ {
			cross := p.Mean[k] * p.Mean[w]
			for i := -p.Covar.Bandwidth; i <= p.Covar.Bandwidth; i++ {
				for j := -p.Covar.Bandwidth; j <= p.Covar.Bandwidth; j++ {
					p.Covar.Set(i, j, k, w, p.Covar.At(i, j, k, w)-cross)
				}
			}
		}
	}
}
