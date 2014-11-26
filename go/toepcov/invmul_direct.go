package whog

import (
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/lin-go/lapack"

	"fmt"
	"log"
)

// Computes S \ f using LAPACK (Cholesky factorization).
func InvMulDirect(cov *Covar, f *rimg64.Multi) (*rimg64.Multi, error) {
	if f.Channels != cov.Channels {
		panic(fmt.Sprintf(
			"different number of channels: covar %d, image %d",
			cov.Channels, f.Channels,
		))
	}

	// Instantiate full covariance matrix.
	s := cov.Matrix(f.Width, f.Height)

	m, n := s.Dims()
	log.Printf("solve %dx%d linear system", m, n)
	x, err := lapack.SolvePosDef(s, f.Elems)
	if err != nil {
		return nil, err
	}
	w := &rimg64.Multi{x, f.Width, f.Height, f.Channels}
	return w, nil
}
