package circcov

import (
	"io"
	"log"

	"github.com/jvlmdr/go-cg/pcg"
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/shift-invar/go/toepcov"
)

// ToeplitzInvMulPCG solves for x in S x = b
// where S is the Toeplitz covariance matrix.
// The solution is obtained using pre-conditioned conjugate gradient,
// with a circulant preconditioner.
func ToeplitzInvMulPCG(g *toepcov.Covar, b, x *rimg64.Multi, tol float64, iter int, debug io.Writer) (*rimg64.Multi, error) {
	log.Println("ToeplitzInvMulPCG: init Muler")
	var muler toepcov.MulerFFT
	muler.Init(g, b.Width, b.Height)
	a := func(x []float64) []float64 {
		f := &rimg64.Multi{x, b.Width, b.Height, b.Channels}
		g := muler.Mul(f)
		return g.Elems
	}

	var invmuler InvMuler
	log.Println("ToeplitzInvMulPCG: init InvMuler")
	if err := invmuler.Init(g, b.Width, b.Height); err != nil {
		return nil, err
	}
	cinv := func(x []float64) []float64 {
		f := &rimg64.Multi{x, b.Width, b.Height, b.Channels}
		g := invmuler.Mul(f)
		return g.Elems
	}

	log.Println("ToeplitzInvMulPCG: solve PCG")
	elems, err := pcg.Solve(a, b.Elems, cinv, x.Elems, tol, iter, debug)
	if err != nil {
		return nil, err
	}
	log.Println("ToeplitzInvMulPCG: done")
	y := &rimg64.Multi{elems, b.Width, b.Height, b.Channels}
	return y, nil
}
