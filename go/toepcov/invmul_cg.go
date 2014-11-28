package toepcov

import (
	"io"

	"github.com/jvlmdr/go-cg/cg"
	"github.com/jvlmdr/go-cv/rimg64"
)

// InvMulConjGrad solves for x in S x = b using conjugate gradient.
// The fast multiplication routine in the Fourier domain is used to compute products with the matrix S.
// x is an initial guess, tol and iter are specified in package cg.
func InvMulConjGrad(g *Covar, b, x *rimg64.Multi, tol float64, iter int, debug io.Writer) (*rimg64.Multi, error) {
	var muler MulerFFT
	muler.Init(g, b.Width, b.Height)
	a := func(x []float64) []float64 {
		f := &rimg64.Multi{x, b.Width, b.Height, b.Channels}
		g := muler.Mul(f)
		return g.Elems
	}
	elems, err := cg.Solve(a, b.Elems, x.Elems, tol, iter, debug)
	if err != nil {
		return nil, err
	}
	y := &rimg64.Multi{elems, b.Width, b.Height, b.Channels}
	return y, nil
}
