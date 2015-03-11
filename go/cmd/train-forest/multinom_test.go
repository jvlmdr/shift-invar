package main

import (
	"testing"

	"code.google.com/p/probab/dst"
)

var cases = [][]float64{
	{1, 1, 1, 1},
	{1, 2, 3},
	{3, 2, 1},
	{1},
}

func sqr(x float64) float64 { return x * x }

func TestMultinom(t *testing.T) {
	// Number of samples.
	const n = 10000
	// Null hypothesis: Multinomial distribution specified.
	// Check if there is little or no evidence against null hypothesis.
	const pval = 0.1

	for _, ws := range cases {
		k := len(ws)
		x := make([]int, k)
		for i := 0; i < n; i++ {
			x[Multinom(ws)]++
		}
		var total float64
		for _, m := range ws {
			total += float64(m)
		}
		// Reject this hypothessis
		var v float64
		for j := range x {
			e := ws[j] / total * float64(n)
			v += sqr(float64(x[j])-e) / e
		}
		th := dst.ChiSquareQtl(int64(k))(1 - pval)
		if v > th {
			t.Errorf("chi-square value %.3g, threshold %.3g (p = %g)", v, th, pval)
		}
	}
}

func TestMultinomSum(t *testing.T) {
	// Number of samples.
	const n = 10000
	// Null hypothesis: Multinomial distribution specified.
	// Check if there is little or no evidence against null hypothesis.
	const pval = 0.1

	for _, ws := range cases {
		k := len(ws)
		sum := CumSum(ws)
		x := make([]int, k)
		for i := 0; i < n; i++ {
			x[MultinomSum(sum)]++
		}
		var total float64
		for _, m := range ws {
			total += float64(m)
		}
		// Reject this hypothessis
		var v float64
		for j := range x {
			e := ws[j] / total * float64(n)
			v += sqr(float64(x[j])-e) / e
		}
		th := dst.ChiSquareQtl(int64(k))(1 - pval)
		if v > th {
			t.Errorf("chi-square value %.3g, threshold %.3g (p = %g)", v, th, pval)
		}
	}
}
