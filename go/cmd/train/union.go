package main

import (
	"fmt"
	"sort"

	"github.com/jvlmdr/go-svm/svm"
)

type Union struct {
	Sets []svm.Set
	cdf  []int
}

func (u *Union) Len() int {
	return u.cdf[len(u.cdf)-1]
}

func (u *Union) Dim() int {
	if len(u.Sets) == 0 {
		panic("empty")
	}
	var n int
	for i, xi := range u.Sets {
		ni := xi.Dim()
		if i == 0 {
			n = ni
			continue
		}
		if ni != n {
			panic(fmt.Sprintf("dimension: found %d and %d", n, ni))
		}
	}
	return n
}

func (u *Union) At(i int) []float64 {
	// Find set which contains i-th example.
	s := sort.Search(len(u.Sets), func(s int) bool { return i < u.cdf[s+1] })
	// Index into set.
	t := i - u.cdf[s]
	return u.Sets[s].At(t)
}

func NewUnion(sets []svm.Set) *Union {
	u := new(Union)
	u.Sets = sets
	u.cdf = cumSum(setLens(sets))
	return u
}

func (u *Union) Append(set svm.Set) {
	u.Sets = append(u.Sets, set)
	u.cdf = append(u.cdf, u.cdf[len(u.cdf)-1]+set.Len())
}

func setLens(x []svm.Set) []int {
	n := make([]int, len(x))
	for i, xi := range x {
		n[i] = xi.Len()
	}
	return n
}

func cumSum(x []int) []int {
	s := make([]int, len(x)+1)
	for i, xi := range x {
		s[i+1] = s[i] + xi
	}
	return s
}
