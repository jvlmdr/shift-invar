package main

import "math/rand"

// Split divides x randomly into n groups.
func split(x []string, n int) [][]string {
	y := make([][]string, n)
	for _, xi := range x {
		r := rand.Intn(n)
		y[r] = append(y[r], xi)
	}
	return y
}

func mergeExcept(x [][]string, j int) []string {
	var y []string
	for i, xi := range x {
		if i == j {
			continue
		}
		y = append(y, xi...)
	}
	return y
}
