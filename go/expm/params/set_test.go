package main

import "testing"

type Struct struct {
	X float64
	Y string
	Z int
	W int
}

func TestCombinations(t *testing.T) {
	set := struct {
		X []float64
		Y []string
		W []int
	}{
		X: []float64{0.707, 0.866},
		Y: []string{"one", "two", "three", "four"},
		W: []int{1, 2, 3},
	}
	init := []Struct{{X: -1, Y: "non-empty", Z: -2, W: -3}}
	got := enumerate(set, init, []string{"X", "Y", "W"}).([]Struct)

	want := map[Struct]bool{
		Struct{X: 0.707, Y: "one", Z: -2, W: 1}:   true,
		Struct{X: 0.707, Y: "one", Z: -2, W: 2}:   true,
		Struct{X: 0.707, Y: "one", Z: -2, W: 3}:   true,
		Struct{X: 0.707, Y: "two", Z: -2, W: 1}:   true,
		Struct{X: 0.707, Y: "two", Z: -2, W: 2}:   true,
		Struct{X: 0.707, Y: "two", Z: -2, W: 3}:   true,
		Struct{X: 0.707, Y: "three", Z: -2, W: 1}: true,
		Struct{X: 0.707, Y: "three", Z: -2, W: 2}: true,
		Struct{X: 0.707, Y: "three", Z: -2, W: 3}: true,
		Struct{X: 0.707, Y: "four", Z: -2, W: 1}:  true,
		Struct{X: 0.707, Y: "four", Z: -2, W: 2}:  true,
		Struct{X: 0.707, Y: "four", Z: -2, W: 3}:  true,
		Struct{X: 0.866, Y: "one", Z: -2, W: 1}:   true,
		Struct{X: 0.866, Y: "one", Z: -2, W: 2}:   true,
		Struct{X: 0.866, Y: "one", Z: -2, W: 3}:   true,
		Struct{X: 0.866, Y: "two", Z: -2, W: 1}:   true,
		Struct{X: 0.866, Y: "two", Z: -2, W: 2}:   true,
		Struct{X: 0.866, Y: "two", Z: -2, W: 3}:   true,
		Struct{X: 0.866, Y: "three", Z: -2, W: 1}: true,
		Struct{X: 0.866, Y: "three", Z: -2, W: 2}: true,
		Struct{X: 0.866, Y: "three", Z: -2, W: 3}: true,
		Struct{X: 0.866, Y: "four", Z: -2, W: 1}:  true,
		Struct{X: 0.866, Y: "four", Z: -2, W: 2}:  true,
		Struct{X: 0.866, Y: "four", Z: -2, W: 3}:  true,
	}
	if len(want) != len(got) {
		t.Errorf("number of elements: want %d, got %d", len(want), len(got))
	}
	for _, elem := range got {
		if !want[elem] {
			t.Errorf("unexpected: %v", elem)
			continue
		}
		delete(want, elem)
	}
	for elem := range want {
		t.Errorf("absent: %v", elem)
	}
}
