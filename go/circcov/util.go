package circcov

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func mod(a, b int) int {
	if b <= 0 {
		panic("non-positive mod")
	}
	return ((a % b) + b) % b
}
