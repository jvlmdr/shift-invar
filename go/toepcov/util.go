package whog

func max(a, b int) int {
	if a < b {
		return b
	}
	return a
}

func min(a, b int) int {
	if b < a {
		return b
	}
	return a
}

func abs(x int) int {
	return max(x, -x)
}

func mod(a, b int) int {
	if b <= 0 {
		panic("non-positive denominator")
	}
	return ((a % b) + b) % b
}
