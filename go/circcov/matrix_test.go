package circcov

import "testing"

func TestMatrix_symm(t *testing.T) {
	const (
		channels  = 5
		bandwidth = 10
		width     = 8
		height    = 4
	)

	cov := randCovar(channels, bandwidth)
	a := Matrix(cov, width, height)
	if !isSymm(a) {
		t.Errorf("not symmetric")
	}
}

func TestMatrixMode_symm(t *testing.T) {
	const (
		channels  = 5
		bandwidth = 10
		width     = 8
		height    = 4
	)
	modes := []struct {
		Name string
		Func CoeffsFunc
	}{
		{"convex", Convex},
		{"mean", Mean},
		{"nearest", Nearest},
	}

	cov := randCovar(channels, bandwidth)
	for _, mode := range modes {
		a := MatrixMode(cov, width, height, mode.Func)
		if !isSymm(a) {
			t.Errorf(`not symmetric: mode "%s"`, mode.Name)
		}
	}
}
