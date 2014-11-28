package circcov

import (
	"testing"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/lin-go/lapack"
	"github.com/jvlmdr/lin-go/mat"
)

// Checks that InvMul(Mul(.)) is identity.
func TestInvMul_mul(t *testing.T) {
	const (
		bandwidth = 50
		width     = 40
		height    = 30
		channels  = 4
		eps       = 1e-6
	)

	g := randCovar(channels, bandwidth)
	f := randImage(width, height, channels)
	gf := Mul(g, f)
	got := InvMul(g, gf)
	if eq, msg := imagesEq(f, got); !eq {
		t.Error(msg)
	}
}

// Checks that InvMul(Mul(.)) is identity.
func TestInvMulMode_mul(t *testing.T) {
	const (
		bandwidth = 50
		width     = 40
		height    = 30
		channels  = 4
		eps       = 1e-6
	)
	modes := []struct {
		Name string
		Func CoeffsFunc
	}{
		{"convex", Convex},
		{"mean", Mean},
		{"nearest", Nearest},
	}

	g := randCovar(channels, bandwidth)
	f := randImage(width, height, channels)
	for _, mode := range modes {
		gf := MulMode(g, f, mode.Func)
		got := InvMulMode(g, gf, mode.Func)
		if eq, msg := imagesEq(f, got); !eq {
			t.Errorf(`mode "%s": %s`, mode.Name, msg)
		}
	}
}

// Checks that we get the same solution by solving the time-domain equations.
func TestInvMul_vsMat(t *testing.T) {
	const (
		bandwidth = 50
		width     = 40
		height    = 30
		channels  = 4
		eps       = 1e-6
	)

	g := randCovar(channels, bandwidth)
	f := randImage(width, height, channels)

	// Solve in Fourier domain.
	got := InvMul(g, f)

	// Solve in time domain.
	A := Matrix(g, f.Width, f.Height)
	t.Logf("matrix is %dx%d", A.Rows, A.Cols)
	want, err := matInvMulImage(A, f)
	if err != nil {
		t.Fatal(err)
	}

	if eq, msg := imagesEq(want, got); !eq {
		t.Error(msg)
	}
}

func matInvMulImage(A *mat.Mat, f *rimg64.Multi) (*rimg64.Multi, error) {
	pix, err := lapack.SolvePosDef(A, f.Elems)
	if err != nil {
		return nil, err
	}
	return &rimg64.Multi{pix, f.Width, f.Height, f.Channels}, nil
}
