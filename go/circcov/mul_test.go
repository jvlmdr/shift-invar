package circcov

import (
	"testing"

	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/lin-go/mat"
)

func TestMul(t *testing.T) {
	const (
		bandwidth = 50
		width     = 40
		height    = 30
		channels  = 4
	)

	g := randCovar(channels, bandwidth)
	f := randImage(width, height, channels)
	// Multiply in Fourier domain.
	got := Mul(g, f)
	// Multiply in time domain.
	A := Matrix(g, f.Width, f.Height)
	want := matMulImage(A, f)
	if eq, msg := imagesEq(want, got); !eq {
		t.Error(msg)
	}
}

func TestMulMode(t *testing.T) {
	const (
		bandwidth = 50
		width     = 40
		height    = 30
		channels  = 4
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
		// Multiply in Fourier domain.
		got := MulMode(g, f, mode.Func)
		// Multiply in time domain.
		A := MatrixMode(g, f.Width, f.Height, mode.Func)
		want := matMulImage(A, f)
		if eq, msg := imagesEq(want, got); !eq {
			t.Errorf(`mode "%s": %s`, mode.Name, msg)
		}
	}
}

func TestMul_addLambdaI(t *testing.T) {
	const (
		bandwidth = 50
		width     = 40
		height    = 30
		channels  = 4
		lambda    = 0.5
	)

	g := randCovar(channels, bandwidth)
	f := randImage(width, height, channels)
	// Multiply in Fourier domain.
	want := Mul(g, f)
	want = want.Plus(f.Scale(lambda))
	g.AddLambdaI(lambda)
	got := Mul(g, f)
	if eq, msg := imagesEq(want, got); !eq {
		t.Error(msg)
	}
}

func matMulImage(A *mat.Mat, f *rimg64.Multi) *rimg64.Multi {
	pix := mat.MulVec(A, f.Elems)
	return &rimg64.Multi{pix, f.Width, f.Height, f.Channels}
}
