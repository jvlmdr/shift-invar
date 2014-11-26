package whog

import (
	"testing"

	"github.com/gonum/floats"
)

func testInvMul(t *testing.T, eps float64, width, height, bandwidth, channels int, lambda float64) {
	cov := randCovar(channels, bandwidth)
	cov.AddLambdaI(lambda)
	x := randImage(width, height, channels)

	y := Mul(cov, x)
	z, err := InvMulDirect(cov, y)
	if err != nil {
		t.Fatal(err)
	}

	delta := make([]float64, len(x.Elems))
	floats.SubTo(delta, x.Elems, z.Elems)

	r := floats.Norm(delta, 2) / floats.Norm(z.Elems, 2)
	if r > eps {
		t.Errorf("residual too large: want %g <= %g)", r, eps)
	}
	testImageEq(t, x, z)
}

func TestInvMulDirect(t *testing.T) {
	const (
		eps       = 1e-10
		width     = 8
		height    = 16
		bandwidth = 20
		channels  = 3
		lambda    = 1e-2
	)
	testInvMul(t, eps, width, height, bandwidth, channels, lambda)
}
