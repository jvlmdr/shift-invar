package circcov

import "testing"

// Checks that InvMul(Mul(.)) is identity.
func TestInvMuler(t *testing.T) {
	const (
		bandwidth = 50
		width     = 40
		height    = 30
		channels  = 4
		eps       = 1e-6
	)

	g := randCovar(channels, bandwidth)
	f1 := randImage(width, height, channels)
	f2 := randImage(width, height, channels)
	gf1 := Mul(g, f1)
	gf2 := Mul(g, f2)

	var muler InvMuler
	muler.Init(g, width, height)
	x1 := muler.Mul(gf1)
	if eq, msg := imagesEq(f1, x1); !eq {
		t.Error(msg)
	}
	x2 := muler.Mul(gf2)
	if eq, msg := imagesEq(f2, x2); !eq {
		t.Error(msg)
	}
}
