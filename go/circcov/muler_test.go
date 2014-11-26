package circcov

import "testing"

// Checks that muler.Mul() matches Mul().
func TestMuler(t *testing.T) {
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
	y1 := Mul(g, f1)
	y2 := Mul(g, f2)

	var muler Muler
	muler.Init(g, width, height)
	x1 := muler.Mul(f1)
	if eq, msg := imagesEq(y1, x1); !eq {
		t.Error(msg)
	}
	x2 := muler.Mul(f2)
	if eq, msg := imagesEq(y2, x2); !eq {
		t.Error(msg)
	}
}
