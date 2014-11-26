package whog

import (
	"github.com/gonum/floats"
	"github.com/jvlmdr/go-cv/rimg64"

	"fmt"
	"math"
)

func errIfSizeNotEq(a, b *rimg64.Multi) error {
	if a.Width != b.Width || a.Height != b.Height || a.Channels != b.Channels {
		return fmt.Errorf("sizes not equal: %s, %s", sizeStr(a), sizeStr(b))
	}
	return nil
}

func panicIf(err error) {
	if err != nil {
		panic(err)
	}
}

func sizeStr(x *rimg64.Multi) string {
	return fmt.Sprintf("%dx%dx%d", x.Width, x.Height, x.Channels)
}

func plus(a, b *rimg64.Multi) *rimg64.Multi {
	panicIf(errIfSizeNotEq(a, b))
	c := rimg64.NewMulti(a.Width, a.Height, a.Channels)
	floats.AddTo(c.Elems, a.Elems, b.Elems)
	return c
}

func minus(a, b *rimg64.Multi) *rimg64.Multi {
	panicIf(errIfSizeNotEq(a, b))
	c := rimg64.NewMulti(a.Width, a.Height, a.Channels)
	floats.SubTo(c.Elems, a.Elems, b.Elems)
	return c
}

func scale(k float64, x *rimg64.Multi) *rimg64.Multi {
	y := x.Clone()
	copy(y.Elems, x.Elems)
	floats.Scale(k, y.Elems)
	return y
}

func sqrnorm(x *rimg64.Multi) float64 {
	return dot(x, x)
}

func norm(x *rimg64.Multi) float64 {
	return math.Sqrt(sqrnorm(x))
}

func dot(x, y *rimg64.Multi) float64 {
	panicIf(errIfSizeNotEq(x, y))
	return dotSlice(x.Elems, y.Elems)
}

func dotSlice(x, y []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	if len(x) == 1 {
		return x[0] * y[0]
	}
	m := (len(x) + 1) / 2
	return dotSlice(x[:m], y[:m]) + dotSlice(x[m:], y[m:])
}

func cosine(x, y *rimg64.Multi) float64 {
	return dot(x, y) / (norm(x) * norm(y))
}
