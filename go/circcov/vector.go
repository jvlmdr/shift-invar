package circcov

import (
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/lin-go/vec"

	"math"
)

func plus(a, b *rimg64.Multi) *rimg64.Multi {
	if a.Width != b.Width || a.Height != b.Height || a.Channels != b.Channels {
		panic("different size")
	}
	c := rimg64.NewMulti(a.Width, a.Height, a.Channels)
	vec.Copy(vec.Slice(c.Elems), vec.Plus(vec.Slice(a.Elems), vec.Slice(b.Elems)))
	return c
}

func minus(a, b *rimg64.Multi) *rimg64.Multi {
	if a.Width != b.Width || a.Height != b.Height || a.Channels != b.Channels {
		panic("different size")
	}
	c := rimg64.NewMulti(a.Width, a.Height, a.Channels)
	vec.Copy(vec.Slice(c.Elems), vec.Minus(vec.Slice(a.Elems), vec.Slice(b.Elems)))
	return c
}

func scale(k float64, x *rimg64.Multi) *rimg64.Multi {
	y := rimg64.NewMulti(x.Width, x.Height, x.Channels)
	vec.Copy(vec.Slice(y.Elems), vec.Scale(k, vec.Slice(x.Elems)))
	return y
}

func sqrnorm(x *rimg64.Multi) float64 {
	return dot(x, x)
}

func norm(x *rimg64.Multi) float64 {
	return math.Sqrt(sqrnorm(x))
}

func dot(x, y *rimg64.Multi) float64 {
	if x.Width != y.Width || x.Height != y.Height || x.Channels != y.Channels {
		panic("different size")
	}
	return dotSlice(x.Elems, y.Elems)
}

func dotSlice(x, y []float64) float64 {
	if len(x) == 1 {
		return x[0] * y[0]
	}
	m := (len(x) + 1) / 2
	return dotSlice(x[:m], y[:m]) + dotSlice(x[m:], y[m:])
}

func cosine(x, y *rimg64.Multi) float64 {
	return dot(x, y) / (norm(x) * norm(y))
}
