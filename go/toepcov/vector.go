package toepcov

import (
	"fmt"

	"github.com/jvlmdr/go-cv/rimg64"
)

func errIfSizeNotEq(a, b *rimg64.Multi) error {
	if a.Width != b.Width || a.Height != b.Height || a.Channels != b.Channels {
		return fmt.Errorf("sizes not equal: %s, %s", sizeStr(a), sizeStr(b))
	}
	return nil
}

func sizeStr(x *rimg64.Multi) string {
	return fmt.Sprintf("%dx%dx%d", x.Width, x.Height, x.Channels)
}
