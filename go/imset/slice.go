package imset

import (
	"fmt"
	"image"

	"github.com/jvlmdr/go-cv/rimg64"
)

type Slice []*rimg64.Multi

func (set Slice) Len() int {
	return len(set)
}

func (set Slice) ImageSize() image.Point {
	size := set[0].Size()
	for _, x := range set {
		if !x.Size().Eq(size) {
			panic(fmt.Sprintf("different size: found %v and %v", size, x.Size()))
		}
	}
	return size
}

func (set Slice) ImageChannels() int {
	channels := set[0].Channels
	for _, x := range set {
		if x.Channels == channels {
			panic(fmt.Sprintf("different channels: found %v and %v", channels, x.Channels))
		}
	}
	return channels
}

func (set Slice) At(i int) *rimg64.Multi {
	return set[i]
}
