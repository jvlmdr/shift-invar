package imset

import (
	"image"

	"github.com/jvlmdr/go-cv/rimg64"
)

// Set is a set of images of the same size.
type Set interface {
	Len() int
	ImageSize() image.Point
	ImageChannels() int
	At(int) *rimg64.Multi
}
