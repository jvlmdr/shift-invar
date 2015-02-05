package main

import (
	"crypto/sha1"
	"fmt"
	"image"

	"github.com/jvlmdr/go-cv/featset"
)

type ParamSet struct {
	Lambda []float64
	Gamma  []float64
	Epochs []int
	// Universal.
	NegFrac []float64
	Overlap []OverlapMessage
	Size    []image.Point
	Feat    []featset.ImageMarshaler
}

// TODO: Make pad part of feat?

type Param struct {
	Lambda float64
	Gamma  float64
	Epochs int
	// Universal.
	NegFrac float64
	Overlap OverlapMessage
	Size    image.Point
	Feat    featset.ImageMarshaler
}

func (p Param) ID() string {
	return fmt.Sprintf("%+v", p)
}

func (p Param) Hash() string {
	return fmt.Sprintf("%x", sha1.Sum([]byte(p.ID())))[:8]
}
