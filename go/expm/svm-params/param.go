package main

import (
	"crypto/sha1"
	"encoding/json"
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
	// Use JSON string instead of fmt.Sprintf("%+v") since pointers
	// will not traversed and their addresses will be displayed.
	repr, err := json.Marshal(p)
	if err != nil {
		panic(fmt.Sprintf("encode struct: %v", err))
	}
	return string(repr)
}

func (p Param) Hash() string {
	return fmt.Sprintf("%x", sha1.Sum([]byte(p.ID())))[:8]
}
