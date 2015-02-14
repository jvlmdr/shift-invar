package main

import (
	"crypto/sha1"
	"encoding/json"
	"fmt"
	"image"
	"strconv"
	"strings"

	"github.com/jvlmdr/go-cv/featset"
)

type Param struct {
	Trainer TrainerMessage
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

func (p Param) Field(name string) string {
	if strings.HasPrefix(name, "Trainer.") {
		name = strings.TrimPrefix(name, "Trainer.")
		return p.Trainer.Spec.Field(name)
	}
	switch name {
	case "NegFrac":
		return fmt.Sprint(p.NegFrac)
	case "Overlap":
		return strconv.Quote(p.Overlap.Spec.Name())
	case "Size":
		return strconv.Quote(p.Size.String())
	case "Feat":
		return strconv.Quote(fmt.Sprint(p.Feat))
	default:
		return ""
	}
}
