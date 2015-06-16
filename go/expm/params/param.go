package main

import (
	"crypto/sha1"
	"encoding/json"
	"fmt"
	"image"
	"reflect"
	"strings"

	"github.com/jvlmdr/go-cv/featset"
)

// Param specifies options for training and testing a detector.
type Param struct {
	Trainer TrainerMessage
	NegFrac float64
	Overlap OverlapMessage
	Size    image.Point
	Feat    Feature

	TrainPad      int
	AspectReject  float64
	ResizeFor     string
	MaxTrainScale float64
	PyrStep       float64
	MaxTestScale  float64
	TestMargin    int
}

// Serialize is a representation of Param as a string.
func (p Param) Serialize() string {
	// Use JSON string instead of fmt.Sprintf("%+v") since pointers
	// will not traversed and their addresses will be displayed.
	repr, err := json.Marshal(p)
	if err != nil {
		panic(fmt.Sprintf("encode struct: %v", err))
	}
	return string(repr)
}

// Ident is a short unique string.
func (p Param) Ident() string {
	return fmt.Sprintf("%x", sha1.Sum([]byte(p.Serialize())))[:8]
}

// Field returns the value of the field with the given name.
// Field names with the prefix  "Trainer." are passed
// to p.Trainer.Spec.Field() with the prefix removed.
func (p Param) Field(name string) string {
	if strings.HasPrefix(name, "Trainer.") {
		name = strings.TrimPrefix(name, "Trainer.")
		if name == "Type" {
			return p.Trainer.Type
		}
		return p.Trainer.Spec.Field(name)
	}
	switch name {
	case "Overlap":
		return p.Overlap.Spec.Name()
	case "Feat":
		repr, err := json.Marshal(p.Feat)
		if err != nil {
			panic(fmt.Sprintf("encode feature: %v", err))
		}
		return string(repr)
	}
	fieldValue := reflect.ValueOf(p).FieldByName(name)
	if !fieldValue.IsValid() {
		return ""
	}
	return fmt.Sprint(fieldValue.Interface())
}

// Feature is a feature function with pre-computed statistics.
type Feature struct {
	Transform featset.ImageMarshaler
	StatsFile string
}
