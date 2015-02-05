package main

import (
	"encoding/json"
	"fmt"
	"image"
)

type Overlap interface {
	// Eval returns true if a and b overlap.
	Eval(a, b image.Rectangle) bool
	// Name gives a human-readable name for the criterion.
	Name() string
}

func init() {
	DefaultOverlaps.Types["inter-over-union"] = func() (Overlap, error) { return new(InterOverUnion), nil }
	DefaultOverlaps.Types["inter-over-min"] = func() (Overlap, error) { return new(InterOverMin), nil }
}

type InterOverUnion struct {
	Min float64
}

func (f InterOverUnion) Eval(a, b image.Rectangle) bool {
	inter := area(a.Intersect(b))
	union := area(a) + area(b) - inter
	if union == 0 {
		panic("both rectangles empty")
	}
	overlap := float64(inter) / float64(union)
	return overlap >= f.Min
}

func (f InterOverUnion) Name() string {
	return fmt.Sprintf("Inter/Union > %g", f.Min)
}

type InterOverMin struct {
	Min float64
}

func (f InterOverMin) Eval(a, b image.Rectangle) bool {
	inter := area(a.Intersect(b))
	min := min(area(a), area(b))
	if min == 0 {
		panic("at least one empty rectangle")
	}
	overlap := float64(inter) / float64(min)
	return overlap >= f.Min
}

func (f InterOverMin) Name() string {
	return fmt.Sprintf("Inter/Min > %g", f.Min)
}

// Want to be able to marshal after unmarshaling,
// as well as marshal from a programmatic construction.

type NewOverlapFunc func() (Overlap, error)

type OverlapFactory struct {
	Types map[string]NewOverlapFunc
}

func (fact *OverlapFactory) New(typ string) (Overlap, error) {
	if typ == "" {
		return nil, fmt.Errorf("no overlap type specified")
	}
	newOverlap, ok := fact.Types[typ]
	if !ok {
		return nil, fmt.Errorf("overlap type not found: %s", typ)
	}
	return newOverlap()
}

func NewOverlapFactory() *OverlapFactory {
	fact := new(OverlapFactory)
	fact.Types = make(map[string]NewOverlapFunc)
	return fact
}

// To use a specific factory to decode an Overlap,
// every type which has an Overlap member must be modified
// to initialize its member to the given factory.

var DefaultOverlaps = NewOverlapFactory()

type OverlapMessage struct {
	Type string
	Spec Overlap
}

func (f *OverlapMessage) Eval(a, b image.Rectangle) bool { return f.Spec.Eval(a, b) }
func (f *OverlapMessage) Name() string                   { return f.Spec.Name() }

func (f *OverlapMessage) UnmarshalJSON(data []byte) error {
	// Umarshal type from message.
	var raw struct {
		Type string
		Spec json.RawMessage
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	f.Type = raw.Type
	overlap, err := DefaultOverlaps.New(raw.Type)
	if err != nil {
		return err
	}
	// Initialize overlap and re-unmarshal.
	f.Spec = overlap
	return json.Unmarshal(raw.Spec, f.Spec)
}
