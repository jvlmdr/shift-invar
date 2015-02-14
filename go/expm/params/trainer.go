package main

import (
	"encoding/json"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/nfnt/resize"
)

// Trainer takes a training set which was extracted
// using some configuration for training examples.
type Trainer interface {
	Train(examples *data.TrainingSet, dataset data.ImageSet, phi feat.Image, region detect.PadRect, flip bool, interp resize.InterpolationFunction) (*detect.FeatTmpl, error)
	Field(string) string
}

// TrainerSet describes a set of Trainers of the same type.
type TrainerSet interface {
	Enumerate() []Trainer
	Fields() []string
}

var DefaultTrainers = NewTrainerFactory()

func init() {
	DefaultTrainers.Register("svm",
		func() (Trainer, error) { return new(SVMTrainer), nil },
		func() (TrainerSet, error) { return new(SVMTrainerSet), nil },
	)
}

type trainerType struct {
	create    func() (Trainer, error)
	createSet func() (TrainerSet, error)
}

type TrainerFactory struct {
	types map[string]trainerType
}

func NewTrainerFactory() *TrainerFactory {
	f := new(TrainerFactory)
	f.types = make(map[string]trainerType)
	return f
}

func (f *TrainerFactory) Register(name string, create func() (Trainer, error), createSet func() (TrainerSet, error)) {
	f.types[name] = trainerType{create, createSet}
}

func (f *TrainerFactory) New(name string) (Trainer, error) {
	return f.types[name].create()
}

func (f *TrainerFactory) NewSet(name string) (TrainerSet, error) {
	return f.types[name].createSet()
}

type TrainerMessage struct {
	Type string
	Spec Trainer
}

func (t *TrainerMessage) UnmarshalJSON(data []byte) error {
	// Umarshal type from message.
	var raw struct {
		Type string
		Spec json.RawMessage
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	t.Type = raw.Type
	trainer, err := DefaultTrainers.New(raw.Type)
	if err != nil {
		return err
	}
	// Initialize and re-unmarshal.
	t.Spec = trainer
	return json.Unmarshal(raw.Spec, t.Spec)
}

type TrainerSetMessage struct {
	Type string
	Spec TrainerSet
}

func (t *TrainerSetMessage) UnmarshalJSON(data []byte) error {
	// Umarshal type from message.
	var raw struct {
		Type string
		Spec json.RawMessage
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	t.Type = raw.Type
	set, err := DefaultTrainers.NewSet(raw.Type)
	if err != nil {
		return err
	}
	// Initialize and re-unmarshal.
	t.Spec = set
	return json.Unmarshal(raw.Spec, t.Spec)
}