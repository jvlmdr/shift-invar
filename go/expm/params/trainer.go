package main

import (
	"encoding/json"
	"time"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/nfnt/resize"
)

// Trainer takes a training set which was extracted
// using some configuration for training examples.
type Trainer interface {
	Train(posIms, negIms []string, dataset data.ImageSet, phi feat.Image, statsFile string, region detect.PadRect, exampleOpts data.ExampleOpts, flip bool, interp resize.InterpolationFunction, searchOpts detect.MultiScaleOpts) (*SolveResult, error)
	Field(string) string
}

type TrainResult struct {
	Tmpl   *detect.FeatTmpl
	Report *TrainReport
}

// TrainReport contains metadata about the outcome of training.
type TrainReport struct {
	TotalDur time.Duration
	SolveDur SolveDuration
	Error    string
}

// SolveResult is the result of trying to solve the training problem.
// These are bundled together to avoid having multiple returns to Trainer.Train(),
// especially since some Trainers may not report a duration.
type SolveResult struct {
	Tmpl  *detect.FeatTmpl
	Dur   SolveDuration
	Error string
}

// Fail returns false iff Error is empty.
func (solveResult *SolveResult) Fail() bool {
	return solveResult.Error != ""
}

func NewTrainResult(solveResult *SolveResult, total time.Duration) *TrainResult {
	if solveResult.Fail() {
		return &TrainResult{
			Report: &TrainReport{Error: solveResult.Error},
		}
	}
	return &TrainResult{
		Tmpl:   solveResult.Tmpl,
		Report: &TrainReport{TotalDur: total, SolveDur: solveResult.Dur},
	}
}

type SolveDuration struct {
	Total, Subst time.Duration
}

// TrainerSet describes a set of Trainers of the same type.
type TrainerSet interface {
	// Trainers can use the search options.
	Enumerate() []Trainer
	Fields() []string
}

var DefaultTrainers = NewTrainerFactory()

func init() {
	DefaultTrainers.Register("svm",
		func() (Trainer, error) { return new(SVMTrainer), nil },
		func() (TrainerSet, error) { return new(SVMTrainerSet), nil },
	)
	DefaultTrainers.Register("set-svm",
		func() (Trainer, error) { return new(SetSVMTrainer), nil },
		func() (TrainerSet, error) { return new(SetSVMTrainerSet), nil },
	)
	DefaultTrainers.Register("hard-neg",
		func() (Trainer, error) { return new(HardNegTrainer), nil },
		func() (TrainerSet, error) { return new(HardNegTrainerSet), nil },
	)
	DefaultTrainers.Register("toeplitz",
		func() (Trainer, error) { return new(ToeplitzTrainer), nil },
		func() (TrainerSet, error) { return new(ToeplitzTrainerSet), nil },
	)
	DefaultTrainers.Register("toep-inv",
		func() (Trainer, error) { return new(ToepInvTrainer), nil },
		func() (TrainerSet, error) { return new(ToepInvTrainerSet), nil },
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
