package main

import (
	"fmt"
)

type Experiment struct {
	TrainDataset string // Either "train" or "test".
	TestDataset  string
	SubsetPairs  []TrainTestPair
}

type Dataset struct {
	Name, Spec string
}

type TrainTestPair struct {
	Train, Test string
}

type Set struct {
	Dataset, Subset string
}

func (x Set) Key() string {
	return fmt.Sprintf("%s-%s", x.Dataset, x.Subset)
}

// DetectorKey is the Cartesian product of detector parameters
// and a training set identifier.
type DetectorKey struct {
	Param
	TrainSet Set
}

func (x DetectorKey) Key() string {
	return fmt.Sprintf("param-%s-%s", x.Param.Key(), x.TrainSet.Key())
}

func (x DetectorKey) TmplFile() string {
	return fmt.Sprintf("tmpl-%s.gob", x.Key())
}

type ResultsKey struct {
	DetectorKey
	TestSet Set
}

func (x ResultsKey) Key() string {
	return fmt.Sprintf("param-%s-%s-%s", x.Param.Key(), x.TrainSet.Key(), x.TestSet.Key())
}

func (x ResultsKey) PerfFile() string {
	return fmt.Sprintf("perf-%s.json", x.Key())
}
