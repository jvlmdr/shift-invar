package main

import (
	"fmt"
)

// Experiment specified by a collection of training and testing sets.
// The mean and standard deviation of the results will be reported.
// It is assumed that all training sets are subsets of one dataset,
// and the same for the testing sets.
// Each set of examples is identified by a Dataset string and a Subset string.
type Experiment struct {
	TrainDataset string // Either "train" or "test".
	TestDataset  string
	SubsetPairs  []SubsetPair
}

// SubsetPair contains the name of a training and testing subset.
// The subsets may belong to distinct datasets.
type SubsetPair struct {
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
