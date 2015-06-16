package main

import "image"

type ParamSet struct {
	TrainerSets []TrainerSetMessage
	NegFrac     []float64
	Overlap     []OverlapMessage
	Size        []image.Point
	Feat        []Feature

	TrainPad      []int
	AspectReject  []float64
	ResizeFor     []string
	MaxTrainScale []float64
	PyrStep       []float64
	MaxTestScale  []float64
	TestMargin    []int
}

func (set *ParamSet) Fields() []string {
	trainerFields := make([][]string, len(set.TrainerSets))
	for i, t := range set.TrainerSets {
		trainerFields[i] = t.Spec.Fields()
	}
	fields := []string{"NegFrac", "Overlap", "Size", "Feat", "Trainer.Type"}
	for _, field := range union(trainerFields...) {
		fields = append(fields, "Trainer."+field)
	}
	fields = append(fields, "TrainPad", "AspectReject", "ResizeFor", "MaxTrainScale")
	fields = append(fields, "PyrStep", "MaxTestScale", "TestMargin")
	return fields
}

func (set *ParamSet) Enumerate() []Param {
	var ps []Param
	// TODO: Not this.
	for _, trainerSet := range set.TrainerSets {
		for _, trainerSpec := range trainerSet.Spec.Enumerate() {
			// Use the Type of TrainerSetMessage for TrainerMessage.
			trainer := TrainerMessage{trainerSet.Type, trainerSpec}
			for _, negFrac := range set.NegFrac {
				for _, overlap := range set.Overlap {
					for _, size := range set.Size {
						for _, feat := range set.Feat {
							for _, trainPad := range set.TrainPad {
								for _, aspectReject := range set.AspectReject {
									for _, resizeFor := range set.ResizeFor {
										for _, maxTrainScale := range set.MaxTrainScale {
											for _, pyrStep := range set.PyrStep {
												for _, maxTestScale := range set.MaxTestScale {
													for _, testMargin := range set.TestMargin {
														p := Param{
															Trainer:       trainer,
															NegFrac:       negFrac,
															Overlap:       overlap,
															Size:          size,
															Feat:          feat,
															TrainPad:      trainPad,
															AspectReject:  aspectReject,
															ResizeFor:     resizeFor,
															MaxTrainScale: maxTrainScale,
															PyrStep:       pyrStep,
															MaxTestScale:  maxTestScale,
															TestMargin:    testMargin,
														}
														ps = append(ps, p)
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	return ps
}

//	type TrainerSetList []TrainerSetMessage
//
//	func (list TrainerSetList) Enumerate() interface{} {
//	}
//
//	type Enumerater interface {
//		Enumerate() interface{}
//	}
