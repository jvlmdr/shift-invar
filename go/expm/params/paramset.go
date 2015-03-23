package main

import "image"

type ParamSet struct {
	TrainerSets []TrainerSetMessage
	NegFrac     []float64
	Overlap     []OverlapMessage
	Size        []image.Point
	Feat        []Feature
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
	return fields
}

func (set *ParamSet) Enumerate() []Param {
	var ps []Param
	for _, trainerSet := range set.TrainerSets {
		for _, trainerSpec := range trainerSet.Spec.Enumerate() {
			// Use the Type of TrainerSetMessage for TrainerMessage.
			trainer := TrainerMessage{trainerSet.Type, trainerSpec}
			for _, negFrac := range set.NegFrac {
				for _, overlap := range set.Overlap {
					for _, size := range set.Size {
						for _, feat := range set.Feat {
							p := Param{
								Trainer: trainer,
								NegFrac: negFrac,
								Overlap: overlap,
								Size:    size,
								Feat:    feat,
							}
							ps = append(ps, p)
						}
					}
				}
			}
		}
	}
	return ps
}
