package data

import (
	"encoding/json"
	"fmt"
	"image"

	"github.com/jvlmdr/go-cv/dataset/caltechped"
)

// ImageSet is a set of annotated images which can be
// used to generate training or testing data for
// cross-validation.
type ImageSet interface {
	Images() []string
	// The path to an image.
	File(im string) string
	// Can the image be used in a training and/or testing set?
	// Some datasets contain images that should
	// be used for training but never for testing.
	// Note that it is up to the user to keep the
	// training and testing sets separate.
	// It is possible that CanTrain() == true but
	// the image contains no examples since
	// IsNeg() == false and Annot().Instances is empty.
	CanTrain(im string) bool
	CanTest(im string) bool
	// Can every window in the image be a negative example?
	IsNeg(im string) bool
	// Does the image contain any positive examples?
	// If so, it can be used as a positive training image.
	// Note that IsNeg() == true implies that
	// Annot().Instances is empty but not the reverse.
	Annot(im string) Annot
}

// Annot gives the annotation of an image
// for the purpose of evaluating a single detector.
type Annot struct {
	// Things which should be detected.
	Instances []image.Rectangle
	// Regions in which it does not matter whether
	// the detector fires or not.
	Ignore []image.Rectangle
}

func Load(name, specJSON string) (ImageSet, error) {
	switch name {
	case "inria":
		var spec INRIASpec
		if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
			return nil, err
		}
		return loadINRIA(spec)
	case "caltech-preset":
		var preset CaltechPreset
		if err := json.Unmarshal([]byte(specJSON), &preset); err != nil {
			return nil, err
		}
		spec := preset.Spec()
		return loadCaltech(spec, caltechped.Reasonable)
	default:
		panic(fmt.Sprintf("unknown dataset: %s", name))
	}
}
