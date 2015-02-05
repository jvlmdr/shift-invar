package main

import (
	"flag"
	"fmt"
	"image"
	"log"

	"github.com/jvlmdr/go-cv/featset"
	"github.com/jvlmdr/go-cv/hog"
	"github.com/jvlmdr/go-pbs-pro/dstrfn"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/nfnt/resize"
)

func main() {
	var (
		datasetName = flag.String("dataset", "", "{inria, caltech}")
		datasetSpec = flag.String("dataset-spec", "", "Dataset parameters (JSON)")
		numFolds    = flag.Int("folds", 5, "Cross-validation folds")
		// Positive example configuration.
		pad           = flag.Int("pad", 0, "Dilate bounding box to obtain region from which features are extracted")
		aspectReject  = flag.Float64("reject-aspect", 0, "Reject examples not between r and 1/r times aspect ratio")
		resizeFor     = flag.String("resize-for", "area", "One of {area, width, height, fit, fill}")
		maxTrainScale = flag.Float64("max-train-scale", 2, "Discount examples which would need to be scaled more than this")
		biasCoeff     = flag.Float64("bias-coeff", 1, "Bias coefficient, zero for no bias")
		flip          = flag.Bool("flip", false, "Incorporate horizontally mirrored examples?")
		trainInterp   = flag.Int("train-interp", 1, "Interpolation for multi-scale search (0=nearest, 1=linear, 2=cubic)")
	)
	flag.Parse()
	dstrfn.ExecIfSlave()

	set := &ParamSet{
		Lambda:  []float64{1e-4, 1e-2, 1, 1e2},
		Gamma:   []float64{0.1, 0.3, 0.5, 0.7, 0.9},
		Epochs:  []int{2, 4},
		NegFrac: []float64{0.1, 0.2},
		Overlap: []OverlapMessage{
			{"inter-over-union", InterOverUnion{0.3}},
			{"inter-over-min", InterOverMin{0.65}},
		},
		Size: []image.Point{{32, 96}},
		Feat: []featset.ImageMarshaler{
			{"hog", hog.Transform{hog.FGMRConfig(4)}},
			{"hog", hog.Transform{hog.FGMRConfig(8)}},
		},
	}

	exampleOpts := data.ExampleOpts{
		AspectReject: *aspectReject,
		FitMode:      *resizeFor,
		MaxScale:     *maxTrainScale,
	}

	params := enumerateParams(set)
	for i, p := range params {
		fmt.Println(i, p.Hash(), p.ID())
	}

	// Load data and determine cross-validation splits.
	// Use same partitions for all methods.
	dataset, err := data.Load(*datasetName, *datasetSpec)
	if err != nil {
		log.Fatal(err)
	}
	// Split images into folds.
	// TODO: Cache splits due to their randomness.
	foldIms := split(dataset.Images(), *numFolds)

	trainInputs := make([]TrainInput, 0, len(foldIms)*len(params))
	for fold := range foldIms {
		for _, p := range params {
			trainInputs = append(trainInputs, TrainInput{fold, p})
		}
	}

	// Train a detector for every configuration.
	// Save weights to file and avoid re-computing weights
	// for configurations which have a file.
	var tmpls []string
	// TODO: Identify subset of inputs for which output file does not exist.
	err = dstrfn.MapFunc("train", &tmpls, trainInputs, foldIms, *datasetName, *datasetSpec, *pad, exampleOpts, *biasCoeff, *flip, resize.InterpolationFunction(*trainInterp))
	if err != nil {
		log.Fatalln("map(train):", err)
	}

	//	// Test each detector on training and validation sets.
	//	// Skip cached results.
	//	dstrfn.Map()

	//	// Dump all results to text file.
}
