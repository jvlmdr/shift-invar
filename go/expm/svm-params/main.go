package main

import (
	"flag"
	"fmt"
	"image"
	"log"
	"os"

	"github.com/gonum/floats"
	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/featset"
	"github.com/jvlmdr/go-cv/hog"
	"github.com/jvlmdr/go-file/fileutil"
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
		// Test configuration.
		pyrStep      = flag.Float64("pyr-step", 1.07, "Geometric scale steps in image pyramid")
		maxTestScale = flag.Float64("max-test-scale", 2, "Do not zoom in further than this")
		testInterp   = flag.Int("test-interp", 1, "Interpolation for multi-scale search (0=nearest, 1=linear, 2=cubic)")
		detsPerIm    = flag.Int("dets-per-im", 0, "Maximum number of detections per image")
		testMargin   = flag.Int("margin", 0, "Margin to add to image before taking features at test time")
		localMax     = flag.Bool("local-max", true, "Suppress detections which are less than a neighbor?")
		minMatch     = flag.Float64("min-match", 0.5, "Minimum intersection-over-union to validate a true positive")
		minIgnore    = flag.Float64("min-ignore", 0.5, "Minimum that a region can be covered to be ignored")
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
	// FPPIs at which to compute miss rate.
	fppis := make([]float64, 9)
	floats.LogSpan(fppis, 1e-2, 1)

	// Train configuration.
	exampleOpts := data.ExampleOpts{
		AspectReject: *aspectReject,
		FitMode:      *resizeFor,
		MaxScale:     *maxTrainScale,
	}

	// Test configuration.
	// Transform and Overlap are taken from Param.
	searchOpts := MultiScaleOpts{
		MaxScale:  *maxTestScale,
		PyrStep:   *pyrStep,
		Interp:    resize.InterpolationFunction(*testInterp),
		PadMargin: feat.UniformMargin(*testMargin),
		DetFilter: detect.DetFilter{
			LocalMax: *localMax,
			MinScore: 0,
		},
		SupprMaxNum: *detsPerIm,
		ExpMinScore: 0,
	}

	params := enumerateParams(set)
	for _, p := range params {
		fmt.Printf("%s\t%s\n", p.Hash(), p.ID())
	}

	// Load data and determine cross-validation splits.
	// Use same partitions for all methods.
	dataset, err := data.Load(*datasetName, *datasetSpec)
	if err != nil {
		log.Fatal(err)
	}
	// Split images into folds.
	// Cache splits due to their randomness.
	var foldIms [][]string
	err = fileutil.Cache(&foldIms, "folds.json", func() [][]string {
		return split(dataset.Images(), *numFolds)
	})
	if err != nil {
		log.Fatal(err)
	}

	// Learn template for each configuration for each fold.
	// Save weights to file and avoid re-computing weights
	// for configurations which have a file.
	trainInputs := make([]TrainInput, 0, len(foldIms)*len(params))
	for fold := range foldIms {
		for _, p := range params {
			trainInputs = append(trainInputs, TrainInput{fold, p})
		}
	}
	var trainSubset []TrainInput
	for _, p := range trainInputs {
		if _, err := os.Stat(p.TmplFile()); os.IsNotExist(err) {
			trainSubset = append(trainSubset, p)
		} else if err != nil {
			log.Fatalln("check if template cache exists:", err)
		}
	}
	if len(trainSubset) > 0 {
		log.Printf("number of detectors to train: %d / %d", len(trainSubset), len(trainInputs))
		err = dstrfn.MapFunc("train", new([]string), trainSubset, foldIms, *datasetName, *datasetSpec, *pad, exampleOpts, *biasCoeff, *flip, resize.InterpolationFunction(*trainInterp))
		if err != nil {
			log.Fatalln("map(train):", err)
		}
	} else {
		log.Println("all templates have cache file")
	}

	// Test each detector.
	testInputs := make([]TestInput, 0, len(foldIms)*len(params))
	for fold := range foldIms {
		for _, p := range params {
			testInputs = append(testInputs, TestInput{fold, p})
		}
	}
	perfs := make(map[string]float64)
	// Identify which params have not been tested yet.
	var testSubset []TestInput
	for _, x := range testInputs {
		if _, err := os.Stat(x.PerfFile()); os.IsNotExist(err) {
			testSubset = append(testSubset, x)
			continue
		} else if err != nil {
			log.Fatal(err)
		}
		// Attempt to load file.
		var perf float64
		if err := fileutil.LoadExt(x.PerfFile(), &perf); err != nil {
			log.Fatal(err)
		}
		perfs[x.Hash()] = perf
	}

	if len(testSubset) > 0 {
		var out []float64
		err := dstrfn.MapFunc("test", &out, testSubset, foldIms, *datasetName, *datasetSpec, *pad, searchOpts, *minMatch, *minIgnore, fppis)
		if err != nil {
			log.Fatalln("map(test):", err)
		}
		for i, x := range testSubset {
			perfs[x.Hash()] = out[i]
		}
	} else {
		log.Println("all results have cache file")
	}

	// Dump all results to text file.
	// TODO: Implement.
}
