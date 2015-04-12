package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/gonum/floats"
	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-file/fileutil"
	"github.com/jvlmdr/go-pbs-pro/dstrfn"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/nfnt/resize"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage:", os.Args[0], "[flags] params.json")
		flag.PrintDefaults()
	}
}

func main() {
	var (
		trainDatasetName = flag.String("train-dataset", "", "{inria, caltech}")
		trainDatasetSpec = flag.String("train-dataset-spec", "", "Dataset parameters (JSON)")
		testDatasetName  = flag.String("test-dataset", "", "{inria, caltech}")
		testDatasetSpec  = flag.String("test-dataset-spec", "", "Dataset parameters (JSON)")
		numFolds         = flag.Int("folds", 5, "Cross-validation folds")
		covarDir         = flag.String("covar-dir", "", "Directory to which CovarFile is relative")
		// Positive example configuration.
		pad           = flag.Int("pad", 0, "Dilate bounding box to obtain region from which features are extracted")
		aspectReject  = flag.Float64("reject-aspect", 0, "Reject examples not between r and 1/r times aspect ratio")
		resizeFor     = flag.String("resize-for", "area", "One of {area, width, height, fit, fill}")
		maxTrainScale = flag.Float64("max-train-scale", 2, "Discount examples which would need to be scaled more than this")
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

	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(1)
	}
	paramsFile := flag.Arg(0)

	paramset := new(ParamSet)
	if err := fileutil.LoadExt(paramsFile, paramset); err != nil {
		log.Fatal(err)
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
	searchOpts := MultiScaleOptsMessage{
		MaxScale:  *maxTestScale,
		PyrStep:   *pyrStep,
		Interp:    resize.InterpolationFunction(*testInterp),
		PadMargin: feat.UniformMargin(*testMargin),
		DetFilter: detect.DetFilter{
			LocalMax: *localMax,
			MinScore: 0,
		},
		SupprMaxNum: *detsPerIm,
	}

	params := paramset.Enumerate()
	for _, p := range params {
		fmt.Printf("%s\t%s\n", p.Key(), p.ID())
	}

	// Load data and determine cross-validation splits.
	// Use same partitions for all methods.
	trainData, err := data.Load(*trainDatasetName, *trainDatasetSpec)
	if err != nil {
		log.Fatal(err)
	}
	trainIms := trainData.Images()
	if len(trainIms) == 0 {
		log.Fatal("training dataset is empty")
	}
	// Split images into folds.
	// Cache splits due to their randomness.
	var foldIms [][]string
	err = fileutil.Cache(&foldIms, "folds.json", func() [][]string {
		return split(trainIms, *numFolds)
	})
	if err != nil {
		log.Fatal(err)
	}
	// Complement of each fold is its training data.
	foldTrainIms := make([][]string, len(foldIms))
	for i := range foldIms {
		foldTrainIms[i] = mergeExcept(foldIms, i)
	}
	// Load testing data.
	testData, err := data.Load(*testDatasetName, *testDatasetSpec)
	if err != nil {
		log.Fatal(err)
	}
	testIms := testData.Images()
	if len(testIms) == 0 {
		log.Fatal("testing dataset is empty")
	}

	// Learn template for each configuration for each fold.
	// Save weights to file and avoid re-computing weights
	// for configurations which have a file.
	xvalTrainInputs := make([]TrainInput, 0, len(foldIms)*len(params))
	for fold := range foldIms {
		for _, p := range params {
			xvalTrainInputs = append(xvalTrainInputs, TrainInput{CrossValKey(p, fold), foldTrainIms[fold]})
		}
	}
	err = trainMap(xvalTrainInputs, *trainDatasetName, *trainDatasetSpec, *covarDir, *pad, exampleOpts, *flip, resize.InterpolationFunction(*trainInterp), searchOpts)
	if err != nil {
		log.Fatal(err)
	}

	// Test each detector on its validation fold.
	xvalTestInputs := make([]TestInput, 0, len(foldIms)*len(params))
	for fold := range foldIms {
		for _, p := range params {
			xvalTestInputs = append(xvalTestInputs, TestInput{CrossValKey(p, fold), foldIms[fold]})
		}
	}
	xvalPerfs, err := testMap(xvalTestInputs, *trainDatasetName, *trainDatasetSpec, *pad, searchOpts, *minMatch, *minIgnore, fppis)
	if err != nil {
		log.Fatal(err)
	}

	trainInputs := make([]TrainInput, 0, len(params))
	for _, p := range params {
		trainInputs = append(trainInputs, TrainInput{TestKey(p), trainIms})
	}
	err = trainMap(trainInputs, *trainDatasetName, *trainDatasetSpec, *covarDir, *pad, exampleOpts, *flip, resize.InterpolationFunction(*trainInterp), searchOpts)
	if err != nil {
		log.Fatal(err)
	}

	// Test each detector on the testing set.
	testInputs := make([]TestInput, 0, len(params))
	for _, p := range params {
		testInputs = append(testInputs, TestInput{TestKey(p), testIms})
	}
	testPerfs, err := testMap(testInputs, *testDatasetName, *testDatasetSpec, *pad, searchOpts, *minMatch, *minIgnore, fppis)
	if err != nil {
		log.Fatal(err)
	}

	// Dump all results to text file.
	fields := paramset.Fields()
	out, err := os.Create("perfs.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()
	buf := bufio.NewWriter(out)
	defer buf.Flush()
	fmt.Fprint(buf, "Key")
	for _, name := range fields {
		fmt.Fprintf(buf, "\t%s", name)
	}
	for fold := range foldIms {
		fmt.Fprintf(buf, "\t%d", fold)
	}
	fmt.Fprint(buf, "\tCrossValMissRateMean\tCrossValMissRateVar")
	fmt.Fprintln(buf, "\tMissRate")
	for _, p := range params {
		fmt.Fprint(buf, p.Key())
		for _, name := range fields {
			fmt.Fprintf(buf, "\t%s", p.Field(name))
		}
		var mean, stddev float64
		for fold := 0; fold < *numFolds; fold++ {
			perf, ok := xvalPerfs[CrossValKey(p, fold).Key()]
			if !ok {
				log.Fatalln("did not find cross-val perf:", CrossValKey(p, fold), p.ID())
			}
			mean += perf
			stddev += perf * perf
			fmt.Fprintf(buf, "\t%g", perf)
		}
		mean = mean / float64(*numFolds)
		stddev = math.Sqrt(stddev/float64(*numFolds) - mean*mean)
		fmt.Fprintf(buf, "\t%g", mean)
		fmt.Fprintf(buf, "\t%g", stddev)
		testPerf, ok := testPerfs[TestKey(p).Key()]
		if !ok {
			log.Fatalln("did not find test perf:", TestKey(p), p.ID())
		}
		fmt.Fprintf(buf, "\t%g", testPerf)
		fmt.Fprintln(buf)
	}
}

func trainMap(inputs []TrainInput, datasetName, datasetSpec, covarDir string, pad int, exampleOpts data.ExampleOpts, flip bool, interp resize.InterpolationFunction, searchOpts MultiScaleOptsMessage) error {
	var subset []TrainInput
	for _, p := range inputs {
		if _, err := os.Stat(p.TmplFile()); os.IsNotExist(err) {
			subset = append(subset, p)
		} else if err != nil {
			log.Fatalln("check if template cache exists:", err)
		}
	}
	if len(subset) == 0 {
		log.Println("all templates have cache file")
		return nil
	}
	log.Printf("number of detectors to train: %d / %d", len(subset), len(inputs))
	// Discard output since result is saved to file.
	err := dstrfn.MapFunc("train", new([]string), subset, datasetName, datasetSpec, covarDir, pad, exampleOpts, flip, interp, searchOpts)
	if err != nil {
		log.Fatalln("map(train):", err)
	}
	return nil
}

func testMap(inputs []TestInput, datasetName, datasetSpec string, pad int, searchOpts MultiScaleOptsMessage, minMatchIOU, minIgnoreCover float64, fppis []float64) (map[string]float64, error) {
	perfs := make(map[string]float64)
	// Identify which params have not been tested yet.
	var subset []TestInput
	for _, x := range inputs {
		fname := x.PerfFile()
		if _, err := os.Stat(fname); os.IsNotExist(err) {
			subset = append(subset, x)
			continue
		} else if err != nil {
			log.Fatalf(`stat cache file "%s": %v`, fname, err)
		}
		// Attempt to load file.
		var perf float64
		if err := fileutil.LoadExt(fname, &perf); err != nil {
			log.Fatalf(`load cache file "%s": %v`, fname, err)
		}
		perfs[x.Key()] = perf
	}

	if len(subset) == 0 {
		log.Println("all results have cache file")
		return perfs, nil
	}
	var out []float64
	err := dstrfn.MapFunc("test", &out, subset, datasetName, datasetSpec, pad, searchOpts, minMatchIOU, minIgnoreCover, fppis)
	if err != nil {
		log.Fatalln("map(test):", err)
	}
	for i, x := range subset {
		perfs[x.Key()] = out[i]
	}
	return perfs, nil
}
