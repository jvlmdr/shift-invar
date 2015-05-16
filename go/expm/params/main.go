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

type DatasetMessage struct {
	Name, Spec string
}

func main() {
	var trainDatasetMessage, testDatasetMessage DatasetMessage
	flag.StringVar(&trainDatasetMessage.Name, "train-dataset", "", "{inria, caltech}")
	flag.StringVar(&trainDatasetMessage.Spec, "train-dataset-spec", "", "Dataset parameters (JSON)")
	flag.StringVar(&testDatasetMessage.Name, "test-dataset", "", "{inria, caltech}")
	flag.StringVar(&testDatasetMessage.Spec, "test-dataset-spec", "", "Dataset parameters (JSON)")

	var (
		numFolds = flag.Int("folds", 5, "Cross-validation folds")
		covarDir = flag.String("covar-dir", "", "Directory to which StatsFile is relative")
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

	// Load training data and determine cross-validation splits.
	// Use same partitions for all methods.
	trainDataset, err := data.Load(trainDatasetMessage.Name, trainDatasetMessage.Spec)
	if err != nil {
		log.Fatal(err)
	}
	trainIms := trainDataset.Images()
	if len(trainIms) == 0 {
		log.Fatal("training dataset is empty")
	}
	// Split images into folds.
	// Cache splits due to their randomness.
	var trainSplits [][]string
	err = fileutil.Cache(&trainSplits, "folds.json", func() [][]string {
		return split(trainIms, *numFolds)
	})
	if err != nil {
		log.Fatal(err)
	}

	// Load testing data.
	testDataset, err := data.Load(testDatasetMessage.Name, testDatasetMessage.Spec)
	if err != nil {
		log.Fatal(err)
	}
	testIms := testDataset.Images()
	if len(testIms) == 0 {
		log.Fatal("testing dataset is empty")
	}
	// Split testing data into folds too.
	var testSplits [][]string
	err = fileutil.Cache(&testSplits, "test-folds.json", func() [][]string {
		return split(testIms, *numFolds)
	})
	if err != nil {
		log.Fatal(err)
	}

	datasets := map[string]DatasetMessage{
		"train": DatasetMessage{Name: trainDatasetMessage.Name, Spec: trainDatasetMessage.Spec},
		"test":  DatasetMessage{Name: testDatasetMessage.Name, Spec: testDatasetMessage.Spec},
	}

	sets := map[string]SubsetImages{
		"train": make(map[string][]string),
		"test":  make(map[string][]string),
	}
	sets["train"]["all"] = trainIms
	for i := range trainSplits {
		sets["train"][fmt.Sprintf("fold-%d", i)] = trainSplits[i]
		// Complement of each fold.
		sets["train"][fmt.Sprintf("excl-fold-%d", i)] = mergeExcept(trainSplits, i)
	}
	sets["test"]["all"] = testIms
	for i := range testSplits {
		sets["test"][fmt.Sprintf("fold-%d", i)] = testSplits[i]
	}

	crossVal := Experiment{TrainDataset: "train", TestDataset: "train", SubsetPairs: make([]SubsetPair, *numFolds)}
	for i := range crossVal.SubsetPairs {
		crossVal.SubsetPairs[i] = SubsetPair{
			Train: fmt.Sprintf("excl-fold-%d", i),
			Test:  fmt.Sprintf("fold-%d", i),
		}
	}
	full := Experiment{TrainDataset: "train", TestDataset: "test", SubsetPairs: []SubsetPair{{Train: "all", Test: "all"}}}
	testVar := Experiment{TrainDataset: "train", TestDataset: "test", SubsetPairs: make([]SubsetPair, *numFolds)}
	for i := range testVar.SubsetPairs {
		testVar.SubsetPairs[i] = SubsetPair{
			Train: fmt.Sprintf("excl-fold-%d", i),
			Test:  fmt.Sprintf("fold-%d", i),
		}
	}
	expms := map[string]Experiment{
		"cross-val": crossVal,
		"full":      full,
		"test-var":  testVar,
	}
	expmNames := []string{"cross-val", "full", "test-var"}
	expmPerfs := make(map[string]map[string]float64)

	trainMapFunc := func(trainInputs []TrainInput, trainDataset DatasetMessage) error {
		return trainMap(trainInputs, trainDataset, *covarDir, *pad, exampleOpts, *flip, resize.InterpolationFunction(*trainInterp), searchOpts)
	}

	testMapFunc := func(testInputs []TestInput, testDataset DatasetMessage) (map[string]float64, error) {
		return testMap(testInputs, testDataset, *pad, searchOpts, *minMatch, *minIgnore, fppis)
	}

	for _, expmName := range expmNames {
		expm := expms[expmName]
		perfs, err := runExperiment(expm, sets, datasets, params, trainMapFunc, testMapFunc)
		if err != nil {
			log.Fatal(err)
		}
		expmPerfs[expmName] = perfs
	}

	if err := printResults(paramset, params, expmNames, expms, expmPerfs); err != nil {
		log.Fatal(err)
	}
}

type SubsetImages map[string][]string

type TrainMapFunc func([]TrainInput, DatasetMessage) error

type TestMapFunc func(testInputs []TestInput, testDataset DatasetMessage) (map[string]float64, error)

func runExperiment(expm Experiment, sets map[string]SubsetImages, datasets map[string]DatasetMessage, params []Param, trainMapFunc TrainMapFunc, testMapFunc TestMapFunc) (map[string]float64, error) {
	var err error
	trainDataset := datasets[expm.TrainDataset]
	trainInputs := make([]TrainInput, 0, len(expm.SubsetPairs)*len(params))
	for _, subsets := range expm.SubsetPairs {
		for _, p := range params {
			input := TrainInput{
				DetectorKey: DetectorKey{
					Param:    p,
					TrainSet: Set{Dataset: expm.TrainDataset, Subset: subsets.Train},
				},
				Images: sets[expm.TrainDataset][subsets.Train],
			}
			trainInputs = append(trainInputs, input)
		}
	}
	err = trainMapFunc(trainInputs, trainDataset)
	if err != nil {
		return nil, err
	}

	testDataset := datasets[expm.TestDataset]
	testInputs := make([]TestInput, 0, len(expm.SubsetPairs)*len(params))
	for _, subsets := range expm.SubsetPairs {
		for _, p := range params {
			input := TestInput{
				ResultsKey: ResultsKey{
					DetectorKey: DetectorKey{
						Param:    p,
						TrainSet: Set{Dataset: expm.TrainDataset, Subset: subsets.Train},
					},
					TestSet: Set{Dataset: expm.TestDataset, Subset: subsets.Test},
				},
				Images: sets[expm.TestDataset][subsets.Test],
			}
			testInputs = append(testInputs, input)
		}
	}
	return testMapFunc(testInputs, testDataset)
}

func trainMap(inputs []TrainInput, dataset DatasetMessage, covarDir string, pad int, exampleOpts data.ExampleOpts, flip bool, interp resize.InterpolationFunction, searchOpts MultiScaleOptsMessage) error {
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
	err := dstrfn.MapFunc("train", new([]string), subset, dataset, covarDir, pad, exampleOpts, flip, interp, searchOpts)
	if err != nil {
		log.Fatalln("map(train):", err)
	}
	return nil
}

func testMap(inputs []TestInput, dataset DatasetMessage, pad int, searchOpts MultiScaleOptsMessage, minMatchIOU, minIgnoreCover float64, fppis []float64) (map[string]float64, error) {
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
	err := dstrfn.MapFunc("test", &out, subset, dataset, pad, searchOpts, minMatchIOU, minIgnoreCover, fppis)
	if err != nil {
		log.Fatalln("map(test):", err)
	}
	for i, x := range subset {
		perfs[x.Key()] = out[i]
	}
	return perfs, nil
}

func printResults(paramset *ParamSet, params []Param, expmNames []string, expms map[string]Experiment, expmPerfs map[string]map[string]float64) error {
	// Dump all results to text file.
	fields := paramset.Fields()
	out, err := os.Create("perfs.txt")
	if err != nil {
		return err
	}
	defer out.Close()
	buf := bufio.NewWriter(out)
	defer buf.Flush()

	fmt.Fprint(buf, "Key")
	for _, name := range fields {
		fmt.Fprintf(buf, "\t%s", name)
	}
	for _, expmName := range expmNames {
		expm := expms[expmName]
		for _, subsets := range expm.SubsetPairs {
			trainSetName := Set{Dataset: expm.TrainDataset, Subset: subsets.Train}
			testSetName := Set{Dataset: expm.TestDataset, Subset: subsets.Test}
			fmt.Fprintf(buf, "\t%s-%s", trainSetName.Key(), testSetName.Key())
		}
		if len(expm.SubsetPairs) > 1 {
			fmt.Fprint(buf, "\tMean")
			fmt.Fprint(buf, "\tVar")
		}
	}
	fmt.Fprintln(buf)

	for _, p := range params {
		fmt.Fprint(buf, p.Key())
		for _, name := range fields {
			fmt.Fprintf(buf, "\t%s", p.Field(name))
		}
		for _, expmName := range expmNames {
			expm := expms[expmName]
			var mean, stddev float64
			for _, subsets := range expm.SubsetPairs {
				resultParam := ResultsKey{
					DetectorKey: DetectorKey{
						Param:    p,
						TrainSet: Set{Dataset: expm.TrainDataset, Subset: subsets.Train},
					},
					TestSet: Set{Dataset: expm.TestDataset, Subset: subsets.Test},
				}
				perf := expmPerfs[expmName][resultParam.Key()]
				mean += perf
				stddev += perf * perf
				fmt.Fprintf(buf, "\t%g", perf)
			}
			n := len(expm.SubsetPairs)
			if n > 1 {
				mean = mean / float64(n)
				stddev = math.Sqrt(stddev/float64(n) - mean*mean)
				fmt.Fprintf(buf, "\t%g", mean)
				fmt.Fprintf(buf, "\t%g", stddev)
			}
		}
		fmt.Fprintln(buf)
	}
	return nil
}
