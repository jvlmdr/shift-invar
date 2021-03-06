package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/gonum/floats"
	"github.com/jvlmdr/go-cv/detect"
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
		flip        = flag.Bool("flip", false, "Incorporate horizontally mirrored examples?")
		trainInterp = flag.Int("train-interp", 1, "Interpolation for multi-scale search (0=nearest, 1=linear, 2=cubic)")
		// Test configuration.
		testInterp = flag.Int("test-interp", 1, "Interpolation for multi-scale search (0=nearest, 1=linear, 2=cubic)")
		detsPerIm  = flag.Int("dets-per-im", 0, "Maximum number of detections per image")
		localMax   = flag.Bool("local-max", true, "Suppress detections which are less than a neighbor?")
		minMatch   = flag.Float64("min-match", 0.5, "Minimum intersection-over-union to validate a true positive")
		minIgnore  = flag.Float64("min-ignore", 0.5, "Minimum that a region can be covered to be ignored")
		doTest     = flag.Bool("do-test", false, "Run experiments to measure performance on test set?")
		doTestVar  = flag.Bool("do-test-var", false, "Run experiments to measure variance of performance on test set?")
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

	// Test configuration.
	// Transform and Overlap are taken from Param.
	searchOpts := MultiScaleOptsMessage{
		Interp: resize.InterpolationFunction(*testInterp),
		DetFilter: detect.DetFilter{
			LocalMax: *localMax,
			MinScore: 0, // Ignored; later set to -inf.
		},
		SupprMaxNum: *detsPerIm,
	}

	params := paramset.Enumerate()
	for _, p := range params {
		fmt.Printf("%s\t%s\n", p.Ident(), p.Serialize())
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
	expmNames := []string{"cross-val"}
	if *doTest {
		expmNames = append(expmNames, "full")
	}
	if *doTestVar {
		expmNames = append(expmNames, "test-var")
	}
	expmResults := make(map[string]ExperimentResult)

	trainMapFunc := func(trainInputs []TrainInput, trainDataset DatasetMessage) error {
		return trainMap(trainInputs, trainDataset, *covarDir, *flip, resize.InterpolationFunction(*trainInterp), searchOpts)
	}

	testMapFunc := func(testInputs []TestInput, testDataset DatasetMessage) (ExperimentResult, error) {
		return testMap(testInputs, testDataset, searchOpts, *minMatch, *minIgnore, fppis)
	}

	for _, expmName := range expmNames {
		expm := expms[expmName]
		perfs, err := runExperiment(expm, sets, datasets, params, trainMapFunc, testMapFunc)
		if err != nil {
			log.Fatal(err)
		}
		expmResults[expmName] = perfs
	}

	if err := printResults(paramset, params, expmNames, expms, expmResults); err != nil {
		log.Fatal(err)
	}
}

// ExperimentResult contains the results for one or more partitions.
type ExperimentResult map[string]*TestResult

type SubsetImages map[string][]string

type TrainMapFunc func([]TrainInput, DatasetMessage) error

type TestMapFunc func(testInputs []TestInput, testDataset DatasetMessage) (ExperimentResult, error)

func runExperiment(expm Experiment, sets map[string]SubsetImages, datasets map[string]DatasetMessage, params []Param, trainMapFunc TrainMapFunc, testMapFunc TestMapFunc) (ExperimentResult, error) {
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

func trainMap(inputs []TrainInput, dataset DatasetMessage, covarDir string, flip bool, interp resize.InterpolationFunction, searchOpts MultiScaleOptsMessage) error {
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
	err := dstrfn.MapFunc("train", new([]string), subset, dataset, covarDir, flip, interp, searchOpts)
	if err != nil {
		log.Fatalln("map(train):", err)
	}
	return nil
}

type TestResult struct {
	Perf        float64
	TrainReport *TrainReport
}

// testMap tests all detectors in a list.
func testMap(inputs []TestInput, dataset DatasetMessage, searchOpts MultiScaleOptsMessage, minMatchIOU, minIgnoreCover float64, fppis []float64) (ExperimentResult, error) {
	// Identify configurations for which a training error occurred.
	// Load cached results and identify un-tested subset.
	results := make(map[string]*TestResult)
	var subset []TestInput
	// TODO: Not maintain this extra array?
	// Needed to copy information to TestResult.
	var subsetTrainReports []*TrainReport // For subset
	for _, x := range inputs {
		tmplFile := x.TmplFile()
		var trainResult TrainResult
		if err := fileutil.LoadExt(tmplFile, &trainResult); err != nil {
			return nil, err
		}
		if trainResult.Report.Error != "" {
			// Testing result inherits error from training result.
			results[x.Ident()] = &TestResult{TrainReport: trainResult.Report}
			// Do not add to subset.
			continue
		}

		perfFile := x.PerfFile()
		if _, err := os.Stat(perfFile); os.IsNotExist(err) {
			subset = append(subset, x)
			subsetTrainReports = append(subsetTrainReports, trainResult.Report)
			continue
		} else if err != nil {
			return nil, fmt.Errorf(`stat cache file "%s": %v`, perfFile, err)
		}
		// Attempt to load file.
		var perf float64
		if err := fileutil.LoadExt(perfFile, &perf); err != nil {
			return nil, fmt.Errorf(`load cache file "%s": %v`, perfFile, err)
		}
		results[x.Ident()] = &TestResult{Perf: perf, TrainReport: trainResult.Report}
	}

	if len(subset) == 0 {
		log.Println("all results have cache file")
		return results, nil
	}
	var out []float64
	err := dstrfn.MapFunc("test", &out, subset, dataset, searchOpts, minMatchIOU, minIgnoreCover, fppis)
	if err != nil {
		log.Fatalln("map(test):", err)
	}
	for i, x := range subset {
		results[x.Ident()] = &TestResult{Perf: out[i], TrainReport: subsetTrainReports[i]}
	}
	return results, nil
}

func printResults(paramset *ParamSet, params []Param, expmNames []string, expms map[string]Experiment, expmResults map[string]ExperimentResult) error {
	// Dump all results to text file.
	fields := paramset.Fields()
	out, err := os.Create("perfs.txt")
	if err != nil {
		return err
	}
	defer out.Close()
	buf := bufio.NewWriter(out)
	defer buf.Flush()

	for _ = range fields {
		fmt.Fprint(buf, "\t")
	}
	for _, expmName := range expmNames {
		expm := expms[expmName]
		for _ = range expm.SubsetPairs {
			fmt.Fprintf(buf, "\t\t\t\t\t")
		}
		if len(expm.SubsetPairs) > 1 {
			fmt.Fprintf(buf, "\tTotal\t\t\t\t\t\t\t")
		}
	}
	fmt.Fprintln(buf)

	for _ = range fields {
		fmt.Fprint(buf, "\t")
	}
	for _, expmName := range expmNames {
		expm := expms[expmName]
		for _, subsets := range expm.SubsetPairs {
			trainSetName := Set{Dataset: expm.TrainDataset, Subset: subsets.Train}
			testSetName := Set{Dataset: expm.TestDataset, Subset: subsets.Test}
			fmt.Fprintf(buf, "\t%s-%s\t\t\t\t", trainSetName.Ident(), testSetName.Ident())
		}
		if len(expm.SubsetPairs) > 1 {
			fmt.Fprintf(buf, "\tPerf\t\tTrainDur\t\tSolveDur\t\tSubstDur\t")
		}
	}
	fmt.Fprintln(buf)

	fmt.Fprint(buf, "Key")
	for _, name := range fields {
		fmt.Fprintf(buf, "\t%s", name)
	}
	for _, expmName := range expmNames {
		expm := expms[expmName]
		for _ = range expm.SubsetPairs {
			fmt.Fprintf(buf, "\tPerf\tError\tTrainDur\tSolveDur\tSubstDur")
		}
		if len(expm.SubsetPairs) > 1 {
			fmt.Fprintf(buf, "\tMean\tVar\tMean\tVar\tMean\tVar\tMean\tVar")
		}
	}
	fmt.Fprintln(buf)

	for _, p := range params {
		fmt.Fprint(buf, p.Ident())
		for _, name := range fields {
			fmt.Fprintf(buf, "\t%s", strconv.Quote(p.Field(name)))
		}
		for _, expmName := range expmNames {
			expm := expms[expmName]
			// var mean, stddev float64
			var results []*TestResult
			var fail bool
			for _, subsets := range expm.SubsetPairs {
				resultParam := ResultsKey{
					DetectorKey: DetectorKey{
						Param:    p,
						TrainSet: Set{Dataset: expm.TrainDataset, Subset: subsets.Train},
					},
					TestSet: Set{Dataset: expm.TestDataset, Subset: subsets.Test},
				}
				result := expmResults[expmName][resultParam.Ident()]
				results = append(results, result)
				if result.TrainReport.Error != "" {
					fail = true
					fmt.Fprintf(buf, "\t\t%s\t\t\t", result.TrainReport.Error)
				} else {
					fmt.Fprintf(buf, "\t%.6g\t\t%.6g\t%.6g\t%.6g", result.Perf,
						result.TrainReport.TotalDur.Seconds(),
						result.TrainReport.SolveDur.Total.Seconds(),
						result.TrainReport.SolveDur.Subst.Seconds(),
					)
				}
			}

			n := len(expm.SubsetPairs)
			if n <= 1 {
				// Do not report statistics.
				continue
			}
			if fail {
				// At least one error occurred.
				fmt.Fprint(buf, "\t\t\t\t\t\t\t\t")
				continue
			}
			stats := EstimateExperimentStats(results)
			fmt.Fprintf(buf, "\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g",
				stats.Perf.Mean, stats.Perf.Var,
				stats.TrainDur.Mean, stats.TrainDur.Var,
				stats.SolveDur.Mean, stats.SolveDur.Var,
				stats.SubstDur.Mean, stats.SubstDur.Var,
			)
		}
		fmt.Fprintln(buf)
	}
	return nil
}

type Stats struct {
	Mean, Var float64
}

type ExperimentStats struct {
	Perf, TrainDur, SolveDur, SubstDur Stats
}

func EstimateExperimentStats(results []*TestResult) *ExperimentStats {
	var stats ExperimentStats
	// TODO: Programmatize this! map[string]Stats or StatsAccumer?
	for _, r := range results {
		stats.Perf.Mean += r.Perf
		stats.TrainDur.Mean += r.TrainReport.TotalDur.Seconds()
		stats.SolveDur.Mean += r.TrainReport.SolveDur.Total.Seconds()
		stats.SubstDur.Mean += r.TrainReport.SolveDur.Subst.Seconds()
		stats.Perf.Var += square(r.Perf)
		stats.TrainDur.Var += square(r.TrainReport.TotalDur.Seconds())
		stats.SolveDur.Var += square(r.TrainReport.SolveDur.Total.Seconds())
		stats.SubstDur.Var += square(r.TrainReport.SolveDur.Subst.Seconds())
	}
	stats.Perf.Mean /= float64(len(results))
	stats.TrainDur.Mean /= float64(len(results))
	stats.SolveDur.Mean /= float64(len(results))
	stats.SubstDur.Mean /= float64(len(results))
	stats.Perf.Var /= float64(len(results))
	stats.TrainDur.Var /= float64(len(results))
	stats.SolveDur.Var /= float64(len(results))
	stats.SubstDur.Var /= float64(len(results))
	stats.Perf.Var = math.Sqrt(stats.Perf.Var - square(stats.Perf.Mean))
	stats.TrainDur.Var = math.Sqrt(stats.TrainDur.Var - square(stats.TrainDur.Mean))
	stats.SolveDur.Var = math.Sqrt(stats.SolveDur.Var - square(stats.SolveDur.Mean))
	stats.SubstDur.Var = math.Sqrt(stats.SubstDur.Var - square(stats.SubstDur.Mean))
	return &stats
}

func square(x float64) float64 { return x * x }
