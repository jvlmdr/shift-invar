package main

/*
This command-line tool trains a detector using a given
dataset, feature transform and training algorithm.
Accuracy is estimated using k-fold cross-validation.
*/

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"log"
	"math"
	"os"
	"path"

	"github.com/gonum/floats"
	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/featset"
	"github.com/jvlmdr/go-cv/imgpyr"
	"github.com/jvlmdr/go-cv/imsamp"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/nfnt/resize"
)

var DefaultInterp = resize.Bilinear

func init() {
	imgpyr.DefaultInterp = DefaultInterp

	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage:")
		fmt.Fprintln(os.Stderr, path.Base(os.Args[0]), "[flags]")
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "Trains and tests a detector.")
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "Options:")
		flag.PrintDefaults()
		fmt.Fprintln(os.Stderr)
	}
}

var (
	saveExamples = flag.Bool("save-examples", false, "Save positive examples")
)

func main() {
	var (
		datasetName = flag.String("dataset", "", "{inria, caltech}")
		datasetSpec = flag.String("dataset-spec", "", "Dataset parameters (JSON)")
		numFolds    = flag.Int("folds", 5, "Cross-validation folds")

		//	algoName  = flag.String("algo", "", "{svm, struct-svm, hnm-svm, toep, circ}")
		//	algoSpec  = flag.String("algo-spec", "", "Algorithm options (JSON)")

		// Feature configuration.
		featJSON = flag.String("feat", "", "")
		// Positive example configuration.
		width         = flag.Int("width", 0, "Pixel width of examples (before padding)")
		height        = flag.Int("height", 0, "Pixel height of examples (before padding)")
		pad           = flag.Int("pad", 0, "Dilate bounding box to obtain region from which features are extracted")
		aspectReject  = flag.Float64("reject-aspect", 0, "Reject examples not between r and 1/r times aspect ratio")
		resizeFor     = flag.String("resize-for", "area", "One of {area, width, height, fit, fill}")
		maxTrainScale = flag.Float64("max-train-scale", 2, "Discount examples which would need to be scaled more than this")
		flip          = flag.Bool("flip", false, "Incorporate horizontally mirrored examples?")
		trainInterp   = flag.Int("train-interp", 1, "Interpolation for multi-scale search (0=nearest, 1=linear, 2=cubic)")

		//	// Train configuration.
		//	lambda    = flag.Float64("lambda", 1e-4, "Regularization coefficient")
		//	circulant = flag.Bool("circulant", false, "Use circulant matrix? (or Toeplitz)")
		//	uniform   = flag.Bool("uniform", false, "Use uniform normalization? (ensure semidefinite)")
		//	tol       = flag.Float64("tol", 1e-6, "Tolerance for convergence of training")
		//	maxIter   = flag.Int("max-iter", 0, "Maximum number of iterations for training")

		// Test configuration.
		pyrStep      = flag.Float64("pyr-step", 1.07, "Geometric scale steps in image pyramid")
		maxTestScale = flag.Float64("max-test-scale", 2, "Do not zoom in further than this")
		testInterp   = flag.Int("test-interp", 1, "Interpolation for multi-scale search (0=nearest, 1=linear, 2=cubic)")
		detsPerIm    = flag.Int("dets-per-im", 0, "Maximum number of detections per image")
		maxTestIOU   = flag.Float64("max-iou", 0.3, "Intersection-over-union threshold for non-max suppression")
		testMargin   = flag.Int("margin", 0, "Margin to add to image before taking features at test time")
		localMax     = flag.Bool("local-max", true, "Suppress detections which are less than a neighbor?")
		minMatch     = flag.Float64("min-match", 0.5, "Minimum intersection-over-union to validate a true positive")
		minIgnore    = flag.Float64("min-ignore", 0.5, "Minimum that a region can be covered to be ignored")
	)
	flag.Parse()

	//	if flag.NArg() != 2 {
	//		flag.Usage()
	//		os.Exit(1)
	//	}

	// Geometry of template.
	interior := image.Rect(0, 0, *width, *height).Add(image.Pt(*pad, *pad))
	size := image.Pt(*pad*2+*width, *pad*2+*height)
	region := detect.PadRect{size, interior}
	// Options for choosing rectangles.
	exampleOpts := data.ExampleOpts{
		AspectReject: *aspectReject,
		FitMode:      *resizeFor,
		MaxScale:     *maxTrainScale,
	}
	// FPPIs at which to compute miss rate.
	fppis := make([]float64, 9)
	floats.LogSpan(fppis, 1e-2, 1)

	var phi featset.Image = new(featset.ImageMarshaler)
	if err := json.Unmarshal([]byte(*featJSON), phi); err != nil {
		log.Fatalln("load transform:", err)
	}

	// Test configuration.
	searchOpts := detect.MultiScaleOpts{
		MaxScale:  *maxTestScale,
		PyrStep:   *pyrStep,
		Interp:    resize.InterpolationFunction(*testInterp),
		Transform: phi,
		Pad:       feat.Pad{feat.UniformMargin(*testMargin), imsamp.Continue},
		DetFilter: detect.DetFilter{
			LocalMax: *localMax,
			MinScore: math.Inf(-1),
		},
		SupprFilter: detect.SupprFilter{
			MaxNum:  *detsPerIm,
			Overlap: func(a, b image.Rectangle) bool { return detect.IOU(a, b) > *maxTestIOU },
		},
	}

	// Load raw annotations of dataset.
	dataset, err := data.Load(*datasetName, *datasetSpec)
	if err != nil {
		log.Fatal(err)
	}
	// Split images into folds.
	folds := split(dataset.Images(), *numFolds)

	// Train a detector for each fold.
	perfs := make([]float64, *numFolds)
	for i := range folds {
		trainIms := mergeExcept(folds, i)
		testIms := folds[i]
		// Extract positive examples and negative images.
		trainData, err := data.ExtractTrainingSet(dataset, trainIms, region, exampleOpts)
		if err != nil {
			log.Fatal(err)
		}
		if len(trainData.PosImages) == 0 {
			log.Fatal("set of positive examples is empty")
		}
		if len(trainData.NegImages) == 0 {
			log.Fatal("set of negative images is empty")
		}
		// Obtain detector.
		weights, bias, err := trainSVM(trainData, dataset, phi, 1, region, *flip, 0.5, 0.5, 1, resize.InterpolationFunction(*trainInterp))
		if err != nil {
			log.Fatal(err)
		}
		tmpl := &detect.FeatTmpl{weights, bias, region.Size, region.Int}
		fmt.Println("fold", i)

		trainPerf, err := test(tmpl, trainIms, dataset, phi, searchOpts, *minMatch, *minIgnore, fppis)
		if err != nil {
			log.Fatal(err)
		}
		testPerf, err := test(tmpl, testIms, dataset, phi, searchOpts, *minMatch, *minIgnore, fppis)
		if err != nil {
			log.Fatal(err)
		}
		perfs[i] = testPerf
		fmt.Printf("fold %d, train miss rate %.3g, test miss rate %.3g\n", i, trainPerf, testPerf)
	}
	fmt.Println(perfs)
}
