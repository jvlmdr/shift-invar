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
	"os"
	"path"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/featset"
	"github.com/jvlmdr/go-cv/imgpyr"
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
		width        = flag.Int("width", 0, "Pixel width of examples (before padding)")
		height       = flag.Int("height", 0, "Pixel height of examples (before padding)")
		pad          = flag.Int("pad", 0, "Dilate bounding box to obtain region from which features are extracted")
		aspectReject = flag.Float64("reject-aspect", 0, "Reject examples not between r and 1/r times aspect ratio")
		resizeFor    = flag.String("resize-for", "area", "One of {area, width, height, fit, fill}")
		maxScale     = flag.Float64("max-scale", 2, "Discount examples which would need to be scaled more than this")
		flip         = flag.Bool("flip", false, "Incorporate horizontally mirrored examples?")

		//	// Train configuration.
		//	lambda    = flag.Float64("lambda", 1e-4, "Regularization coefficient")
		//	circulant = flag.Bool("circulant", false, "Use circulant matrix? (or Toeplitz)")
		//	uniform   = flag.Bool("uniform", false, "Use uniform normalization? (ensure semidefinite)")
		//	tol       = flag.Float64("tol", 1e-6, "Tolerance for convergence of training")
		//	maxIter   = flag.Int("max-iter", 0, "Maximum number of iterations for training")

		//	// Test configuration.
		//	pyrStep  = flag.Float64("pyr-step", 1.07, "Geometric scale steps in image pyramid")
		//	maxIOU   = flag.Float64("max-iou", 0.3, "Intersection-over-union threshold for non-max suppression")
		//	margin   = flag.Int("margin", 0, "Margin to add to image before taking features")
		//	localMax = flag.Bool("local-max", true, "Suppress detections which are less than a neighbor?")
		//	minInter = flag.Float64("min-inter", 0.5, "Minimum intersection-over-union to validate a true positive")
	)
	flag.Parse()

	//	if flag.NArg() != 2 {
	//		flag.Usage()
	//		os.Exit(1)
	//	}

	interior := image.Rect(0, 0, *width, *height).Add(image.Pt(*pad, *pad))
	size := image.Pt(*pad*2+*width, *pad*2+*height)
	region := detect.PadRect{size, interior}

	var phi featset.Image = new(featset.ImageMarshaler)
	if err := json.Unmarshal([]byte(*featJSON), phi); err != nil {
		log.Fatalln("load transform:", err)
	}

	featImSize := phi.Size(region.Size)

	// Load raw annotations of dataset.
	data, err := loadDataset(*datasetName, *datasetSpec)
	if err != nil {
		log.Fatal(err)
	}
	// Split images into folds.
	folds := split(data.Images(), *numFolds)

	// Train a detector for each fold.
	//	perfs := make([]float64, numFolds)
	for i := 0; i < *numFolds; i++ {
		// Take union of sets.
		var trainIms []string
		for j := 0; j < *numFolds; j++ {
			if j == i {
				continue
			}
			trainIms = append(trainIms, folds[j]...)
		}
		// Extract positive examples and negative images.
		trainData, err := extractTrainingData(data, trainIms, region, *aspectReject, *resizeFor, *maxScale)
		if err != nil {
			log.Fatal(err)
		}
		if len(trainData.PosImages) == 0 {
			log.Fatal("set of positive examples is empty")
		}
		if len(trainData.NegImages) == 0 {
			log.Fatal("set of negative images is empty")
		}

		tmpl, err := trainSVM(trainData, data, phi, 1, region, featImSize, phi.Channels(), *flip, 0.5, 0.5, 1)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("fold", i)
		fmt.Println(tmpl)

		//	perf, err := test(folds[i], data, localMax, minInter)
		//	if err != nil {
		//		log.Fatal(err)
		//	}
		//	perfs[i] = perf
	}
	//	fmt.Println(perfs)

	//	detOpts := DetectOpts{
	//		FeatName: *featName,
	//		FeatSpec: *featSpec,
	//		PyrStep:  *pyrStep,
	//		MaxIOU:   *maxIOU,
	//		Margin:   *margin,
	//		LocalMax: *localMax,
	//	}

	//	{
	//		// Test template on training set.
	//		log.Print("test detector: training set")
	//		valDets, err := test(inriaDir, "Train", tmpl, detOpts, *minInter)
	//		if err != nil {
	//			log.Fatal(err)
	//		}
	//		if err := fileutil.SaveExt("val-dets-train.json", valDets); err != nil {
	//			log.Print(err)
	//		}
	//		results := ml.ResultSet(mergeDets(valDets).Enum())
	//		fmt.Println("training set: avgprec:", results.AveragePrecision())
	//	}

	//	{
	//		// Test template.
	//		log.Print("test detector: testing set")
	//		valDets, err := test(inriaDir, "Test", tmpl, detOpts, *minInter)
	//		if err != nil {
	//			log.Fatal(err)
	//		}
	//		if err := fileutil.SaveExt("val-dets-test.json", valDets); err != nil {
	//			log.Print(err)
	//		}
	//		results := ml.ResultSet(mergeDets(valDets).Enum())
	//		fmt.Println("testing set: avgprec:", results.AveragePrecision())
	//	}
}
