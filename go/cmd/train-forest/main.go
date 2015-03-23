package main

import (
	"flag"
	"fmt"
	"image"
	"log"
	"math"
	"math/rand"
	"os"

	"github.com/gonum/floats"
	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/hog"
	"github.com/jvlmdr/go-cv/imsamp"
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-ml/ml"
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
		// numFolds    = flag.Int("folds", 5, "Cross-validation folds")
		// Positive example configuration.
		width         = flag.Int("width", 32, "Example width before padding")
		height        = flag.Int("height", 96, "Example width before padding")
		pad           = flag.Int("pad", 0, "Dilate bounding box to obtain region from which features are extracted")
		aspectReject  = flag.Float64("reject-aspect", 0, "Reject examples not between r and 1/r times aspect ratio")
		resizeFor     = flag.String("resize-for", "area", "One of {area, width, height, fit, fill}")
		maxTrainScale = flag.Float64("max-train-scale", 2, "Discount examples which would need to be scaled more than this")
		flip          = flag.Bool("flip", false, "Incorporate horizontally mirrored examples?")
		trainInterp   = flag.Int("train-interp", 1, "Interpolation for multi-scale search (0=nearest, 1=linear, 2=cubic)")
		numNeg        = flag.Int("num-neg", 1000, "Number of negative examples")
		// Forest options.
		numTrees = flag.Int("trees", 100, "Number of trees in forest")
		depth    = flag.Int("depth", 4, "Depth of trees")
		numCands = flag.Int("candidates", 100, "Number of features to consider at each split")
		// Training and testing options.
		margin = flag.Int("margin", 0, "Margin to add to image before taking features at test time")
		// Test configuration.
		pyrStep      = flag.Float64("pyr-step", 1.07, "Geometric scale steps in image pyramid")
		maxTestScale = flag.Float64("max-test-scale", 2, "Do not zoom in further than this")
		testInterp   = flag.Int("test-interp", 1, "Interpolation for multi-scale search (0=nearest, 1=linear, 2=cubic)")
		detsPerIm    = flag.Int("dets-per-im", 0, "Maximum number of detections per image")
		localMax     = flag.Bool("local-max", true, "Suppress detections which are less than a neighbor?")
		minMatch     = flag.Float64("min-match", 0.5, "Minimum intersection-over-union to validate a true positive")
		minIgnore    = flag.Float64("min-ignore", 0.5, "Minimum that a region can be covered to be ignored")
	)
	flag.Parse()

	exampleOpts := data.ExampleOpts{
		AspectReject: *aspectReject,
		FitMode:      *resizeFor,
		MaxScale:     *maxTrainScale,
	}

	trainDataset, err := data.Load(*trainDatasetName, *trainDatasetSpec)
	if err != nil {
		log.Fatal(err)
	}
	testDataset, err := data.Load(*testDatasetName, *testDatasetSpec)
	if err != nil {
		log.Fatal(err)
	}

	// Determine dimensions of template.
	size := image.Pt(*width, *height)
	region := detect.PadRect{
		Size: image.Pt(size.X+(*pad)*2, size.Y+(*pad)*2),
		Int:  image.Rectangle{image.ZP, size}.Add(image.Pt((*pad), (*pad))),
	}
	phi := hog.Transform{hog.FGMRConfig(8)}
	//distr := UniformElem{Size: phi.Size(region.Size), Channels: phi.Channels()}
	distr := UniformDiff{Size: phi.Size(region.Size), Channels: phi.Channels(), SameChannel: true}
	fppis := make([]float64, 9)
	floats.LogSpan(fppis, 1e-2, 1)

	// Test configuration.
	searchOpts := detect.MultiScaleOpts{
		MaxScale:  *maxTestScale,
		PyrStep:   *pyrStep,
		Interp:    resize.InterpolationFunction(*testInterp),
		Transform: phi,
		Pad: feat.Pad{
			Margin: feat.UniformMargin(*margin),
			Extend: imsamp.Continue,
		},
		DetFilter: detect.DetFilter{
			LocalMax: *localMax,
			MinScore: math.Inf(-1),
		},
		SupprFilter: detect.SupprFilter{
			MaxNum: *detsPerIm,
			Overlap: func(a, b image.Rectangle) bool {
				return interOverMin(a, b) >= 0.65
			},
		},
	}

	// Split images into positive and negative.
	ims := trainDataset.Images()
	var posIms, negIms []string
	for _, im := range ims {
		if trainDataset.IsNeg(im) {
			negIms = append(negIms, im)
		} else {
			posIms = append(posIms, im)
		}
	}
	//	// Take subset of negative images.
	//	numNegIms := int(u.NegFrac * float64(len(negIms)))
	//	negIms = selectSubset(negIms, randSubset(len(negIms), numNegIms))
	//	log.Println("number of negative images:", len(negIms))

	posRects, err := data.PosExampleRects(posIms, trainDataset, feat.UniformMargin(*margin), region, exampleOpts)
	if err != nil {
		log.Fatal(err)
	}

	// Positive examples are extracted and stored as vectors.
	pos, err := data.Examples(posIms, posRects, trainDataset, phi, imsamp.Continue, region, *flip, resize.InterpolationFunction(*trainInterp))
	if err != nil {
		log.Fatal(err)
	}
	if len(pos) == 0 {
		log.Fatal("empty positive set")
	}

	// Choose an initial set of random negatives.
	// TODO: Check trainDataset.CanTrain()?
	log.Print("choose initial negative examples")
	negRects, err := data.RandomWindows(*numNeg, negIms, trainDataset, feat.UniformMargin(*margin), region.Size)
	if err != nil {
		log.Fatal(err)
	}
	log.Print("sample initial negative examples")
	neg, err := data.Examples(negIms, negRects, trainDataset, phi, imsamp.Continue, region, false, resize.InterpolationFunction(*trainInterp))
	if err != nil {
		log.Fatal(err)
	}
	log.Println("number of negatives:", len(neg))

	var (
		x []*rimg64.Multi
		y []float64
	)
	x = append(x, pos...)
	for _ = range pos {
		y = append(y, 1)
	}
	x = append(x, neg...)
	for _ = range neg {
		y = append(y, -1)
	}

	forest, err := TrainForest(x, y, distr, phi.Size(region.Size), *numTrees, *depth, *numCands)
	if err != nil {
		log.Fatal(err)
	}

	//	// Measure training error.
	//	vals := make([]ml.ValScore, len(x))
	//	for i := range x {
	//		vals = append(vals, ml.ValScore{Score: forest.Eval(x[i]), Pos: y[i] > 0})
	//	}
	//	ml.Sort(vals)
	//	fmt.Printf("avg prec: %.4g\n", ml.Enum(vals).AvgPrec())
	//	shuffle(ScoreList(vals))
	//	fmt.Printf("chance: %.4g\n", ml.Enum(vals).AvgPrec())

	trainIms := trainDataset.Images()
	// Remove images from list which should not be used for testing.
	trainIms = canTestSubset(trainDataset, trainIms)
	trainVals, err := test(trainDataset, trainIms, forest, region, searchOpts, *minMatch, *minIgnore)
	if err != nil {
		log.Fatal(err)
	}

	// Get average miss rate.
	trainRates := detect.MissRateAtFPPIs(detect.MergeValSets(trainVals...), fppis)
	for i := range trainRates {
		log.Printf("fppi %g, miss rate %g", fppis[i], trainRates[i])
	}
	trainPerf := floats.Sum(trainRates) / float64(len(fppis))
	fmt.Println("train:", trainPerf)

	testIms := testDataset.Images()
	// Remove images from list which should not be used for testing.
	testIms = canTestSubset(testDataset, testIms)
	testVals, err := test(testDataset, testIms, forest, region, searchOpts, *minMatch, *minIgnore)
	if err != nil {
		log.Fatal(err)
	}

	// Get average miss rate.
	testRates := detect.MissRateAtFPPIs(detect.MergeValSets(testVals...), fppis)
	for i := range testRates {
		log.Printf("fppi %g, miss rate %g", fppis[i], testRates[i])
	}
	testPerf := floats.Sum(testRates) / float64(len(fppis))
	fmt.Println("test:", testPerf)
}

type ScoreList []ml.ValScore

func (s ScoreList) Len() int      { return len(s) }
func (s ScoreList) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

type List interface {
	Len() int
	Swap(i, j int)
}

func shuffle(xs List) {
	for i, j := range rand.Perm(xs.Len()) {
		xs.Swap(i, j)
	}
}

func interOverMin(a, b image.Rectangle) float64 {
	inter := area(a.Intersect(b))
	min := min(area(a), area(b))
	if min == 0 {
		panic("at least one empty rectangle")
	}
	return float64(inter) / float64(min)
}
