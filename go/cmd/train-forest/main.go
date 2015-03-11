package main

import (
	"flag"
	"fmt"
	"image"
	"log"
	"math/rand"
	"os"
	"sort"

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
		datasetName = flag.String("dataset", "", "{inria, caltech}")
		datasetSpec = flag.String("dataset-spec", "", "Dataset parameters (JSON)")
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
		numCands = flag.Int("candidates", 100, "Number of features to consider at each split")
		// Training and testing options.
		margin = flag.Int("margin", 0, "Margin to add to image before taking features at test time")
	)
	flag.Parse()

	exampleOpts := data.ExampleOpts{
		AspectReject: *aspectReject,
		FitMode:      *resizeFor,
		MaxScale:     *maxTrainScale,
	}

	dataset, err := data.Load(*datasetName, *datasetSpec)
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
	distr := UniformElem{Size: phi.Size(region.Size), Channels: phi.Channels()}

	// Split images into positive and negative.
	ims := dataset.Images()
	var posIms, negIms []string
	for _, im := range ims {
		if dataset.IsNeg(im) {
			negIms = append(negIms, im)
		} else {
			posIms = append(posIms, im)
		}
	}
	//	// Take subset of negative images.
	//	numNegIms := int(u.NegFrac * float64(len(negIms)))
	//	negIms = selectSubset(negIms, randSubset(len(negIms), numNegIms))
	//	log.Println("number of negative images:", len(negIms))

	posRects, err := data.PosExampleRects(posIms, dataset, feat.UniformMargin(*margin), region, exampleOpts)
	if err != nil {
		log.Fatal(err)
	}

	// Positive examples are extracted and stored as vectors.
	pos, err := data.Examples(posIms, posRects, dataset, phi, imsamp.Continue, region, *flip, resize.InterpolationFunction(*trainInterp))
	if err != nil {
		log.Fatal(err)
	}
	if len(pos) == 0 {
		log.Fatal("empty positive set")
	}

	// Choose an initial set of random negatives.
	// TODO: Check dataset.CanTrain()?
	log.Print("choose initial negative examples")
	negRects, err := data.RandomWindows(*numNeg, negIms, dataset, feat.UniformMargin(*margin), region.Size)
	if err != nil {
		log.Fatal(err)
	}
	log.Print("sample initial negative examples")
	neg, err := data.Examples(negIms, negRects, dataset, phi, imsamp.Continue, region, false, resize.InterpolationFunction(*trainInterp))
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

	forest, err := TrainStumpForest(x, y, distr, *numTrees, *numCands)
	if err != nil {
		log.Fatal(err)
	}

	// Measure training error.
	vals := make([]detect.ValScore, len(x))
	for i := range x {
		vals = append(vals, detect.ValScore{Score: forest.Eval(x[i]), True: y[i] > 0})
	}
	sort.Sort(byScoreDesc(vals))
	fmt.Printf("avg prec: %.4g\n", enumerate(vals).AvgPrec())
	shuffle(byScoreDesc(vals))
	fmt.Printf("chance: %.4g\n", enumerate(vals).AvgPrec())
}

func enumerate(vals []detect.ValScore) ml.PerfPath {
	var pos, neg int
	for _, val := range vals {
		if val.True {
			pos++
		} else {
			neg++
		}
	}
	// Start with high threshold, everything negative,
	// then gradually lower it.
	perfs := make([]ml.Perf, 0, len(vals)+1)
	perf := ml.Perf{FN: pos, TN: neg}
	perfs = append(perfs, perf)
	for i := range vals {
		if vals[i].True {
			// Positive example classified as positive instead of negative.
			perf.TP, perf.FN = perf.TP+1, perf.FN-1
		} else {
			// Negative example classified as positive instead of negative.
			perf.FP, perf.TN = perf.FP+1, perf.TN-1
		}
		perfs = append(perfs, perf)
	}
	return ml.PerfPath(perfs)
}

type byScoreDesc []detect.ValScore

func (s byScoreDesc) Len() int           { return len(s) }
func (s byScoreDesc) Less(i, j int) bool { return s[i].Score > s[j].Score }
func (s byScoreDesc) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func shuffle(xs sort.Interface) {
	for i, j := range rand.Perm(xs.Len()) {
		xs.Swap(i, j)
	}
}
