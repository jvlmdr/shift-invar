package main

import (
	"fmt"
	"image"
	"log"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-file/fileutil"
	"github.com/jvlmdr/go-pbs-pro/dstrfn"
	"github.com/jvlmdr/go-svm/svm"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/jvlmdr/shift-invar/go/vecset"
	"github.com/nfnt/resize"
)

func init() {
	dstrfn.RegisterMap("train", false, dstrfn.ConfigFunc(train))
}

type TrainInput struct {
	Fold int
	Param
}

func (x TrainInput) Hash() string {
	return fmt.Sprintf("param-%s-fold-%d", x.Param.Hash(), x.Fold)
}

func (x TrainInput) TmplFile() string {
	return fmt.Sprintf("tmpl-%s.gob", x.Hash())
}

func train(cfg TrainInput, foldIms [][]string, datasetName, datasetSpec string, pad int, exampleOpts data.ExampleOpts, bias float64, addFlip bool, interp resize.InterpolationFunction) (string, error) {
	fmt.Printf("%s\t%s\n", cfg.Param.Hash(), cfg.Param.ID())
	// Determine dimensions of template.
	region := detect.PadRect{
		Size: image.Pt(cfg.Size.X+pad*2, cfg.Size.Y+pad*2),
		Int:  image.Rectangle{image.ZP, cfg.Size}.Add(image.Pt(pad, pad)),
	}
	phi := cfg.Feat.Transform()

	// Determine training images.
	trainIms := mergeExcept(foldIms, cfg.Fold)
	// Re-load dataset on execution host.
	dataset, err := data.Load(datasetName, datasetSpec)
	if err != nil {
		return "", err
	}
	// Extract positive examples and negative images.
	examples, err := data.ExtractTrainingSet(dataset, trainIms, region, exampleOpts)
	if err != nil {
		log.Fatal(err)
	}
	// Take subset of negative images.
	numNegIms := int(cfg.NegFrac * float64(len(examples.NegImages)))
	negIms := selectSubset(examples.NegImages, subset(len(examples.NegImages), numNegIms))
	log.Println("number of negative images:", len(negIms))

	// Positive examples are extracted and stored as vectors.
	pos, err := data.PosExamples(examples.PosImages, examples.PosRects, dataset, phi, bias, region, addFlip, interp)
	if err != nil {
		return "", err
	}
	if len(pos) == 0 {
		return "", fmt.Errorf("empty positive set")
	}
	// Negative examples are represented as indices into an image.
	neg, err := data.NegWindowSets(negIms, dataset, phi, bias, region, interp)
	if err != nil {
		return "", err
	}
	if len(neg) == 0 {
		return "", fmt.Errorf("empty negative set")
	}
	// Count number of examples for cost normalization.
	var numNegWindows int
	for i := range neg {
		numNegWindows += neg[i].Len()
	}

	var (
		x []vecset.Set
		y []float64
		c []float64
	)
	// Add positive examples as a set of vectors.
	x = append(x, vecset.Slice(pos))
	for _ = range pos {
		y = append(y, 1)
		c = append(c, cfg.Gamma/cfg.Lambda/float64(len(pos)))
	}
	// Add each set of negative vectors.
	for i := range neg {
		x = append(x, neg[i])
		ni := neg[i].Len()
		// Labels and costs for every positive and negative example.
		for j := 0; j < ni; j++ {
			y = append(y, -1)
			c = append(c, (1-cfg.Gamma)/cfg.Lambda/float64(numNegWindows))
		}
	}

	weights, err := svm.Train(vecset.NewUnion(x), y, c,
		func(epoch int, f, fPrev, g, gPrev float64, w, wPrev []float64, a, aPrev map[int]float64) (bool, error) {
			if epoch < cfg.Epochs {
				return false, nil
			}
			return true, nil
		},
	)
	if err != nil {
		return "", err
	}

	featsize := phi.Size(region.Size)
	channels := phi.Channels()
	// Pack weights into detection template.
	tmpl := &detect.FeatTmpl{
		Image: &rimg64.Multi{
			Width:    featsize.X,
			Height:   featsize.Y,
			Channels: channels,
			// Exclude bias term if present.
			Elems: weights[:featsize.X*featsize.Y*channels],
		},
		Size:     region.Size,
		Interior: region.Int,
	}
	if err := fileutil.SaveExt(cfg.TmplFile(), tmpl); err != nil {
		return "", err
	}
	return cfg.TmplFile(), nil
}
