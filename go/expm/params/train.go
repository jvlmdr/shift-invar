package main

import (
	"fmt"
	"image"
	"log"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-file/fileutil"
	"github.com/jvlmdr/go-pbs-pro/dstrfn"
	"github.com/jvlmdr/shift-invar/go/data"
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

func (x TrainInput) PerfFile() string {
	return fmt.Sprintf("perf-%s.json", x.Hash())
}

func train(u TrainInput, foldIms [][]string, datasetName, datasetSpec string, pad int, exampleOpts data.ExampleOpts, addFlip bool, interp resize.InterpolationFunction) (string, error) {
	fmt.Printf("%s\t%s\n", u.Param.Hash(), u.Param.ID())
	// Determine dimensions of template.
	region := detect.PadRect{
		Size: image.Pt(u.Size.X+pad*2, u.Size.Y+pad*2),
		Int:  image.Rectangle{image.ZP, u.Size}.Add(image.Pt(pad, pad)),
	}
	phi := u.Feat.Transform()

	// Determine training images.
	trainIms := mergeExcept(foldIms, u.Fold)
	// Re-load dataset on execution host.
	dataset, err := data.Load(datasetName, datasetSpec)
	if err != nil {
		return "", err
	}
	// Extract positive examples and negative images.
	examples, err := data.ExtractTrainingSet(dataset, trainIms, region, exampleOpts)
	if err != nil {
		return "", err
	}
	// Take subset of negative images.
	numNegIms := int(u.NegFrac * float64(len(examples.NegImages)))
	examples.NegImages = selectSubset(examples.NegImages, randSubset(len(examples.NegImages), numNegIms))
	log.Println("number of negative images:", len(examples.NegImages))

	tmpl, err := u.Trainer.Spec.Train(examples, dataset, phi, region, addFlip, interp)
	if err != nil {
		return "", err
	}
	if err := fileutil.SaveExt(u.TmplFile(), tmpl); err != nil {
		return "", err
	}
	return u.TmplFile(), nil
}
