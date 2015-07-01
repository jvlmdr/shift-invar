package main

import (
	"fmt"
	"image"
	"log"
	"path"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/imsamp"
	"github.com/jvlmdr/go-file/fileutil"
	"github.com/jvlmdr/go-pbs-pro/dstrfn"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/nfnt/resize"
)

func init() {
	dstrfn.RegisterMap("train", true, dstrfn.ConfigFunc(train))
}

type TrainInput struct {
	DetectorKey
	Images []string
}

func train(u TrainInput, datasetMessage DatasetMessage, covarDir string, addFlip bool, interp resize.InterpolationFunction, searchOptsMsg MultiScaleOptsMessage) (string, error) {
	fmt.Printf("%s\t%s\n", u.Param.Ident(), u.Param.Serialize())
	examplePad := u.Param.TrainPad
	exampleOpts := data.ExampleOpts{
		AspectReject: u.Param.AspectReject,
		FitMode:      u.Param.ResizeFor,
		MaxScale:     u.Param.MaxTrainScale,
	}
	// Determine dimensions of template.
	phi := u.Feat.Transform.Transform()
	maxSize := u.Size.Add(image.Pt(2*examplePad, 2*examplePad))
	region := windowGeom(u.Size, maxSize, phi)
	log.Printf("window size %v, object bounds %v", region.Size, region.Int)
	// Supply training algorithm with search options.
	// TODO: phi will be decoded twice. Is this an issue?
	searchOpts := searchOptsMsg.Content(u.Param, imsamp.Continue, u.Overlap.Spec.Eval)

	// Re-load dataset on execution host.
	dataset, err := data.Load(datasetMessage.Name, datasetMessage.Spec)
	if err != nil {
		return "", err
	}
	// Split images into positive and negative.
	var posIms, negIms []string
	for _, im := range u.Images {
		if dataset.IsNeg(im) {
			negIms = append(negIms, im)
		} else {
			posIms = append(posIms, im)
		}
	}
	// Take subset of negative images.
	numNegIms := int(u.NegFrac * float64(len(negIms)))
	negIms = selectSubset(negIms, randSubset(len(negIms), numNegIms))
	log.Println("number of negative images:", len(negIms))

	statsFile := path.Join(covarDir, u.Feat.StatsFile)
	result, err := u.Trainer.Spec.Train(posIms, negIms, dataset, phi, statsFile, region, exampleOpts, addFlip, interp, searchOpts)
	if err != nil {
		return "", err
	}
	if err := fileutil.SaveExt(u.TmplFile(), result); err != nil {
		return "", err
	}
	return u.TmplFile(), nil
}

func windowGeom(obj, max image.Point, phi feat.Image) detect.PadRect {
	min := phi.MinInputSize(phi.Size(max))
	rect := image.Rectangle{Max: obj}.Add(min.Sub(obj).Div(2))
	return detect.PadRect{min, rect}
}
