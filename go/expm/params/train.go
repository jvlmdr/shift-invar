package main

import (
	"fmt"
	"image"
	"log"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/imsamp"
	"github.com/jvlmdr/go-file/fileutil"
	"github.com/jvlmdr/go-pbs-pro/dstrfn"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/nfnt/resize"
)

func init() {
	dstrfn.RegisterMap("train", false, dstrfn.ConfigFunc(train))
}

// DetectorKey is the Cartesian product of detector parameters
// and a training set identifier.
type DetectorKey struct {
	Ident string // e.g. "fold-2" or "test"
	Param
}

func (x DetectorKey) Key() string {
	return fmt.Sprintf("param-%s-%s", x.Param.Key(), x.Ident)
}

func (x DetectorKey) TmplFile() string {
	return fmt.Sprintf("tmpl-%s.gob", x.Key())
}

func (x DetectorKey) PerfFile() string {
	return fmt.Sprintf("perf-%s.json", x.Key())
}

func CrossValKey(p Param, fold int) DetectorKey {
	return DetectorKey{Ident: fmt.Sprintf("fold-%d", fold), Param: p}
}

func TestKey(p Param) DetectorKey {
	return DetectorKey{Ident: "test", Param: p}
}

type TrainInput struct {
	DetectorKey
	Images []string
}

func train(u TrainInput, datasetName, datasetSpec string, pad int, exampleOpts data.ExampleOpts, addFlip bool, interp resize.InterpolationFunction, searchOptsMsg MultiScaleOptsMessage) (string, error) {
	fmt.Printf("%s\t%s\n", u.Param.Key(), u.Param.ID())
	// Determine dimensions of template.
	region := detect.PadRect{
		Size: image.Pt(u.Size.X+pad*2, u.Size.Y+pad*2),
		Int:  image.Rectangle{image.ZP, u.Size}.Add(image.Pt(pad, pad)),
	}
	phi := u.Feat.Transform()
	// Supply training algorithm with search options.
	searchOpts := searchOptsMsg.Content(phi, imsamp.Continue, u.Overlap.Spec.Eval)

	// Re-load dataset on execution host.
	dataset, err := data.Load(datasetName, datasetSpec)
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

	tmpl, err := u.Trainer.Spec.Train(posIms, negIms, dataset, phi, region, exampleOpts, addFlip, interp, searchOpts)
	if err != nil {
		return "", err
	}
	if err := fileutil.SaveExt(u.TmplFile(), tmpl); err != nil {
		return "", err
	}
	return u.TmplFile(), nil
}
