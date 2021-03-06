package main

import (
	"fmt"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"math"
	"time"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/imsamp"
	"github.com/jvlmdr/go-file/fileutil"
	"github.com/jvlmdr/go-pbs-pro/dstrfn"
	"github.com/jvlmdr/shift-invar/go/data"
	"github.com/nfnt/resize"
)

func init() {
	dstrfn.RegisterMap("test", true, dstrfn.ConfigFunc(test))
}

// Remove feat.Pad since feat.Pad.Extend cannot be marshaled.
type MultiScaleOptsMessage struct {
	Interp resize.InterpolationFunction
	// MinScore will be ignored and set to -inf.
	detect.DetFilter
	// Replace SupprFilter with SupprMaxNum due to functional member.
	SupprMaxNum int
}

// Content combines MultiScaleOptsMessage, Param, and other parameters into MultiScaleOpts.
// TODO: Is it ugly for Transform() to be called internally?
func (msg MultiScaleOptsMessage) Content(param Param, padExtend imsamp.At, overlap detect.OverlapFunc) detect.MultiScaleOpts {
	// Override DetFilter.MinScore.
	detFilter := msg.DetFilter
	detFilter.MinScore = math.Inf(-1)
	margin := param.TrainPad + param.TestMargin
	return detect.MultiScaleOpts{
		MaxScale:    param.MaxTestScale,
		PyrStep:     param.PyrStep,
		Interp:      msg.Interp,
		Transform:   param.Feat.Transform.Transform(),
		Pad:         feat.Pad{feat.UniformMargin(margin), padExtend},
		DetFilter:   detFilter,
		SupprFilter: detect.SupprFilter{msg.SupprMaxNum, overlap},
	}
}

type TestInput struct {
	ResultsKey
	Images []string
}

func test(x TestInput, datasetMessage DatasetMessage, optsMsg MultiScaleOptsMessage, minMatchIOU, minIgnoreCover float64, fppis []float64) (float64, error) {
	fmt.Printf("%s\t%s\n", x.Param.Ident(), x.Param.Serialize())
	opts := optsMsg.Content(x.Param, imsamp.Continue, x.Param.Overlap.Spec.Eval)
	// Load template from disk.
	trainResult := new(TrainResult)
	if err := fileutil.LoadExt(x.TmplFile(), trainResult); err != nil {
		return 0, err
	}
	if trainResult.Report.Error != "" {
		// This function should not have been called for this configuration.
		return 0, fmt.Errorf("training error: %s", trainResult.Report.Error)
	}
	tmpl := trainResult.Tmpl

	// Re-load dataset on execution host.
	dataset, err := data.Load(datasetMessage.Name, datasetMessage.Spec)
	if err != nil {
		return 0, err
	}
	// Remove images from list which should not be used for testing.
	var ims []string
	for _, name := range x.Images {
		if dataset.CanTest(name) {
			ims = append(ims, name)
		}
	}

	// Cache validated detections.
	// Should have made this map[string]*ValImage for debug?
	var imvals []*detect.ValSet
	err = fileutil.Cache(&imvals, fmt.Sprintf("val-dets-%s.json", x.Ident()), func() ([]*detect.ValSet, error) {
		imdets := make(map[string][]detect.Det)
		var imvals []*detect.ValSet // Shadow variable in parent scope.
		for i, name := range ims {
			log.Printf("test image %d / %d: %s", i+1, len(ims), name)
			// Load image.
			file := dataset.File(name)
			t := time.Now()
			im, err := loadImage(file)
			if err != nil {
				return nil, err
			}
			durLoad := time.Since(t)
			dets, durSearch, err := detect.MultiScale(im, tmpl.Scorer, tmpl.PixelShape, opts)
			if err != nil {
				return nil, err
			}
			imdets[name] = dets
			// Validate detections.
			annot := dataset.Annot(name)
			imval := detect.Validate(dets, annot.Instances, annot.Ignore, minMatchIOU, minIgnoreCover)
			imvals = append(imvals, imval.Set())
			log.Printf(
				"load %v, resize %v, feat %v, slide %v, suppr %v",
				durLoad, durSearch.Resize, durSearch.Feat, durSearch.Slide, durSearch.Suppr,
			)
		}
		// Also save (un-validated) detections.
		err = fileutil.SaveExt(fmt.Sprintf("dets-%s.json", x.Ident()), imdets)
		if err != nil {
			return nil, err
		}
		return imvals, nil
	})
	if err != nil {
		return 0, err
	}

	valset := detect.MergeValSets(imvals...)
	// Get average miss rate.
	rates := detect.MissRateAtFPPIs(valset, fppis)
	for i := range rates {
		log.Printf("fppi %g, miss rate %g", fppis[i], rates[i])
	}
	perf := logAvg(rates)
	if err := fileutil.SaveExt(x.PerfFile(), perf); err != nil {
		return 0, err
	}
	return perf, nil
}

func logAvg(xs []float64) float64 {
	var t float64
	for _, x := range xs {
		t += math.Log(math.Max(0, x))
	}
	t /= float64(len(xs))
	return math.Exp(t)
}
