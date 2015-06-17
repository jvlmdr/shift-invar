package main

import (
	"flag"
	"fmt"
	"image"
	"log"
	"os"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-file/fileutil"
	"github.com/jvlmdr/shift-invar/go/data"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage:", os.Args[0], "[flags] dets.json")
		flag.PrintDefaults()
	}
}

func main() {
	var (
		datasetName = flag.String("dataset", "", "{inria, caltech}")
		datasetSpec = flag.String("dataset-spec", "", "Dataset parameters (JSON)")
	)
	flag.Parse()
	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(1)
	}
	detsFile := flag.Arg(0)

	// Load dataset.
	dataset, err := data.Load(*datasetName, *datasetSpec)
	if err != nil {
		log.Fatalln("load dataset:", err)
	}
	// Load detections.
	var imdets map[string][]detect.Det
	if err := fileutil.LoadExt(detsFile, &imdets); err != nil {
		log.Fatalln("load detections:", err)
	}
	// Remove images from list which should not be used for testing.
	var ims []string
	for _, im := range dataset.Images() {
		ims = append(ims, im)
	}

	for _, im := range ims {
		dets, ok := imdets[im]
		if !ok {
			continue
		}
		annot := dataset.Annot(im)
		Validate(dets, annot.Instances, annot.Ignore, 0.5, 0.5)
		fmt.Println()
	}
}

func rectStr(r image.Rectangle) string {
	return fmt.Sprintf("[%d %d %d %d]", r.Min.X, r.Min.Y, r.Dx(), r.Dy())
}
