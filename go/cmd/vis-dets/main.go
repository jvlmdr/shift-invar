package main

import (
	"flag"
	"fmt"
	"image"
	"log"
	"os"
	"os/exec"
	"strings"

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
		minScore    = flag.Float64("min-score", 0, "Minimum score to include detection")
		datasetName = flag.String("dataset", "", "{inria, caltech}")
		datasetSpec = flag.String("dataset-spec", "", "Dataset parameters (JSON)")
	)
	flag.Parse()
	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(1)
	}
	detsFile := flag.Arg(0)

	// Load training data and determine cross-validation splits.
	// Use same partitions for all methods.
	dataset, err := data.Load(*datasetName, *datasetSpec)
	if err != nil {
		log.Fatalln("load dataset:", err)
	}

	// Load detections.
	var imdets map[string][]detect.Det
	if err := fileutil.LoadExt(detsFile, &imdets); err != nil {
		log.Fatalln("load detections:", err)
	}

	for im, dets := range imdets {
		log.Println("image:", im)
		imfile := dataset.File(im)
		dets = filterDets(dets, *minScore)
		dst := strings.Replace(im, "/", "-", -1) + ".jpeg"
		if err := drawDets(dst, imfile, dets); err != nil {
			log.Fatalln("draw detections:", err)
		}
	}
}

func filterDets(dets []detect.Det, minScore float64) []detect.Det {
	var keep []detect.Det
	for _, det := range dets {
		if det.Score >= minScore {
			keep = append(keep, det)
		}
	}
	return keep
}

func drawDets(dst, src string, dets []detect.Det) error {
	var args []string
	args = append(args, src)
	args = append(args, "-strokewidth", "2")
	//args = append(args, "-undercolor", "#00000080")
	for i := range dets {
		// Reverse.
		det := dets[len(dets)-1-i]
		r := det.Rect
		// Draw rectangle.
		args = append(args, "-stroke", "blue", "-fill", "none")
		args = append(args, "-draw", rectStr(r))
		// Label with score.
		text := fmt.Sprintf("%.3g", det.Score)
		args = append(args, "-fill", "white", "-stroke", "none")
		pos := fmt.Sprintf("+%d+%d", r.Min.X, r.Max.Y)
		args = append(args, "-annotate", pos, text)
	}
	args = append(args, dst)
	cmd := exec.Command("convert", args...)
	return cmd.Run()
}

func rectStr(r image.Rectangle) string {
	return fmt.Sprintf("rectangle %d,%d %d,%d", r.Min.X, r.Min.Y, r.Max.X, r.Max.Y)
}
