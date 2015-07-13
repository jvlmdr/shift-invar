package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-file/fileutil"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage:", os.Args[0], "[flags] val-dets.json")
		flag.PrintDefaults()
	}
}

func main() {
	fppi := flag.Float64("fppi", 1, "Maximum false positives per image")
	flag.Parse()
	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(1)
	}
	valDetsFile := flag.Arg(0)

	// Load validated detections from file.
	var imvals []*detect.ValSet
	if err := fileutil.LoadExt(valDetsFile, &imvals); err != nil {
		log.Fatalln("load validations:", err)
	}
	valset := detect.MergeValSets(imvals...)
	// FP / N <= fppi
	maxFalsePos := int((*fppi) * float64(len(imvals)))
	threshold := identify(valset, maxFalsePos)
	fmt.Println(threshold)
}

func identify(valset *detect.ValSet, maxFalsePos int) float64 {
	path := valset.Enum()
	var last int
	for i, perf := range path {
		if perf.FP > maxFalsePos {
			break
		}
		last = i
	}

	if last == 0 {
		return math.Inf(1)
	}
	if last == len(path)-1 {
		return math.Inf(-1)
	}
	return valset.Dets[last-1].Score
}
