package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path"
	"strings"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-file/fileutil"
	//	"github.com/jvlmdr/shift-invar/go/data"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage:", os.Args[0], "[flags] dets.json")
		flag.PrintDefaults()
	}
}

func main() {
	//	var (
	//		datasetName  = flag.String("dataset", "", "{inria, caltech}")
	//		datasetSpec  = flag.String("dataset-spec", "", "Dataset parameters (JSON)")
	//	)
	flag.Parse()
	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(1)
	}
	detsFile := flag.Arg(0)

	//	// Load dataset.
	//	dataset, err := data.Load(*datasetName, *datasetSpec)
	//	if err != nil {
	//		log.Fatalln("load dataset:", err)
	//	}
	// Load detections.
	var imdets map[string][]detect.Det
	if err := fileutil.LoadExt(detsFile, &imdets); err != nil {
		log.Fatalln("load detections:", err)
	}
	// Save detections.
	for name, dets := range imdets {
		dir, file, err := splitName(name)
		if err != nil {
			log.Fatalln("split name:", err)
		}
		if err := os.MkdirAll(dir, 0755); err != nil {
			log.Fatalln("mkdir:", err)
		}
		if err := saveDets(path.Join(dir, file+".txt"), dets); err != nil {
			log.Fatalf("save detections: %v", err)
		}
	}
}

func splitName(name string) (dir, file string, err error) {
	parts := strings.Split(name, "/")
	if len(parts) < 2 {
		return "", "", fmt.Errorf(`name does not contain /: "%s"`, name)
	}
	dir = strings.Join(parts[:len(parts)-1], "/")
	file = parts[len(parts)-1]
	return
}

func saveDets(fname string, dets []detect.Det) error {
	file, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer file.Close()
	return writeDets(file, dets)
}

func writeDets(w io.Writer, dets []detect.Det) error {
	for _, det := range dets {
		r := det.Rect
		fmt.Fprintf(w, "%d,%d,%d,%d,%e\n", r.Min.X, r.Min.Y, r.Dx(), r.Dy(), det.Score)
	}
	return nil
}
