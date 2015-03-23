package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path"

	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/featset"
	"github.com/jvlmdr/go-file/fileutil"
	"github.com/jvlmdr/shift-invar/go/toepcov"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage:", os.Args[0], "[flags] images.txt feat.json stats.(gob|json|csv)")
		flag.PrintDefaults()
	}
}

func main() {
	var (
		dir  = flag.String("images-dir", "", "Directory to which paths in images.txt are relative.")
		band = flag.Int("bandwidth", 16, "Covariance bandwidth (in feature pixels not image pixels).")
	)
	flag.Parse()
	if flag.NArg() != 3 {
		flag.Usage()
		os.Exit(1)
	}
	var (
		imsFile   = flag.Arg(0)
		featFile  = flag.Arg(1)
		covarFile = flag.Arg(2)
	)

	phi := new(featset.ImageMarshaler)
	if err := fileutil.LoadJSON(featFile, phi); err != nil {
		log.Fatalln("load feature:", err)
	}
	ims, err := fileutil.LoadLines(imsFile)
	if err != nil {
		log.Fatalln("load image list:", err)
	}
	total, err := totalStats(ims, *dir, phi, *band)
	if err != nil {
		log.Fatalln("compute stats:", err)
	}
	log.Print("save stats totals")
	if err := toepcov.SaveTotalExt(covarFile, total); err != nil {
		log.Fatalln("save stats totals:", err)
	}
}

func totalStats(files []string, dir string, phi feat.Image, band int) (*toepcov.Total, error) {
	var total *toepcov.Total
	for i, file := range files {
		log.Printf("image %d of %d: %s", i+1, len(files), file)
		distr, err := imageStats(path.Join(dir, file), phi, band)
		if err != nil {
			return nil, err
		}
		if total == nil {
			total = distr
		} else {
			total = toepcov.AddTotal(total, distr)
		}
	}
	return total, nil
}

func imageStats(file string, phi feat.Image, band int) (*toepcov.Total, error) {
	log.Print("load feature image")
	im, err := loadImage(file)
	if err != nil {
		return nil, err
	}
	f, err := phi.Apply(im)
	if err != nil {
		return nil, err
	}
	log.Printf("feature image: %d x %d x %d", f.Width, f.Height, f.Channels)
	log.Print("compute stats")
	return toepcov.Stats(f, band), nil
}
