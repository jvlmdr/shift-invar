package data

import (
	"fmt"
	"image"
	"log"
	"time"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/imsamp"
	"github.com/jvlmdr/shift-invar/go/vecset"
	"github.com/nfnt/resize"
)

func PosExamples(ims []string, rects map[string][]image.Rectangle, dataset ImageSet, phi feat.Image, bias float64, shape detect.PadRect, addFlip bool, interp resize.InterpolationFunction) ([][]float64, error) {
	var pos [][]float64
	for _, name := range ims {
		if !dataset.CanTrain(name) {
			continue
		}
		log.Println("load positive image:", name)
		t := time.Now()
		file := dataset.File(name)
		im, err := loadImage(file)
		if err != nil {
			log.Printf("load positive image: %s, error: %v", file, err)
			continue
		}
		durLoad := time.Since(t)
		var durSamp, durResize, durFlip, durFeat time.Duration
		for _, rect := range rects[name] {
			// Extract and resize window.
			t = time.Now()
			subim := imsamp.Rect(im, rect, imsamp.Continue)
			durSamp += time.Since(t)
			t = time.Now()
			subim = resize.Resize(uint(shape.Size.X), uint(shape.Size.Y), subim, interp)
			durResize += time.Since(t)
			// Add flip if desired.
			flips := []bool{false}
			if addFlip {
				flips = []bool{false, true}
			}
			for _, flip := range flips {
				pix := subim
				t = time.Now()
				if flip {
					pix = flipImageX(subim)
				}
				durFlip += time.Since(t)
				t = time.Now()
				x, err := phi.Apply(pix)
				if err != nil {
					return nil, err
				}
				durFeat += time.Since(t)
				vec := x.Elems
				if bias != 0 {
					vec = append(vec, bias)
				}
				pos = append(pos, vec)
			}
		}
		log.Printf(
			"load %.3gms, sample %.3gms, resize %.3gms, flip %.3gms, feat %.3gms",
			durLoad.Seconds()*1000, durSamp.Seconds()*1000, durResize.Seconds()*1000,
			durFlip.Seconds()*1000, durFeat.Seconds()*1000,
		)
	}
	return pos, nil
}

func flipImageX(src image.Image) image.Image {
	r := src.Bounds()
	dst := image.NewRGBA64(r)
	q := dst.Bounds()
	for j := 0; j < q.Dy(); j++ {
		for i := 0; i < q.Dx(); i++ {
			dst.Set(q.Min.X+i, q.Min.Y+j, src.At(r.Max.X-1-i, r.Min.Y+j))
		}
	}
	return dst
}

func NegWindowSets(negIms []string, dataset ImageSet, phi feat.Image, bias float64, shape detect.PadRect, interp resize.InterpolationFunction) ([]*vecset.WindowSet, error) {
	featSize := phi.Size(shape.Size)
	var neg []*vecset.WindowSet
	for _, name := range negIms {
		if !dataset.CanTrain(name) {
			continue
		}
		log.Println("load negative image:", name)
		t := time.Now()
		file := dataset.File(name)
		im, err := loadImage(file)
		if err != nil {
			log.Printf("load negative image: %s, error: %v", file, err)
			continue
		}
		durLoad := time.Since(t)
		t = time.Now()
		// Take transform of entire image.
		x, err := phi.Apply(im)
		if err != nil {
			return nil, err
		}
		durFeat := time.Since(t)
		set := new(vecset.WindowSet)
		set.Image = x
		set.Size = featSize
		for u := 0; u < x.Width-featSize.X+1; u++ {
			for v := 0; v < x.Height-featSize.Y+1; v++ {
				set.Windows = append(set.Windows, image.Pt(u, v))
			}
		}
		set.Bias = bias
		neg = append(neg, set)
		log.Printf("load %.3gms, feat %.3gms", durLoad.Seconds()*1000, durFeat.Seconds()*1000)
	}
	if len(neg) == 0 {
		return nil, fmt.Errorf("empty negative set")
	}
	return neg, nil
}
