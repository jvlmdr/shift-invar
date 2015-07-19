package data

import (
	"image"
	"log"
	"time"

	"github.com/jvlmdr/go-cv/detect"
	"github.com/jvlmdr/go-cv/feat"
	"github.com/jvlmdr/go-cv/imsamp"
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/shift-invar/go/imset"
	"github.com/nfnt/resize"
)

// Examples extracts windows from the image, resizes them to
// the given size and computes their feature transform.
// Does not check dataset.CanTrain or CanTest.
func Examples(ims []string, rects map[string][]image.Rectangle, dataset ImageSet, phi feat.Image, extend imsamp.At, shape detect.PadRect, addFlip bool, interp resize.InterpolationFunction) ([]*rimg64.Multi, error) {
	var examples []*rimg64.Multi
	for _, name := range ims {
		log.Println("load image:", name)
		t := time.Now()
		file := dataset.File(name)
		im, err := loadImage(file)
		if err != nil {
			return nil, err
		}
		durLoad := time.Since(t)
		var durSamp, durResize, durFlip, durFeat time.Duration
		for _, rect := range rects[name] {
			// Extract and resize window.
			t = time.Now()
			subim := imsamp.Rect(im, rect, extend)
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
				examples = append(examples, x)
			}
		}
		log.Printf(
			"load %.3gms, sample %.3gms, resize %.3gms, flip %.3gms, feat %.3gms",
			durLoad.Seconds()*1000, durSamp.Seconds()*1000, durResize.Seconds()*1000,
			durFlip.Seconds()*1000, durFeat.Seconds()*1000,
		)
	}
	return examples, nil
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

// WindowSets computes the features of each image and returns
// the set of all windows in the feature image.
// Does not check dataset.CanTrain or CanTest.
// Window size and stride are specified in feature pixels.
func WindowSets(ims []string, dataset ImageSet, phi feat.Image, pad feat.Pad, size image.Point, stride int, interp resize.InterpolationFunction) ([]imset.Set, error) {
	var sets []imset.Set
	for _, name := range ims {
		log.Println("load image:", name)
		t := time.Now()
		file := dataset.File(name)
		im, err := loadImage(file)
		if err != nil {
			return nil, err
		}
		durLoad := time.Since(t)
		t = time.Now()
		// Take transform of entire image.
		x, err := feat.ApplyPad(phi, im, pad)
		if err != nil {
			return nil, err
		}
		durFeat := time.Since(t)
		set := new(imset.WindowSet)
		set.Image = x
		set.Size = size
		for u := 0; u < x.Width-size.X+1; u += stride {
			for v := 0; v < x.Height-size.Y+1; v += stride {
				set.Windows = append(set.Windows, image.Pt(u, v))
			}
		}
		sets = append(sets, set)
		log.Printf("load %.3gms, feat %.3gms", durLoad.Seconds()*1000, durFeat.Seconds()*1000)
	}
	return sets, nil
}
