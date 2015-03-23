package main

import (
	"image"
	"image/jpeg"
	"image/png"
	"os"
)

func loadImage(name string) (image.Image, error) {
	file, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	im, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	return im, nil
}

func loadImageSize(name string) (image.Point, error) {
	file, err := os.Open(name)
	if err != nil {
		return image.ZP, err
	}
	defer file.Close()
	cfg, _, err := image.DecodeConfig(file)
	if err != nil {
		return image.ZP, err
	}
	return image.Pt(cfg.Width, cfg.Height), nil
}

func saveJPEG(fname string, im image.Image) error {
	f, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer f.Close()
	return jpeg.Encode(f, im, nil)
}

func savePNG(fname string, im image.Image) error {
	f, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, im)
}
