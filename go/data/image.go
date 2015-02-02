package data

import (
	"image"
	_ "image/jpeg"
	_ "image/png"
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
