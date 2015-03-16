package main

import (
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
)

func loadImage(fname string) (image.Image, error) {
	file, err := os.Open(fname)
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

func area(r image.Rectangle) int { return r.Dx() * r.Dy() }

func min(a, b int) int {
	if b < a {
		return b
	}
	return a
}
