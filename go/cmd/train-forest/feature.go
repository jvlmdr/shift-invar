package main

import (
	"image"
	"math"
	"math/rand"

	"github.com/jvlmdr/go-cv/rimg64"
)

// Feature produces a number from an image.
type Feature interface {
	Eval(*rimg64.Multi) float64
}

// ElemFeature returns an element in the image.
type ElemFeature struct {
	Point   image.Point
	Channel int
}

func (f ElemFeature) Eval(im *rimg64.Multi) float64 {
	return im.At(f.Point.X, f.Point.Y, f.Channel)
}

// SumFeature returns the sum over a rectangle region in one channel.
type SumFeature struct {
	Region  image.Rectangle
	Channel int
}

// DiffFeature returns the difference between two elements.
type DiffFeature struct {
	Point1, Point2     image.Point
	Channel1, Channel2 int
}

func (f DiffFeature) Eval(im *rimg64.Multi) float64 {
	return im.At(f.Point1.X, f.Point1.Y, f.Channel1) - im.At(f.Point2.X, f.Point2.Y, f.Channel2)
}

type FeatureDistribution interface {
	Sample() Feature
}

// UniformElem is a uniform distribution over ElemFeatures.
type UniformElem struct {
	Size     image.Point
	Channels int
}

func (d UniformElem) Sample() Feature {
	u := rand.Intn(d.Size.X)
	v := rand.Intn(d.Size.Y)
	p := rand.Intn(d.Channels)
	return ElemFeature{image.Pt(u, v), p}
}

// UniformDiff is a distribution over DiffFeatures.
type UniformDiff struct {
	Size     image.Point
	Channels int
}

func (d UniformDiff) Sample() Feature {
	u := rand.Intn(d.Size.X)
	v := rand.Intn(d.Size.Y)
	p := rand.Intn(d.Channels)
	i := rand.Intn(d.Size.X)
	j := rand.Intn(d.Size.Y)
	q := rand.Intn(d.Channels)
	return DiffFeature{Point1: image.Pt(u, v), Channel1: p, Point2: image.Pt(i, j), Channel2: q}
}

// NormalDiff is a distribution over difference features
// where the position of the second point follows a Gaussian
// about the first point.
type NormalDiff struct {
	Size        image.Point
	Channels    int
	SameChannel bool
	Sigma       float64
}

func (d NormalDiff) Sample() Feature {
	u := rand.Intn(d.Size.X)
	v := rand.Intn(d.Size.Y)
	p := rand.Intn(d.Channels)
	var q int
	if d.SameChannel {
		q = p
	} else {
		q = rand.Intn(d.Channels)
	}
	n := d.Size.X * d.Size.Y
	pts := make([]image.Point, 0, n)
	cdf := make([]float64, 1, n+1)
	for i := 0; i < d.Size.X; i++ {
		for j := 0; j < d.Size.Y; j++ {
			dx, dy := float64(i-u), float64(j-v)
			l := math.Exp(-(dx*dx + dy*dy) / (2 * d.Sigma * d.Sigma))
			pts = append(pts, image.Pt(i, j))
			cdf = append(cdf, cdf[len(cdf)-1]+l)
		}
	}
	// Find x such that cdf[x] <= r < cdf[x+1], or
	// the minimum x such that r < cdf[x+1].
	x := MultinomSum(cdf)
	return DiffFeature{Point1: image.Pt(u, v), Channel1: p, Point2: pts[x], Channel2: q}
}
