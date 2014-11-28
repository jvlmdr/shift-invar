package toepcov

import "github.com/jvlmdr/go-cv/rimg64"

// MulFunc is the type of a function which multiples a covariance by an image.
type MulFunc func(*Covar, *rimg64.Multi) *rimg64.Multi

// Mul is the default multiplication method.
var Mul MulFunc = MulFFT
