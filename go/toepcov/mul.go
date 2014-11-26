package whog

import "github.com/jvlmdr/go-cv/rimg64"

type MulFunc func(*Covar, *rimg64.Multi) *rimg64.Multi

// Default multiplication method.
var Mul MulFunc = MulFFT
