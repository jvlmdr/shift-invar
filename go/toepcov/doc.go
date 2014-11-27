/*
Package toepcov provides Toeplitz covariance for feature images.

To accumulate evidence for a covariance matrix from a list of feature images:
	func distr(ims []*rimg64.Multi, bandwidth int) *toepcov.Distr {
		var total *toepcov.Total
		for _, im := range ims {
			curr := toepcov.Stats(im, bandwidth)
			total = toepcov.AddTotalToEither(total, curr)
		}
		return toepcov.Normalize(total, true)
	}

Computing the weights of a detector from the mean positive example can be done either using Cholesky factorization:
	func train(distr *toepcov.Distr, pos *rimg64.Multi) (*rimg64.Multi, error) {
		b := toepcov.SubMean(pos, distr.Mean)
		return toepcov.InvMulDirect(distr.Covar, b)
	}
or conjugate gradient:
	func train(distr *toepcov.Distr, pos *rimg64.Multi, tol float64, iter int) (*rimg64.Multi, error) {
		b := toepcov.SubMean(pos, distr.Mean)
		zero := rimg64.NewMulti(b.Width, b.Height, b.Channels)
		return toepcov.InvMulConjGrad(distr.Covar, b, zero, tol, iter, os.Stderr)
	}
There is also a preconditioned conjugate gradient method for Toeplitz covariance matrices in package circcov.

It may be necessary to add some regularization between these steps:
	distr.Covar.AddLambdaI(1e-3)
*/
package toepcov
