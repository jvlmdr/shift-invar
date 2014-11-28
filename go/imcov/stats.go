package imcov

import "github.com/jvlmdr/go-cv/rimg64"

// CovarSum computes the un-normalized covariance of all windows in an image.
func CovarSum(im *rimg64.Multi, width, height int) *Covar {
	if im.Width < width || im.Height < height {
		return nil
	}

	cov := NewCovar(width, height, im.Channels)
	for a := 0; a < im.Width-width+1; a++ {
		for b := 0; b < im.Height-height+1; b++ {
			for u := 0; u < width; u++ {
				for v := 0; v < height; v++ {
					for p := 0; p < im.Channels; p++ {
						for i := 0; i < width; i++ {
							for j := 0; j < height; j++ {
								for q := 0; q < im.Channels; q++ {
									uvp := im.At(a+u, b+v, p)
									ijq := im.At(a+i, b+j, q)
									cov.AddAt(u, v, p, i, j, q, uvp*ijq)
								}
							}
						}
					}
				}
			}
		}
	}
	return cov
}

// MeanSum computes the un-normalized mean of all windows in an image.
func MeanSum(im *rimg64.Multi, width, height int) *rimg64.Multi {
	if im.Width < width || im.Height < height {
		return nil
	}

	mu := rimg64.NewMulti(width, height, im.Channels)
	for a := 0; a < im.Width-width+1; a++ {
		for b := 0; b < im.Height-height+1; b++ {
			for u := 0; u < width; u++ {
				for v := 0; v < height; v++ {
					for w := 0; w < im.Channels; w++ {
						prev := mu.At(u, v, w)
						curr := im.At(a+u, b+v, w)
						mu.Set(u, v, w, prev+curr)
					}
				}
			}
		}
	}
	return mu
}
