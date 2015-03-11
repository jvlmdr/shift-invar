package imset

// VecSet presents the images in a Set as vectors.
type VecSet struct {
	Set
	Bias float64
}

func (set *VecSet) addBias() bool {
	return set.Bias != 0 // If NaN, return false.
}

func (set *VecSet) Dim() int {
	size, channels := set.ImageSize(), set.ImageChannels()
	n := size.X * size.Y * channels
	if set.addBias() {
		n++
	}
	return n
}

func (set *VecSet) At(i int) []float64 {
	x := set.Set.At(i)
	v := x.Elems
	if set.addBias() {
		v = append(v, set.Bias)
	}
	return v
}
