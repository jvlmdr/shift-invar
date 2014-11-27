package toepcov

type Count struct {
	Band  int
	Elems [][]int64
}

func NewCount(band int) *Count {
	n := 2*band + 1
	elems := make([][]int64, n)
	for i := range elems {
		elems[i] = make([]int64, n)
	}
	return &Count{band, elems}
}

func (cnt *Count) At(i, j int) int64 {
	b := cnt.Band
	return cnt.Elems[b+i][b+j]
}

func (cnt *Count) Set(i, j int, val int64) {
	b := cnt.Band
	cnt.Elems[b+i][b+j] = val
}

func (src *Count) Clone() *Count {
	dst := NewCount(src.Band)
	for i := range src.Elems {
		copy(dst.Elems[i], src.Elems[i])
	}
	return dst
}

// Adds two relative-position counts.
func AddCount(lhs, rhs *Count) *Count {
	// Swap pointers such that lhs.Band >= rhs.Band.
	if lhs.Band < rhs.Band {
		lhs, rhs = rhs, lhs
	}
	// Clone total with larger bandwidth.
	dst := lhs.Clone()

	// Add values and counts.
	for i := -rhs.Band; i <= rhs.Band; i++ {
		for j := -rhs.Band; j <= rhs.Band; j++ {
			dst.Set(i, j, dst.At(i, j)+rhs.At(i, j))
		}
	}
	return dst
}
