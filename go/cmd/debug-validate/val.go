package main

// Functions are copied from go-cv/detect.

import (
	"fmt"
	"image"

	"github.com/jvlmdr/go-cv/detect"
)

func Validate(dets []detect.Det, refs, ignore []image.Rectangle, refMinIOU, ignoreMinCover float64) *detect.ValImage {
	vals, miss := ValidateList(detect.DetSlice(dets), refs, ignore, refMinIOU, ignoreMinCover)
	valdets := make([]detect.ValDet, len(dets))
	for i := range dets {
		valdets[i] = detect.ValDet{dets[i], vals[i]}
	}
	return &detect.ValImage{valdets, miss}
}

func ValidateList(dets detect.DetList, refs, ignore []image.Rectangle, refMinIOU, ignoreMinCover float64) (vals []detect.Val, miss []image.Rectangle) {
	// Match rectangles and then remove any detections (incorrect or otherwise) which are ignored.
	m := detect.Match(dets, refs, refMinIOU)
	// Label each detection as true positive or false positive.
	vals = make([]detect.Val, dets.Len())
	// Record which references were matched.
	used := make(map[int]bool)
	for i := 0; i < dets.Len(); i++ {
		det := dets.At(i)
		j, p := m[i]
		if !p {
			// Detection did not have a match.
			// Check whether to ignore the false positive.
			if anyCovers(ignore, det.Rect, ignoreMinCover) {
				fmt.Printf("ignore: det %s\n", rectStr(det.Rect))
				continue
			}
			vals[i] = detect.Val{True: false}
			fmt.Printf("fp: det %s\n", rectStr(det.Rect))
			continue
		}
		if used[j] {
			panic("already matched")
		}
		used[j] = true
		vals[i] = detect.Val{True: true, Ref: refs[j]}
		fmt.Printf("match: det %s, ref %s\n", rectStr(det.Rect), rectStr(refs[j]))
	}
	miss = make([]image.Rectangle, 0, len(refs)-len(used))
	for j, ref := range refs {
		if used[j] {
			continue
		}
		fmt.Printf("miss: ref %s\n", rectStr(ref))
		miss = append(miss, ref)
	}
	return vals, miss
}

func anyCovers(ys []image.Rectangle, x image.Rectangle, minCover float64) bool {
	for _, y := range ys {
		if covers(y, x, minCover) {
			return true
		}
	}
	return false
}

// Computes whether A covers B.
func covers(a, b image.Rectangle, min float64) bool {
	return float64(area(b.Intersect(a)))/float64(area(b)) > min
}

func area(r image.Rectangle) int {
	return r.Dx() * r.Dy()
}
