package toepcov

import (
	"bytes"
	"testing"
)

func TestDecodeTotalCSV(t *testing.T) {
	const (
		chans = 3
		band  = 4
	)
	f := randImage(4*band, 3*band, chans)
	want := Stats(f, band)
	var b bytes.Buffer
	if err := EncodeTotalCSV(&b, want); err != nil {
		t.Fatal("encode:", err)
	}
	gotChans, gotBand, err := DecodeTotalSizeCSV(bytes.NewReader(b.Bytes()))
	if err != nil {
		t.Fatal("decode size:", err)
	}
	if gotChans != chans {
		t.Fatalf("different number of channels: want %d, got %d", chans, gotChans)
	}
	if gotBand != band {
		t.Fatalf("different bandwidth: want %d, got %d", band, gotBand)
	}
	got, err := DecodeTotalCSV(bytes.NewReader(b.Bytes()), chans, band)
	if err != nil {
		t.Fatal("decode:", err)
	}
	if !sliceEq(t, want.MeanTotal, got.MeanTotal, 0) {
		return
	}
	if !covarEq(t, want.CovarTotal, got.CovarTotal, 0) {
		return
	}
	if !countEq(t, want.Count, got.Count) {
		return
	}
	if want.Images != got.Images {
		t.Fatalf("different number of images: want %d, got %d", want.Images, got.Images)
	}
}
