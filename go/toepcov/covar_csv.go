package toepcov

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path"
	"strconv"

	"github.com/jvlmdr/go-file/fileutil"
)

func SaveCovarExt(fname string, cov *Covar) error {
	switch path.Ext(fname) {
	case ".csv":
		return saveCovarCSV(fname, cov)
	default:
		return fileutil.SaveExt(fname, cov)
	}
}

func LoadCovarExt(fname string) (*Covar, error) {
	switch path.Ext(fname) {
	case ".csv":
		return loadCovarCSV(fname)
	}
	var cov *Covar
	if err := fileutil.LoadExt(fname, &cov); err != nil {
		return nil, err
	}
	return cov, nil
}

func saveCovarCSV(fname string, cov *Covar) error {
	file, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer file.Close()
	return EncodeCovarCSV(file, cov)
}

func loadCovarCSV(fname string) (*Covar, error) {
	file, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return DecodeCovarCSV(file)
}

func EncodeCovarCSV(w io.Writer, cov *Covar) error {
	b, k := cov.Bandwidth, cov.Channels
	ww := csv.NewWriter(w)
	defer ww.Flush()
	for du := -b; du <= b; du++ {
		for dv := -b; dv <= b; dv++ {
			for p := 0; p < k; p++ {
				for q := 0; q < k; q++ {
					rec := formatRecord(du, dv, p, q, cov.At(du, dv, p, q))
					if err := ww.Write(rec); err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}

func formatRecord(du, dv, p, q int, x float64) []string {
	return []string{
		strconv.FormatInt(int64(du), 10),
		strconv.FormatInt(int64(dv), 10),
		strconv.FormatInt(int64(p), 10),
		strconv.FormatInt(int64(q), 10),
		strconv.FormatFloat(x, 'g', -1, 64),
	}
}

func DecodeCovarCSV(r io.ReadSeeker) (*Covar, error) {
	chans, band, err := decodeCovarSizeCSV(r)
	if err != nil {
		return nil, err
	}
	if _, err := r.Seek(0, 0); err != nil {
		return nil, err
	}
	cov := NewCovar(chans, band)
	rr := csv.NewReader(r)
	for {
		rec, err := rr.Read()
		if err == io.EOF {
			return cov, nil
		}
		if err != nil {
			return nil, err
		}
		du, dv, p, q, x, err := parseRecord(rec)
		if err != nil {
			return nil, err
		}
		if err := errIfBadChanPair(p, q); err != nil {
			return nil, err
		}
		cov.Set(du, dv, p, q, x)
	}
}

func decodeCovarSizeCSV(r io.Reader) (chans, band int, err error) {
	rr := csv.NewReader(r)
	for {
		rec, err := rr.Read()
		if err == io.EOF {
			return chans, band, nil
		}
		if err != nil {
			return 0, 0, err
		}
		du, dv, p, q, _, err := parseRecord(rec)
		if err != nil {
			return 0, 0, err
		}
		if err := errIfBadChanPair(p, q); err != nil {
			return 0, 0, err
		}
		chans = max(max(chans, p+1), q+1)
		band = max(max(band, abs(du)), abs(dv))
	}
}

func parseRecord(s []string) (du, dv, p, q int, x float64, err error) {
	if len(s) != 5 {
		err = fmt.Errorf("wrong number of elements in line: %d (expect 5)", len(s))
		return
	}
	du_, err := strconv.ParseInt(s[0], 10, 32)
	if err != nil {
		return
	}
	dv_, err := strconv.ParseInt(s[1], 10, 32)
	if err != nil {
		return
	}
	p_, err := strconv.ParseInt(s[2], 10, 32)
	if err != nil {
		return
	}
	q_, err := strconv.ParseInt(s[3], 10, 32)
	if err != nil {
		return
	}
	x_, err := strconv.ParseFloat(s[4], 64)
	if err != nil {
		return
	}
	return int(du_), int(dv_), int(p_), int(q_), x_, nil
}
