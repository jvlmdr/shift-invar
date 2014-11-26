package whog

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path"
	"strconv"

	"github.com/jvlmdr/go-file/fileutil"
)

func SaveTotalExt(fname string, total *Total) error {
	switch path.Ext(fname) {
	case ".csv":
		return saveTotalCSV(fname, total)
	default:
		return fileutil.SaveExt(fname, total)
	}
}

func LoadTotalExt(fname string) (*Total, error) {
	switch path.Ext(fname) {
	case ".csv":
		return loadTotalCSV(fname)
	}
	var total *Total
	if err := fileutil.LoadExt(fname, &total); err != nil {
		return nil, err
	}
	return total, nil
}

func saveTotalCSV(fname string, total *Total) error {
	file, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer file.Close()
	return EncodeTotalCSV(file, total)
}

func loadTotalCSV(fname string) (*Total, error) {
	file, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	chans, band, err := DecodeTotalSizeCSV(file)
	if err != nil {
		return nil, err
	}
	if _, err := file.Seek(0, 0); err != nil {
		return nil, err
	}
	return DecodeTotalCSV(file, chans, band)
}

func EncodeTotalCSV(w io.Writer, total *Total) error {
	band, chans := total.CovarTotal.Bandwidth, total.CovarTotal.Channels
	ww := csv.NewWriter(w)
	defer ww.Flush()
	// Write number of images.
	rec := formatNumImages(total.Images)
	if err := ww.Write(rec); err != nil {
		return err
	}
	for p := 0; p < chans; p++ {
		rec := formatMeanElem(p, total.MeanTotal[p])
		if err := ww.Write(rec); err != nil {
			return err
		}
	}
	for du := -band; du <= band; du++ {
		for dv := -band; dv <= band; dv++ {
			rec := formatCountElem(du, dv, total.Count.At(du, dv))
			if err := ww.Write(rec); err != nil {
				return err
			}
		}
	}
	for du := -band; du <= band; du++ {
		for dv := -band; dv <= band; dv++ {
			for p := 0; p < chans; p++ {
				for q := 0; q < chans; q++ {
					rec := formatCovarElem(du, dv, p, q, total.CovarTotal.At(du, dv, p, q))
					if err := ww.Write(rec); err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}

func formatCovarElem(du, dv, p, q int, x float64) []string {
	return []string{
		"covar",
		strconv.FormatInt(int64(du), 10),
		strconv.FormatInt(int64(dv), 10),
		strconv.FormatInt(int64(p), 10),
		strconv.FormatInt(int64(q), 10),
		strconv.FormatFloat(x, 'g', -1, 64),
	}
}

func formatMeanElem(p int, x float64) []string {
	return []string{
		"mean",
		strconv.FormatInt(int64(p), 10),
		strconv.FormatFloat(x, 'g', -1, 64),
	}
}

func formatCountElem(du, dv int, x int64) []string {
	return []string{
		"count",
		strconv.FormatInt(int64(du), 10),
		strconv.FormatInt(int64(dv), 10),
		strconv.FormatInt(x, 10),
	}
}

func formatNumImages(x int) []string {
	return []string{"num-images", strconv.FormatInt(int64(x), 10)}
}

func DecodeTotalCSV(r io.Reader, chans, band int) (*Total, error) {
	var (
		cov   = NewCovar(chans, band)
		mean  = make([]float64, chans)
		count = NewCount(band)
		ims   int
	)
	rr := csv.NewReader(r)
	rr.FieldsPerRecord = -1
	for {
		rec, err := rr.Read()
		if err == io.EOF {
			return &Total{mean, cov, count, ims}, nil
		}
		if err != nil {
			return nil, err
		}
		if len(rec) == 0 {
			continue
		}

		var field string
		field, rec = rec[0], rec[1:]
		switch field {
		case "covar":
			du, dv, p, q, x, err := parseCovarElem(rec)
			if err != nil {
				return nil, err
			}
			cov.Set(du, dv, p, q, x)
		case "mean":
			p, x, err := parseMeanElem(rec)
			if err != nil {
				return nil, err
			}
			mean[p] = x
		case "count":
			du, dv, x, err := parseCountElem(rec)
			if err != nil {
				return nil, err
			}
			count.Set(du, dv, x)
		case "num-images":
			x, err := parseNumImages(rec)
			if err != nil {
				return nil, err
			}
			ims = x
		default:
			return nil, fmt.Errorf("unknown field: %s", field)
		}
	}
}

func DecodeTotalSizeCSV(r io.Reader) (chans, band int, err error) {
	rr := csv.NewReader(r)
	rr.FieldsPerRecord = -1
	for {
		rec, err := rr.Read()
		if err == io.EOF {
			return chans, band, nil
		}
		if err != nil {
			return 0, 0, err
		}
		if len(rec) == 0 {
			continue
		}

		var field string
		field, rec = rec[0], rec[1:]
		switch field {
		case "covar":
			du, dv, p, q, _, err := parseCovarElem(rec)
			if err != nil {
				return 0, 0, err
			}
			chans = max(max(chans, p+1), q+1)
			band = max(max(band, abs(du)), abs(dv))
		case "mean":
			p, _, err := parseMeanElem(rec)
			if err != nil {
				return 0, 0, err
			}
			chans = max(chans, p+1)
		case "count":
			du, dv, _, err := parseCountElem(rec)
			if err != nil {
				return 0, 0, err
			}
			band = max(max(band, abs(du)), abs(dv))
		case "num-images":
			_, err := parseNumImages(rec)
			if err != nil {
				return 0, 0, err
			}
		default:
			return 0, 0, fmt.Errorf("unknown field: %s", field)
		}
	}
}

func parseCovarElem(s []string) (du, dv, p, q int, x float64, err error) {
	err = errIfLenNotEq(5, len(s))
	if err != nil {
		return
	}
	du_, err := strconv.ParseInt(s[0], 10, 32)
	if err != nil {
		return
	}
	s = s[1:]
	dv_, err := strconv.ParseInt(s[0], 10, 32)
	if err != nil {
		return
	}
	s = s[1:]
	p_, err := strconv.ParseInt(s[0], 10, 32)
	if err != nil {
		return
	}
	s = s[1:]
	q_, err := strconv.ParseInt(s[0], 10, 32)
	if err != nil {
		return
	}
	s = s[1:]
	x_, err := strconv.ParseFloat(s[0], 64)
	if err != nil {
		return
	}
	return int(du_), int(dv_), int(p_), int(q_), x_, nil
}

func parseMeanElem(s []string) (p int, x float64, err error) {
	err = errIfLenNotEq(2, len(s))
	if err != nil {
		return
	}
	p_, err := strconv.ParseInt(s[0], 10, 32)
	if err != nil {
		return
	}
	s = s[1:]
	x_, err := strconv.ParseFloat(s[0], 64)
	if err != nil {
		return
	}
	return int(p_), x_, nil
}

func parseCountElem(s []string) (du, dv int, x int64, err error) {
	err = errIfLenNotEq(3, len(s))
	if err != nil {
		return
	}
	du_, err := strconv.ParseInt(s[0], 10, 32)
	if err != nil {
		return
	}
	s = s[1:]
	dv_, err := strconv.ParseInt(s[0], 10, 32)
	if err != nil {
		return
	}
	s = s[1:]
	x_, err := strconv.ParseInt(s[0], 10, 64)
	if err != nil {
		return
	}
	return int(du_), int(dv_), x_, nil
}

func parseNumImages(s []string) (x int, err error) {
	err = errIfLenNotEq(1, len(s))
	if err != nil {
		return
	}
	x_, err := strconv.ParseInt(s[0], 10, 64)
	if err != nil {
		return
	}
	return int(x_), nil
}

func errIfLenNotEq(want, got int) error {
	if want != got {
		return fmt.Errorf("wrong number of elements in line: %d (expect %d)", got, want)
	}
	return nil
}

func errIfBadChanPair(p, q int) error {
	if p < 0 || q < 0 {
		return fmt.Errorf("invalid channel index: (%d,%d)", p, q)
	}
	return nil
}
