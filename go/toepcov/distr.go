package whog

//	"archive/tar"
//	"encoding/json"
//	"io"
//	"io/ioutil"

type Distr struct {
	Mean  []float64
	Covar *Covar
}

func (p *Distr) Center() {
	for k := 0; k < p.Covar.Channels; k++ {
		for w := 0; w < p.Covar.Channels; w++ {
			cross := p.Mean[k] * p.Mean[w]
			for i := -p.Covar.Bandwidth; i <= p.Covar.Bandwidth; i++ {
				for j := -p.Covar.Bandwidth; j <= p.Covar.Bandwidth; j++ {
					p.Covar.Set(i, j, k, w, p.Covar.At(i, j, k, w)-cross)
				}
			}
		}
	}
}

//	// Saves distribution in binary format.
//	// Creates tar file containing mean.json and covar.tar.
//	func (p *Distr) WriteTo(w io.Writer) (int64, error) {
//		return p.writeTo(w, false)
//	}
//
//	// Reads from a tar containing mean.json and covar.tar.
//	func (p *Distr) ReadFrom(r io.Reader) (int64, error) {
//		// Count reads to return number of bytes.
//		rc := countReads(r)
//		r = rc
//		tr := tar.NewReader(r)
//
//		for {
//			hdr, err := tr.Next()
//			if err == io.EOF {
//				break
//			}
//			if err != nil {
//				return 0, err
//			}
//
//			switch hdr.Name {
//			default:
//			case "mean.json":
//				if err := json.NewDecoder(tr).Decode(&p.Mean); err != nil {
//					return 0, err
//				}
//			case "covar.tar":
//				if _, err := p.Covar.ReadFrom(tr); err != nil {
//					return 0, err
//				}
//			}
//		}
//
//		return rc.N, nil
//	}
//
//	// Saves distribution in human-readable format.
//	// Creates tar file containing mean.json and covar.json.
//	func (p *Distr) WriteHumanTo(w io.Writer) (int64, error) {
//		return p.writeTo(w, true)
//	}
//
//	// Writes distribution to tar file.
//	// Covariance can be human-readable or binary.
//	func (p *Distr) writeTo(w io.Writer, human bool) (int64, error) {
//		wc := countWrites(w)
//		w = wc
//		tw := tar.NewWriter(w)
//		// Ensure that tar writer is closed before taking count.
//		err := func() error {
//			defer tw.Close()
//
//			// Count number of bytes.
//			d := countWrites(ioutil.Discard)
//			enc := json.NewEncoder(d)
//			if err := enc.Encode(p.Mean); err != nil {
//				return err
//			}
//			// Write tar header.
//			if err := tw.WriteHeader(newTarHeader("mean.json", d.N)); err != nil {
//				return err
//			}
//			// Encode to file.
//			enc = json.NewEncoder(tw)
//			if err := enc.Encode(p.Mean); err != nil {
//				return err
//			}
//
//			if human {
//				// Count number of bytes.
//				n, err := p.Covar.WriteHumanTo(ioutil.Discard)
//				if err != nil {
//					return err
//				}
//				// Write tar header.
//				if err := tw.WriteHeader(newTarHeader("covar.json", n)); err != nil {
//					return err
//				}
//				// Write JSON to file.
//				if _, err = p.Covar.WriteHumanTo(tw); err != nil {
//					return err
//				}
//			} else {
//				// Count number of bytes.
//				n, err := p.Covar.WriteTo(ioutil.Discard)
//				if err != nil {
//					return err
//				}
//				// Write tar header.
//				if err := tw.WriteHeader(newTarHeader("covar.tar", n)); err != nil {
//					return err
//				}
//				// Write covar tar to file.
//				if _, err = p.Covar.WriteTo(tw); err != nil {
//					return err
//				}
//			}
//			return nil
//		}()
//		if err != nil {
//			return 0, err
//		}
//		return wc.N, nil
//	}
