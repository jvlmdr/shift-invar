package whog

import "fmt"

func errIfNumChansNotEq(m, n int) error {
	if m != n {
		return fmt.Errorf("channels differ: %d, %d", m, n)
	}
	return nil
}
