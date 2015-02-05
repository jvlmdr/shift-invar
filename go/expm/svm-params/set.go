package main

import "reflect"

func enumerateParams(set *ParamSet) []Param {
	fields := []string{"Lambda", "Gamma", "Epochs", "NegFrac", "Overlap", "Size", "Feat"}
	base := Param{}
	return enumerateParamsFrom(set, []Param{base}, fields)
}

func enumerateParamsFrom(set *ParamSet, in []Param, fields []string) []Param {
	if len(fields) == 0 {
		return in
	}
	name := fields[0]
	setfield := reflect.ValueOf(set).Elem().FieldByName(name)
	var out []Param
	for i := 0; i < setfield.Len(); i++ {
		for _, x := range in {
			// Modify the field of x.
			xfield := reflect.ValueOf(&x).Elem().FieldByName(name)
			xfield.Set(setfield.Index(i))
			out = append(out, x)
		}
	}
	return enumerateParamsFrom(set, out, fields[1:])
}
