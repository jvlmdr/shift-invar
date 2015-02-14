package main

import "reflect"

// set is a struct with each field in fields being a slice.
// curr is a slice of structs having the fields.
func enumerate(set, curr interface{}, fields []string) interface{} {
	if len(fields) == 0 {
		return curr
	}
	name := fields[0]
	setfield := reflect.ValueOf(set).FieldByName(name)
	out := reflect.Zero(reflect.TypeOf(curr))
	for i := 0; i < setfield.Len(); i++ {
		for j := 0; j < reflect.ValueOf(curr).Len(); j++ {
			// Copy the element.
			x := reflect.ValueOf(curr).Index(j)
			// Modify the field of x.
			x.FieldByName(name).Set(setfield.Index(i))
			out = reflect.Append(out, x)
		}
	}
	return enumerate(set, out.Interface(), fields[1:])
}
