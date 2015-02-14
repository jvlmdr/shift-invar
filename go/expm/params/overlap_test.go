package main

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestOverlapMessage_UnmarshalJSON(t *testing.T) {
	orig := OverlapMessage{"inter-over-union", &InterOverUnion{0.3}}
	data, err := json.Marshal(orig)
	if err != nil {
		t.Fatal(err)
	}
	var result OverlapMessage
	if err := json.Unmarshal(data, &result); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(orig, result) {
		t.Fatalf("want %s, got %s", orig, result)
	}
}
