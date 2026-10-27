package main

import "testing"

func TestLongS(t *testing.T) {
	if got := longS("les sciences, aussi"); got != "les ſciences, auſſi" {
		t.Fatalf("%q", got)
	}
}
