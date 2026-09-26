package modernbert

import (
	"math"
	"testing"
)

func TestErf32(t *testing.T) {
	var worst float64
	for x := -6.0; x <= 6; x += 1e-4 {
		d := math.Abs(float64(erf32(float32(x))) - math.Erf(x))
		worst = max(worst, d)
	}
	if worst > 5e-7 {
		t.Fatalf("erreur maximale %.3g", worst)
	}
}

func TestExp32(t *testing.T) {
	var worst float64
	for x := -87.0; x <= 88; x += 1e-3 {
		want := math.Exp(float64(float32(x))) // erreur de l'approximation seule
		d := math.Abs(float64(exp32(float32(x)))-want) / want
		worst = max(worst, d)
	}
	if worst > 5e-7 {
		t.Fatalf("erreur relative maximale %.3g", worst)
	}
	if exp32(float32(math.Inf(-1))) != 0 || exp32(-100) != 0 {
		t.Fatal("exp32(-Inf) doit valoir 0")
	}
}
