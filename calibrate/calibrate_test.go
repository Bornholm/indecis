package calibrate

import (
	"math"
	"testing"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestLogitSigmoidRoundTrip(t *testing.T) {
	for _, p := range []float64{0.001, 0.2, 0.5, 0.73, 0.999} {
		if got := Sigmoid(Logit(p)); !near(got, p) {
			t.Errorf("Sigmoid(Logit(%v)) = %v", p, got)
		}
	}
}

func TestLogitStaysFinite(t *testing.T) {
	for _, p := range []float64{0, 1, -3, 7, math.NaN()} {
		if z := Logit(p); math.IsInf(z, 0) || math.IsNaN(z) {
			t.Errorf("Logit(%v) = %v", p, z)
		}
	}
}

func TestPriorShift_SamePriorIsNeutral(t *testing.T) {
	if s := PriorShift(0.5, 0.5); !near(s, 0) {
		t.Fatalf("got %v", s)
	}
}

// Un modèle équilibré à 0,5 sur un trafic où 1 % des requêtes sont des
// attaques ne dit rien de plus que le prior : la probabilité corrigée doit
// retomber sur 1 %.
func TestPriorShift_UninformativeOutputFallsBackToDeployPrior(t *testing.T) {
	p := Sigmoid(Logit(0.5) + PriorShift(0.5, 0.01))
	if !near(p, 0.01) {
		t.Fatalf("got %v, want 0.01", p)
	}
}

// Cas chiffré : 0,9 à l'entraînement équilibré, prior de production 2 %.
// Odds 9 × (0,02/0,98) = 0,1837, soit p = 0,1552.
func TestPriorShift_KnownValue(t *testing.T) {
	p := Sigmoid(Logit(0.9) + PriorShift(0.5, 0.02))
	if math.Abs(p-0.15517) > 1e-4 {
		t.Fatalf("got %v, want ≈0.1552", p)
	}
}

func TestEvidence_NeutralRiskAddsNothing(t *testing.T) {
	e := DefaultEvidence()
	if l := e.LLR(e.Neutral); !near(l, 0) {
		t.Fatalf("got %v", l)
	}
}

// Le silence d'un détecteur peu sensible pèse moins que son déclenchement,
// à distance égale en log-odds du point neutre.
func TestEvidence_IsAsymmetric(t *testing.T) {
	e := Evidence{Neutral: 0.5, HitWeight: 1, MissWeight: 0.2, Floor: 0.01, Ceil: 0.99}
	hit, miss := e.LLR(0.9), e.LLR(0.1)
	if !near(hit, -miss*5) {
		t.Fatalf("hit %v, miss %v: want hit = -5 × miss", hit, miss)
	}
}

func TestEvidence_IsMonotonic(t *testing.T) {
	e := DefaultEvidence()
	prev := math.Inf(-1)
	for r := 0.0; r <= 1.0; r += 0.05 {
		l := e.LLR(r)
		if l < prev {
			t.Fatalf("LLR(%v) = %v < %v", r, l, prev)
		}
		prev = l
	}
}

func TestEvidence_ZeroRiskIsBounded(t *testing.T) {
	e := DefaultEvidence()
	if l := e.LLR(0); l < -1 {
		t.Fatalf("LLR(0) = %v: un silence ne doit pas écraser le modèle", l)
	}
}
