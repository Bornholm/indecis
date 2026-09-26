package detector

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/bornholm/indecis/calibrate"
)

// fakeBackend renvoie un logit fixé par texte.
type fakeBackend struct {
	info   ModelInfo
	logits map[string]float64
	err    error
}

func (f fakeBackend) Info() ModelInfo { return f.info }

func (f fakeBackend) Score(_ context.Context, segs []Segment) ([]Score, error) {
	if f.err != nil {
		return nil, f.err
	}
	out := make([]Score, len(segs))
	for i, s := range segs {
		out[i] = Score{Logit: f.logits[s.Text], Category: "cat-" + s.Text}
	}
	return out, nil
}

func balanced(logits map[string]float64) fakeBackend {
	return fakeBackend{info: ModelInfo{TrainPrior: 0.5, Temperature: 1}, logits: logits}
}

func ptr(x float64) *float64 { return &x }

func near(a, b, tol float64) bool { return math.Abs(a-b) < tol }

func TestAssess_AppliesDeployPrior(t *testing.T) {
	b := balanced(map[string]float64{"x": calibrate.Logit(0.9)})
	a, err := Assess(context.Background(), b, []Segment{{Kind: User, Text: "x"}}, Priors{User: 0.02}, nil, calibrate.DefaultEvidence())
	if err != nil {
		t.Fatal(err)
	}
	if !near(a.Probability, 0.1552, 1e-3) || a.Risk != a.Probability {
		t.Fatalf("got %+v", a)
	}
}

func TestAssess_AppliesTemperature(t *testing.T) {
	b := fakeBackend{info: ModelInfo{TrainPrior: 0.5, Temperature: 2}, logits: map[string]float64{"x": 4}}
	a, _ := Assess(context.Background(), b, []Segment{{Kind: User, Text: "x"}}, Priors{User: 0.5}, nil, calibrate.DefaultEvidence())
	if !near(a.Probability, calibrate.Sigmoid(2), 1e-9) {
		t.Fatalf("got %v", a.Probability)
	}
}

// À logit égal, le résultat d'outil l'emporte : son prior est plus élevé.
func TestAssess_MaxIsTakenAfterPriorCorrection(t *testing.T) {
	b := balanced(map[string]float64{"u": 1, "t": 1})
	segs := []Segment{{Kind: User, Text: "u"}, {Kind: Tool, Text: "t"}}
	a, _ := Assess(context.Background(), b, segs, Priors{User: 0.01, Tool: 0.1}, nil, calibrate.DefaultEvidence())
	if a.Segment != Tool || a.Category != "cat-t" {
		t.Fatalf("got %+v", a)
	}
}

func TestAssess_GuardHitRaisesRisk(t *testing.T) {
	b := balanced(map[string]float64{"x": 0})
	segs := []Segment{{Kind: User, Text: "x"}}
	alone, _ := Assess(context.Background(), b, segs, Priors{User: 0.05}, nil, calibrate.DefaultEvidence())
	fused, _ := Assess(context.Background(), b, segs, Priors{User: 0.05}, ptr(0.88), calibrate.DefaultEvidence())
	if fused.Risk <= alone.Risk {
		t.Fatalf("fused %v <= alone %v", fused.Risk, alone.Risk)
	}
	if fused.Probability != alone.Probability {
		t.Fatal("la fusion ne doit pas modifier la probabilité du modèle seul")
	}
}

// Un prompt-guard silencieux ne doit pas effacer une attaque que le modèle
// voit nettement : c'est tout l'intérêt de chaîner les deux.
func TestAssess_GuardSilenceDoesNotEraseModel(t *testing.T) {
	b := balanced(map[string]float64{"x": 8})
	a, _ := Assess(context.Background(), b, []Segment{{Kind: User, Text: "x"}}, Priors{User: 0.02}, ptr(0), calibrate.DefaultEvidence())
	if a.Risk < 0.8 {
		t.Fatalf("risk %v: le silence de prompt-guard a écrasé le modèle", a.Risk)
	}
}

func TestAssess_NoSegmentPassesGuardThrough(t *testing.T) {
	a, _ := Assess(context.Background(), balanced(nil), nil, DefaultPriors(), ptr(0.42), calibrate.DefaultEvidence())
	if a.Risk != 0.42 || a.Probability != 0 {
		t.Fatalf("got %+v", a)
	}
}

func TestAssess_BackendErrorIsReturned(t *testing.T) {
	b := fakeBackend{err: errors.New("boom")}
	if _, err := Assess(context.Background(), b, []Segment{{Kind: User, Text: "x"}}, DefaultPriors(), nil, calibrate.DefaultEvidence()); err == nil {
		t.Fatal("expected error")
	}
}
