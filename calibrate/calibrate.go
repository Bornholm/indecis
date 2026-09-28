// Package calibrate combines probabilities in log-odds.
//
// It has two operations only, the two that a detector trained on a balanced
// corpus needs before it meets production:
//
//   - shifting the prior: the model saw as many attacks as honest texts, the
//     real traffic does not. Without the shift, its probability overestimates
//     the risk by a factor that can exceed 20;
//   - adding independent evidence: the risk from another detector comes in
//     as a likelihood ratio, not as an average.
package calibrate

import "math"

// eps keeps probabilities away from 0 and 1, where the logit diverges.
const eps = 1e-6

// Clamp brings p into [eps, 1-eps].
func Clamp(p float64) float64 {
	if math.IsNaN(p) {
		return 0.5
	}
	return math.Min(math.Max(p, eps), 1-eps)
}

// Logit returns log(p / (1-p)).
func Logit(p float64) float64 {
	p = Clamp(p)
	return math.Log(p / (1 - p))
}

// Sigmoid is the inverse of Logit.
func Sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

// PriorShift returns the log-odds to add to the output of a classifier
// trained with the trainPrior proportion of positives so that it
// reflects the deployPrior proportion observed in production.
//
// This is Bayes' rule applied to the only term that changes between the
// two distributions: the likelihood P(text | class) is assumed the
// same, only P(class) differs.
func PriorShift(trainPrior, deployPrior float64) float64 {
	return Logit(deployPrior) - Logit(trainPrior)
}

// Evidence translates the risk from another detector into a
// likelihood ratio (log-LR).
//
// The asymmetry is the central point. A precise but low-sensitivity
// detector, like a rule set, provides strong evidence when it fires and
// weak evidence when it stays silent: its silence also covers all the
// attacks it does not know about. Hence two weights, one above Neutral
// and one below.
//
// Weights below 1 also serve as shrinkage: two detectors reading the
// same text are not independent, and a raw sum of log-LR would count
// the same evidence twice.
type Evidence struct {
	// Neutral is the risk that does not move the decision.
	Neutral float64
	// HitWeight weighs the gap above Neutral.
	HitWeight float64
	// MissWeight weighs the gap below Neutral.
	MissWeight float64
	// Floor and Ceil bound the risk before the logit: a risk of 0 must
	// not amount to infinite proof of innocence.
	Floor, Ceil float64
}

// DefaultEvidence is a starting setting, to be replaced by a setting
// tuned on a real validation set.
func DefaultEvidence() Evidence {
	return Evidence{Neutral: 0.3, HitWeight: 1, MissWeight: 0.2, Floor: 0.01, Ceil: 0.99}
}

// LLR returns the log-likelihood ratio contributed by risk.
func (e Evidence) LLR(risk float64) float64 {
	r := math.Min(math.Max(risk, e.Floor), e.Ceil)
	d := Logit(r) - Logit(e.Neutral)
	if d >= 0 {
		return e.HitWeight * d
	}
	return e.MissWeight * d
}
