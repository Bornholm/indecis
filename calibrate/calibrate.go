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

// eps borne les probabilités loin de 0 et 1, où le logit diverge.
const eps = 1e-6

// Clamp ramène p dans [eps, 1-eps].
func Clamp(p float64) float64 {
	if math.IsNaN(p) {
		return 0.5
	}
	return math.Min(math.Max(p, eps), 1-eps)
}

// Logit retourne log(p / (1-p)).
func Logit(p float64) float64 {
	p = Clamp(p)
	return math.Log(p / (1 - p))
}

// Sigmoid est l'inverse de Logit.
func Sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

// PriorShift retourne les log-odds à ajouter à la sortie d'un classifieur
// entraîné avec la proportion trainPrior de positifs pour qu'elle reflète la
// proportion deployPrior observée en production.
//
// C'est la règle de Bayes appliquée au seul terme qui change entre les deux
// distributions : la vraisemblance P(texte | classe) est supposée la même,
// seul P(classe) diffère.
func PriorShift(trainPrior, deployPrior float64) float64 {
	return Logit(deployPrior) - Logit(trainPrior)
}

// Evidence traduit le risque d'un autre détecteur en rapport de
// vraisemblance (log-LR).
//
// L'asymétrie est le point central. Un détecteur précis mais peu sensible,
// comme un jeu de règles, apporte une preuve forte quand il se déclenche et
// une preuve faible quand il se tait : son silence couvre aussi toutes les
// attaques qu'il ne connaît pas. D'où deux poids, un au-dessus de Neutral et
// un en dessous.
//
// Les poids inférieurs à 1 servent aussi de rétrécissement : deux détecteurs
// qui lisent le même texte ne sont pas indépendants, et une somme brute de
// log-LR compterait deux fois la même preuve.
type Evidence struct {
	// Neutral est le risque qui ne déplace pas la décision.
	Neutral float64
	// HitWeight pondère l'écart au-dessus de Neutral.
	HitWeight float64
	// MissWeight pondère l'écart en dessous de Neutral.
	MissWeight float64
	// Floor et Ceil bornent le risque avant le logit : un risque de 0 ne
	// doit pas valoir une preuve infinie d'innocuité.
	Floor, Ceil float64
}

// DefaultEvidence est un réglage de départ, à remplacer par un réglage
// ajusté sur un jeu de validation réel.
func DefaultEvidence() Evidence {
	return Evidence{Neutral: 0.3, HitWeight: 1, MissWeight: 0.2, Floor: 0.01, Ceil: 0.99}
}

// LLR retourne le log-rapport de vraisemblance apporté par risk.
func (e Evidence) LLR(risk float64) float64 {
	r := math.Min(math.Max(risk, e.Floor), e.Ceil)
	d := Logit(r) - Logit(e.Neutral)
	if d >= 0 {
		return e.HitWeight * d
	}
	return e.MissWeight * d
}
