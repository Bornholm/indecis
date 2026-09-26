// Package detector transforme les scores bruts d'un modèle en décision de
// production : calibration, correction du prior par provenance, agrégation
// sur les segments, fusion avec un détecteur amont.
//
// Le modèle lui-même est derrière Backend. Ce package ne sait pas s'il tourne
// en Go pur, sur un NPU ou dans un test ; il sait seulement ce qu'un logit
// veut dire.
package detector

import (
	"context"
	"math"

	"github.com/bornholm/indecis/calibrate"
)

// Kind est la provenance d'un segment.
type Kind string

const (
	User    Kind = "user"
	History Kind = "history"
	Tool    Kind = "tool"
)

// Segment est un texte à juger, avec sa provenance.
type Segment struct {
	Kind Kind
	Text string
	// Context est le prompt système de la requête : un modèle en paires
	// juge le texte par rapport au périmètre qu'il fixe.
	Context string
}

// Score est la sortie brute du modèle pour un segment.
type Score struct {
	// Logit de la classe « injection », avant température et correction
	// du prior.
	Logit float64
	// Category est la catégorie la plus probable selon la tête de choix du
	// modèle, vide si le modèle n'en a pas.
	Category string
}

// ModelInfo décrit ce qu'il faut savoir d'un modèle pour interpréter ses
// logits. Ces valeurs voyagent avec les poids : elles ont été mesurées à
// l'entraînement, elles ne se configurent pas.
type ModelInfo struct {
	Version string
	// TrainPrior est la proportion d'attaques dans le corpus d'entraînement.
	TrainPrior float64
	// Temperature divise le logit ; ajustée sur le jeu de validation.
	Temperature float64
}

// Backend exécute le modèle.
type Backend interface {
	Info() ModelInfo
	// Score retourne un Score par segment, dans le même ordre.
	Score(ctx context.Context, segments []Segment) ([]Score, error)
}

// Priors donne la proportion d'attaques attendue en production pour chaque
// provenance. Une page web qui s'adresse à l'assistant est plus suspecte
// qu'un utilisateur qui le fait : c'est un prior, pas une pondération ad hoc.
type Priors map[Kind]float64

// DefaultPriors sont des valeurs de départ, à remplacer par celles mesurées
// sur le trafic de l'installation.
func DefaultPriors() Priors {
	return Priors{User: 0.02, History: 0.02, Tool: 0.05}
}

// Assessment est le verdict sur une requête.
type Assessment struct {
	// Probability est la probabilité a posteriori d'injection selon le seul
	// modèle, sur le segment le plus suspect.
	Probability float64
	// Risk ajoute à Probability la preuve du détecteur amont, s'il y en a
	// une. Sans preuve amont, Risk vaut Probability.
	Risk float64
	// Segment est la provenance du segment le plus suspect.
	Segment Kind
	// Category est la catégorie du segment le plus suspect.
	Category string
}

// Assess juge les segments et fusionne le résultat avec guardRisk quand il
// est fourni (nil sinon).
//
// Le maximum se prend sur les log-odds a posteriori, après correction du
// prior de chaque segment : un résultat d'outil et un message utilisateur de
// même logit ne pèsent pas pareil.
func Assess(ctx context.Context, b Backend, segments []Segment, priors Priors, guardRisk *float64, ev calibrate.Evidence) (Assessment, error) {
	var a Assessment
	if len(segments) == 0 {
		a.Risk = fuse(math.Inf(-1), guardRisk, ev)
		return a, nil
	}

	scores, err := b.Score(ctx, segments)
	if err != nil {
		return a, err
	}

	info := b.Info()
	temp := info.Temperature
	if temp <= 0 {
		temp = 1
	}

	best := math.Inf(-1)
	for i, s := range scores {
		if i >= len(segments) {
			break
		}
		prior, ok := priors[segments[i].Kind]
		if !ok {
			prior = info.TrainPrior
		}
		z := s.Logit/temp + calibrate.PriorShift(info.TrainPrior, prior)
		if z > best {
			best = z
			a.Segment = segments[i].Kind
			a.Category = s.Category
		}
	}

	a.Probability = calibrate.Sigmoid(best)
	a.Risk = fuse(best, guardRisk, ev)
	return a, nil
}

// fuse ajoute la preuve amont aux log-odds du modèle. Sans modèle (z = -∞),
// le risque amont passe tel quel : on n'invente pas une probabilité.
func fuse(z float64, guardRisk *float64, ev calibrate.Evidence) float64 {
	if guardRisk == nil {
		if math.IsInf(z, -1) {
			return 0
		}
		return calibrate.Sigmoid(z)
	}
	if math.IsInf(z, -1) {
		return math.Min(math.Max(*guardRisk, 0), 1)
	}
	return calibrate.Sigmoid(z + ev.LLR(*guardRisk))
}
