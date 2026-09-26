package main

import (
	"encoding/json"

	"github.com/bornholm/indecis/calibrate"
	"github.com/bornholm/indecis/plugins/injection-detector/internal/detector"
)

// Config est la configuration d'un nœud injection-detector.
type Config struct {
	AnalyzeToolResults bool
	AnalyzeHistory     bool
	SuspiciousAbove    float64
	Priors             detector.Priors
	Evidence           calibrate.Evidence
}

func defaultConfig() Config {
	return Config{
		AnalyzeToolResults: true,
		AnalyzeHistory:     false,
		SuspiciousAbove:    0.5,
		Priors:             detector.DefaultPriors(),
		Evidence:           calibrate.DefaultEvidence(),
	}
}

// parseConfig applique les champs valides sur la configuration par défaut et
// ignore les autres : une valeur hors bornes ne doit ni faire échouer la
// requête ni changer silencieusement de sens.
func parseConfig(raw string) Config {
	cfg := defaultConfig()
	if raw == "" || raw == "{}" {
		return cfg
	}
	var aux struct {
		AnalyzeToolResults *bool    `json:"analyze_tool_results"`
		AnalyzeHistory     *bool    `json:"analyze_history"`
		SuspiciousAbove    *float64 `json:"suspicious_above"`
		PriorUser          *float64 `json:"prior_user"`
		PriorHistory       *float64 `json:"prior_history"`
		PriorTool          *float64 `json:"prior_tool"`
		GuardNeutral       *float64 `json:"guard_neutral"`
		GuardHitWeight     *float64 `json:"guard_hit_weight"`
		GuardMissWeight    *float64 `json:"guard_miss_weight"`
	}
	if err := json.Unmarshal([]byte(raw), &aux); err != nil {
		return cfg
	}
	if aux.AnalyzeToolResults != nil {
		cfg.AnalyzeToolResults = *aux.AnalyzeToolResults
	}
	if aux.AnalyzeHistory != nil {
		cfg.AnalyzeHistory = *aux.AnalyzeHistory
	}
	if v := aux.SuspiciousAbove; v != nil && *v >= 0 && *v <= 1 {
		cfg.SuspiciousAbove = *v
	}
	setPrior := func(k detector.Kind, v *float64) {
		if v != nil && *v > 0 && *v < 1 {
			cfg.Priors[k] = *v
		}
	}
	setPrior(detector.User, aux.PriorUser)
	setPrior(detector.History, aux.PriorHistory)
	setPrior(detector.Tool, aux.PriorTool)
	if v := aux.GuardNeutral; v != nil && *v > 0 && *v < 1 {
		cfg.Evidence.Neutral = *v
	}
	if v := aux.GuardHitWeight; v != nil && *v >= 0 && *v <= 3 {
		cfg.Evidence.HitWeight = *v
	}
	if v := aux.GuardMissWeight; v != nil && *v >= 0 && *v <= 3 {
		cfg.Evidence.MissWeight = *v
	}
	return cfg
}

const configSchemaJSON = `{
  "type": "object",
  "properties": {
    "analyze_tool_results": {
      "type": "boolean",
      "title": "Analyser les résultats d'outils",
      "description": "Là où arrivent les injections indirectes : pages web, documents récupérés.",
      "default": true
    },
    "analyze_history": {
      "type": "boolean",
      "title": "Analyser les tours utilisateur précédents",
      "default": false
    },
    "suspicious_above": {
      "type": "number",
      "title": "Seuil du port suspicious",
      "minimum": 0, "maximum": 1, "default": 0.5
    },
    "prior_user": {
      "type": "number",
      "title": "Proportion attendue d'attaques dans les messages utilisateur",
      "description": "Prior de production. Le modèle est entraîné sur un corpus équilibré ; ce prior ramène sa probabilité à la réalité du trafic.",
      "exclusiveMinimum": 0, "exclusiveMaximum": 1, "default": 0.02
    },
    "prior_history": {
      "type": "number",
      "title": "Proportion attendue d'attaques dans l'historique",
      "exclusiveMinimum": 0, "exclusiveMaximum": 1, "default": 0.02
    },
    "prior_tool": {
      "type": "number",
      "title": "Proportion attendue d'attaques dans les résultats d'outils",
      "exclusiveMinimum": 0, "exclusiveMaximum": 1, "default": 0.05
    },
    "guard_neutral": {
      "type": "number",
      "title": "Risque amont neutre",
      "description": "Valeur de guard_risk qui ne déplace pas la décision.",
      "exclusiveMinimum": 0, "exclusiveMaximum": 1, "default": 0.3
    },
    "guard_hit_weight": {
      "type": "number",
      "title": "Poids d'un risque amont élevé",
      "minimum": 0, "maximum": 3, "default": 1
    },
    "guard_miss_weight": {
      "type": "number",
      "title": "Poids d'un risque amont faible",
      "description": "Plus faible que le précédent : le silence d'un détecteur à règles couvre aussi les attaques qu'il ne connaît pas.",
      "minimum": 0, "maximum": 3, "default": 0.2
    }
  }
}`
