package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"math"
	"sync"

	proto "github.com/xolo-gateway/xolo/pkg/pluginsdk/proto"

	"github.com/bornholm/indecis/plugins/injection-detector/internal/detector"
	"github.com/bornholm/indecis/plugins/injection-detector/internal/messages"
)

const (
	PluginName    = "injection-detector"
	PluginVersion = "0.0.1"
)

// Plugin estime la probabilité qu'une requête tente de manipuler
// l'assistant, avec un encodeur, et la fusionne avec le risque de
// prompt-guard quand il est branché sur guard_risk.
//
// Il mesure et ne bloque pas : la décision revient au graphe (compare,
// block), comme pour prompt-guard. Sans modèle chargé, il laisse passer le
// risque amont et le dit sur model_ready.
type Plugin struct {
	proto.UnimplementedXoloPluginServer

	// Backend exécute le modèle ; nil tant qu'aucun modèle n'est chargé.
	Backend detector.Backend

	warnOnce sync.Once
}

func (p *Plugin) Describe(_ context.Context, _ *proto.DescribeRequest) (*proto.PluginDescriptor, error) {
	return &proto.PluginDescriptor{
		Name:         PluginName,
		Version:      PluginVersion,
		Description:  "Expérimental. Estime la probabilité d'une injection ou d'une réorientation de prompt avec un encodeur exécuté en Go pur, corrigée du prior de production. Se chaîne après prompt-guard : son risque, branché sur guard_risk, entre dans la décision comme une preuve bayésienne.",
		Capabilities: []proto.PluginDescriptor_Capability{proto.PluginDescriptor_PRE_REQUEST},
		InputPorts: []*proto.PortDescriptor{
			{Name: "request", PortType: "request", Required: true},
			{Name: "guard_risk", PortType: "number"},
		},
		OutputPorts: []*proto.PortDescriptor{
			{Name: "request", PortType: "request", Required: true},
			{Name: "risk", PortType: "number"},
			{Name: "probability", PortType: "number"},
			{Name: "suspicious", PortType: "boolean"},
			{Name: "segment", PortType: "string"},
			{Name: "category", PortType: "string"},
			{Name: "model_ready", PortType: "boolean"},
		},
		ConfigSchema: configSchemaJSON,
	}, nil
}

func (p *Plugin) PreRequest(ctx context.Context, in *proto.PreRequestInput) (*proto.PreRequestOutput, error) {
	cfg := parseConfig(in.GetCtx().GetConfigJson())
	guardRisk := parseGuardRisk(in.GetInputsJson())

	a, ready := p.assess(ctx, cfg, in.GetMessagesJson(), guardRisk)

	b, _ := json.Marshal(map[string]any{
		"risk":        round3(a.Risk),
		"probability": round3(a.Probability),
		"suspicious":  cfg.SuspiciousAbove > 0 && a.Risk >= cfg.SuspiciousAbove,
		"segment":     string(a.Segment),
		"category":    a.Category,
		"model_ready": ready,
	})
	return &proto.PreRequestOutput{Allowed: true, OutputsJson: string(b)}, nil
}

// assess juge la requête. Toute défaillance du modèle rend la main au risque
// amont plutôt que de bloquer ou d'inventer un score : un détecteur
// expérimental ne doit pas devenir un point de panne de la passerelle.
func (p *Plugin) assess(ctx context.Context, cfg Config, messagesJSON string, guardRisk *float64) (detector.Assessment, bool) {
	passthrough := detector.Assessment{}
	if guardRisk != nil {
		passthrough.Risk = *guardRisk
	}

	if p.Backend == nil {
		p.warnOnce.Do(func() {
			slog.Warn("injection-detector: aucun modèle chargé, le risque amont est transmis tel quel")
		})
		return passthrough, false
	}

	a, err := detector.Assess(ctx, p.Backend, segments(cfg, messagesJSON), cfg.Priors, guardRisk, cfg.Evidence)
	if err != nil {
		slog.Error("injection-detector: échec du modèle, risque amont transmis", slog.Any("error", err))
		return passthrough, false
	}
	return a, true
}

func segments(cfg Config, messagesJSON string) []detector.Segment {
	t := messages.Parse(messagesJSON)
	var segs []detector.Segment
	if t.Last != "" {
		segs = append(segs, detector.Segment{Kind: detector.User, Text: t.Last, Context: t.System})
	}
	if cfg.AnalyzeHistory {
		for _, s := range t.Earlier {
			segs = append(segs, detector.Segment{Kind: detector.History, Text: s, Context: t.System})
		}
	}
	if cfg.AnalyzeToolResults {
		for _, s := range t.Tools {
			segs = append(segs, detector.Segment{Kind: detector.Tool, Text: s, Context: t.System})
		}
	}
	return segs
}

// parseGuardRisk lit le port guard_risk. Un port non branché, absent ou hors
// de [0, 1] vaut nil : pas de preuve amont, plutôt qu'une preuve fausse.
func parseGuardRisk(inputsJSON string) *float64 {
	if inputsJSON == "" {
		return nil
	}
	var in struct {
		GuardRisk *float64 `json:"guard_risk"`
	}
	if err := json.Unmarshal([]byte(inputsJSON), &in); err != nil || in.GuardRisk == nil {
		return nil
	}
	r := *in.GuardRisk
	if math.IsNaN(r) || r < 0 || r > 1 {
		return nil
	}
	return &r
}

func round3(x float64) float64 { return math.Round(x*1000) / 1000 }
