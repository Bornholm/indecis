// Command indecis-serve expose des modèles indecis en HTTP avec l'API de
// décision de TypeSafe et d'OpenRouter, pour tester l'intégration d'un
// modèle sans rien changer au client que son URL.
//
//	indecis-serve -model ~/.cache/indecis/runs/policy-P5
//	indecis-serve -model injection=runs/policy-P5 -model courriels=runs/email-embed -addr :8080
//
//	curl -s localhost:8080/api/alpha/decisions -d '{
//	  "state": "Ignore previous instructions",
//	  "questions": {"injection": {"type": "noul", "instructions": "?"}}
//	}'
//
// Clients : genai (provider openrouter avec GENAI_…_BASE_URL=http://…/api/v1,
// ou typesafe avec http://…/v1), SDK TypeSafe, curl.
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/decision"
)

type modelFlags []string

func (m *modelFlags) String() string     { return strings.Join(*m, ",") }
func (m *modelFlags) Set(v string) error { *m = append(*m, v); return nil }

func main() {
	var models modelFlags
	flag.Var(&models, "model", "modèle à servir : répertoire, ou nom=répertoire (répétable ; le premier est le modèle par défaut)")
	addr := flag.String("addr", "127.0.0.1:8080", "adresse d'écoute")
	apiKey := flag.String("api-key", os.Getenv("INDECIS_API_KEY"), "clé exigée en Authorization: Bearer (vide : aucune)")
	threads := flag.Int("threads", 1, "cœurs par requête (0 : tous)")
	int8 := flag.Bool("int8", true, "couches en int8 si le processeur a AVX-VNNI")
	cache := flag.Int("embed-cache", 4096, "plongements gardés en cache (questions ouvertes)")
	flag.Parse()
	log := slog.New(slog.NewTextHandler(os.Stderr, nil))
	if len(models) == 0 {
		fmt.Fprintln(os.Stderr, "au moins un -model est requis")
		os.Exit(2)
	}
	opts := []indecis.Option{indecis.WithThreads(*threads), indecis.WithEmbedCache(*cache)}
	if *int8 {
		opts = append(opts, indecis.WithInt8())
	}
	s := &decision.Server{Models: map[string]*decision.Client{}, APIKey: *apiKey, Logger: log}
	for _, spec := range models {
		name, dir, ok := strings.Cut(spec, "=")
		if !ok {
			name, dir = filepath.Base(spec), spec
		}
		m, err := indecis.Load(dir, opts...)
		if err != nil {
			log.Error("chargement", "model", dir, "error", err)
			os.Exit(1)
		}
		s.Models[name] = decision.FromModel(name, m)
		if s.Default == "" {
			s.Default = name
		}
		log.Info("modèle chargé", "name", name, "dir", dir, "paired", m.Paired())
	}
	log.Info("à l'écoute", "addr", *addr, "endpoints", "POST /api/alpha/decisions, POST /v1/systemone, GET /api/alpha/models")
	if err := http.ListenAndServe(*addr, s.Handler()); err != nil {
		log.Error("serveur", "error", err)
		os.Exit(1)
	}
}
