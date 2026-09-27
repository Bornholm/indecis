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
// -model accepte aussi un backbone brut (config.json, model.safetensors,
// tokenizer.json, sans indecis.json), bekko par exemple : il ne répond
// alors qu'aux questions ouvertes, sans entraînement.
//
// Clients : genai (provider openrouter avec GENAI_…_BASE_URL=http://…/api/v1,
// ou typesafe avec http://…/v1), SDK TypeSafe, curl.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
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
	threads := flag.Int("threads", 0, "cœurs au plus par requête (0 : tous) ; un texte de moins de 1024 tokens en utilise toujours un seul")
	int8 := flag.Bool("int8", true, "couches en int8 si le processeur a AVX-VNNI")
	cache := flag.Int("embed-cache", 4096, "plongements d'options gardés en cache (questions ouvertes)")
	maxConcurrent := flag.Int("max-concurrent", runtime.GOMAXPROCS(0), "décisions en cours au plus, les autres attendent (0 : pas de borne)")
	batching := flag.Bool("batching", false, "regrouper les calculs des requêtes simultanées (gain de quelques % sur des requêtes courtes, plus de mémoire ; monter aussi -max-concurrent)")
	maxLen := flag.Int("max-len", 0, "tokens lus au plus par texte (0 : la valeur du modèle, 256 en général) ; coût quadratique au-delà de 1024")
	memLimit := flag.Int("memory-limit", 0, "limite souple de mémoire du tas, en Mio (0 : aucune) ; le ramasse-miettes travaille davantage à l'approche")
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
	if *maxLen > 0 {
		opts = append(opts, indecis.WithMaxLen(*maxLen))
	}
	if *batching {
		opts = append(opts, indecis.WithBatching(0, 0))
	}
	if *memLimit > 0 {
		debug.SetMemoryLimit(int64(*memLimit) << 20)
	}
	s := &decision.Server{Models: map[string]*decision.Client{}, APIKey: *apiKey, Logger: log, MaxConcurrent: *maxConcurrent}
	for _, spec := range models {
		name, dir, ok := strings.Cut(spec, "=")
		if !ok {
			name, dir = filepath.Base(spec), spec
		}
		m, err := load(dir, opts)
		if err != nil {
			log.Error("chargement", "model", dir, "error", err)
			os.Exit(1)
		}
		// Un appel à vide prépare les poids pour l'inférence (en int8, les
		// matrices float32 sont alors libérées), puis la mémoire temporaire
		// du chargement est rendue au système avant le modèle suivant : le
		// pic de démarrage ne dépend plus du nombre de modèles.
		if _, err := m.Embed(context.Background(), "warm-up"); err != nil {
			log.Error("préchauffage", "model", dir, "error", err)
			os.Exit(1)
		}
		debug.FreeOSMemory()
		s.Models[name] = decision.FromModel(name, m)
		if s.Default == "" {
			s.Default = name
		}
		log.Info("modèle chargé", "name", name, "dir", dir, "paired", m.Paired(), "mémoire", memory())
	}
	log.Info("à l'écoute", "addr", *addr, "endpoints", "POST /api/alpha/decisions, POST /v1/systemone, GET /api/alpha/models")
	if err := http.ListenAndServe(*addr, s.Handler()); err != nil {
		log.Error("serveur", "error", err)
		os.Exit(1)
	}
}

// load lit un modèle indecis, ou prépare un backbone brut pour les seules
// questions ouvertes.
func load(dir string, opts []indecis.Option) (*indecis.Model, error) {
	if _, err := os.Stat(filepath.Join(dir, "indecis.json")); err == nil {
		return indecis.Load(dir, opts...)
	}
	// La question apprise est un simple point d'ancrage : sa tête n'est pas
	// entraînée, elle ne doit pas être interrogée (voir le tutoriel).
	return indecis.New(dir, indecis.Schema{indecis.NewNoul("match", "")}, 1, opts...)
}

// memory résume la mémoire du processus (Linux), pour les journaux.
func memory() string {
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return "?"
	}
	var parts []string
	for _, l := range strings.Split(string(b), "\n") {
		for _, k := range []string{"RssAnon:", "RssFile:", "VmHWM:"} {
			if strings.HasPrefix(l, k) {
				parts = append(parts, strings.Join(strings.Fields(l), " "))
			}
		}
	}
	return strings.Join(parts, ", ")
}
