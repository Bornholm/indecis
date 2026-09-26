package main

import (
	"context"
	"log/slog"
	"os"
	"runtime/debug"
	"strconv"

	"github.com/xolo-gateway/xolo/pkg/pluginsdk"

	"github.com/bornholm/indecis"
)

// Le modèle est désigné par INDECIS_MODEL_DIR (répertoire écrit par
// indecis.Model.Save) et la question par INDECIS_QUESTION (« injection » par
// défaut). Sans modèle, le plugin reste utilisable et le dit sur
// model_ready.
func main() {
	p := &Plugin{}
	if dir := os.Getenv("INDECIS_MODEL_DIR"); dir != "" {
		question := os.Getenv("INDECIS_QUESTION")
		if question == "" {
			question = "injection"
		}
		b, err := loadBackend(dir, question)
		if err != nil {
			slog.Error("injection-detector : modèle non chargé", slog.String("dir", dir), slog.Any("error", err))
		} else {
			p.Backend = b
			slog.Info("injection-detector : modèle chargé", slog.String("dir", dir), slog.String("version", b.Info().Version))
		}
	}
	pluginsdk.Serve(p)
}

func loadBackend(dir, question string) (*modelBackend, error) {
	// Un cœur par requête : chaque requête isolée est plus rapide ainsi, et
	// les requêtes concurrentes occupent les autres cœurs.
	threads := 1
	if v, err := strconv.Atoi(os.Getenv("INDECIS_THREADS")); err == nil && v >= 0 {
		threads = v
	}
	opts := []indecis.Option{indecis.WithThreads(threads)}
	// Couches en int8 quand le processeur a AVX-VNNI : deux fois plus
	// rapide, décisions inchangées sur la référence. INDECIS_INT8=0 revient
	// au float32.
	if os.Getenv("INDECIS_INT8") != "0" {
		opts = append(opts, indecis.WithInt8())
	}
	m, err := indecis.Load(dir, opts...)
	if err != nil {
		return nil, err
	}
	// Une requête à vide prépare les poids pour l'inférence (la première
	// requête réelle n'en paie pas le coût), puis la mémoire temporaire du
	// chargement est rendue au système.
	if _, err := m.Decide(context.Background(), "warm-up"); err != nil {
		return nil, err
	}
	debug.FreeOSMemory()
	return newModelBackend(m, question)
}
