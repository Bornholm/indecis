package main

import (
	"log/slog"
	"os"

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
	m, err := indecis.Load(dir)
	if err != nil {
		return nil, err
	}
	return newModelBackend(m, question)
}
