// Command prunevocab réduit le vocabulaire d'un modèle aux tokens qu'utilise
// un corpus représentatif, et mesure l'effet sur des textes tenus à l'écart.
//
//	go run ./tools/prunevocab -model runs/policy-P5 -out runs/policy-P5-pruned \
//	    -corpus 'datasets/real/*.jsonl,teacher/*_policy.jsonl' \
//	    -eval examples/prompt-injection/eval/policy_eval.jsonl
//
// Le corpus doit ressembler aux textes que le modèle jugera (langues,
// domaines) : les données d'entraînement du modèle conviennent. Les jeux
// d'évaluation ne doivent pas en faire partie, sinon la mesure est
// faussée.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

func main() {
	modelDir := flag.String("model", "", "modèle à élaguer")
	out := flag.String("out", "", "répertoire du modèle élagué")
	corpusGlobs := flag.String("corpus", "", "fichiers JSONL du corpus (motifs, séparés par des virgules)")
	evalGlobs := flag.String("eval", "", "fichiers JSONL tenus à l'écart, pour mesurer l'effet (facultatif)")
	minCount := flag.Int("min-count", 1, "occurrences minimales d'un token dans le corpus")
	flag.Parse()
	if *modelDir == "" || *out == "" || *corpusGlobs == "" {
		log.Fatal("-model, -out et -corpus sont obligatoires")
	}
	ctx := context.Background()
	corpus := read(*corpusGlobs)
	log.Printf("corpus : %d textes", len(corpus))

	m, err := indecis.Load(*modelDir)
	if err != nil {
		log.Fatal(err)
	}
	var evalEx []dataset.Example
	if *evalGlobs != "" {
		evalEx = read(*evalGlobs)
	}
	var beforeTokens [][]int32
	if len(evalEx) > 0 {
		report(ctx, "avant", m, evalEx)
		beforeTokens = tokens(m, evalEx)
	}

	inputs := make([]indecis.Input, len(corpus))
	for i, e := range corpus {
		inputs[i] = indecis.Input{Context: e.Context, Text: e.Text}
	}
	st, err := m.PruneVocabulary(inputs, *minCount)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("vocabulaire : %d → %d tokens (%.1f %%)", st.Before, st.After, 100*float64(st.After)/float64(st.Before))
	if err := m.Save(*out); err != nil {
		log.Fatal(err)
	}
	for _, name := range []string{"model.safetensors", "tokenizer.json"} {
		a, _ := os.Stat(filepath.Join(*modelDir, name))
		b, _ := os.Stat(filepath.Join(*out, name))
		if a != nil && b != nil {
			log.Printf("%-18s %7.1f Mo → %7.1f Mo", name, mib(a.Size()), mib(b.Size()))
		}
	}

	if len(evalEx) > 0 {
		pruned, err := indecis.Load(*out)
		if err != nil {
			log.Fatal(err)
		}
		after := tokens(pruned, evalEx)
		same := 0
		for i := range after {
			if len(after[i]) == len(beforeTokens[i]) {
				same++
			}
		}
		log.Printf("textes tenus à l'écart découpés comme avant : %d/%d (%.1f %%)", same, len(after), 100*float64(same)/float64(len(after)))
		report(ctx, "après", pruned, evalEx)
	}
}

func read(globs string) []dataset.Example {
	var out []dataset.Example
	for _, g := range strings.Split(globs, ",") {
		files, err := filepath.Glob(os.ExpandEnv(strings.Replace(g, "~", "$HOME", 1)))
		if err != nil {
			log.Fatal(err)
		}
		slices.Sort(files)
		for _, f := range files {
			ex, err := dataset.ReadFile(f)
			if err != nil {
				log.Fatalf("%s : %v", f, err)
			}
			out = append(out, ex...)
		}
	}
	return out
}

func tokens(m *indecis.Model, ex []dataset.Example) [][]int32 {
	out := make([][]int32, len(ex))
	for i, e := range ex {
		n, err := m.TokenIDs(indecis.Input{Context: e.Context, Text: e.Text})
		if err != nil {
			log.Fatal(err)
		}
		out[i] = n
	}
	return out
}

// report évalue le modèle sur les exemples étiquetés selon son schéma.
func report(ctx context.Context, label string, m *indecis.Model, ex []dataset.Example) {
	var labeled []dataset.Example
	for _, e := range ex {
		if len(e.Labels) > 0 {
			labeled = append(labeled, e)
		}
	}
	if len(labeled) == 0 {
		return
	}
	metrics, err := m.Evaluate(ctx, labeled)
	if err != nil {
		log.Printf("%s : évaluation impossible : %v", label, err)
		return
	}
	for _, mt := range metrics {
		fmt.Printf("%-6s %v\n", label, mt)
	}
}

func mib(n int64) float64 { return float64(n) / (1 << 20) }
