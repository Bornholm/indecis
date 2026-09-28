package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

// runCompact réduit la place d'un modèle et mesure l'effet sur des textes
// tenus à l'écart :
//
//   - -corpus réduit le vocabulaire aux tokens qu'utilise un corpus
//     représentatif (Model.PruneVocabulary) : sans perte sur les textes
//     qu'il couvre, avec une perte possible sur des textes imprévus ;
//   - -int8-embeddings écrit la table d'embeddings en int8
//     (WithInt8Embeddings) : deux fois plus petite, sans perte mesurée.
//
// Le corpus doit ressembler aux textes que le modèle jugera ; les jeux
// d'évaluation ne doivent pas en faire partie.
func runCompact(args []string) error {
	fs := flag.NewFlagSet("compact", flag.ExitOnError)
	modelDir := fs.String("model", "", "modèle à réduire")
	out := fs.String("out", "", "répertoire du modèle réduit")
	corpusGlobs := fs.String("corpus", "", "élaguer le vocabulaire sur ce corpus (JSONL, motifs séparés par des virgules)")
	evalGlobs := fs.String("eval", "", "exemples tenus à l'écart, pour mesurer l'effet (facultatif)")
	minCount := fs.Int("min-count", 1, "occurrences minimales d'un token dans le corpus")
	int8Emb := fs.Bool("int8-embeddings", false, "écrire la table d'embeddings en int8")
	fs.Parse(args)
	if err := required(fs, "model", "out"); err != nil {
		return err
	}
	if *corpusGlobs == "" && !*int8Emb {
		return fmt.Errorf("rien à faire : -corpus et/ou -int8-embeddings")
	}
	ctx := context.Background()
	m, err := indecis.Load(*modelDir)
	if err != nil {
		return err
	}
	var evalEx []dataset.Example
	if *evalGlobs != "" {
		if evalEx, err = readExamples(*evalGlobs); err != nil {
			return err
		}
		fmt.Println("avant :")
		if err := report(ctx, m, nil, evalEx, false); err != nil {
			return err
		}
	}
	if *corpusGlobs != "" {
		corpus, err := readExamples(*corpusGlobs)
		if err != nil {
			return err
		}
		inputs := make([]indecis.Input, len(corpus))
		for i, e := range corpus {
			inputs[i] = indecis.Input{Context: e.Context, Text: e.Text}
		}
		st, err := m.PruneVocabulary(inputs, *minCount)
		if err != nil {
			return err
		}
		log.Printf("vocabulaire : %d → %d tokens (%.1f %%), corpus de %d textes", st.Before, st.After, 100*float64(st.After)/float64(st.Before), len(corpus))
	}
	if *int8Emb {
		indecis.WithInt8Embeddings()(m)
	}
	if err := m.Save(*out); err != nil {
		return err
	}
	for _, name := range []string{"model.safetensors", "tokenizer.json"} {
		a, _ := os.Stat(filepath.Join(*modelDir, name))
		b, _ := os.Stat(filepath.Join(*out, name))
		if a != nil && b != nil {
			log.Printf("%-18s %7.1f Mo → %7.1f Mo", name, float64(a.Size())/(1<<20), float64(b.Size())/(1<<20))
		}
	}
	if len(evalEx) > 0 {
		reduced, err := indecis.Load(*out)
		if err != nil {
			return err
		}
		fmt.Println("après :")
		return report(ctx, reduced, nil, evalEx, false)
	}
	return nil
}
