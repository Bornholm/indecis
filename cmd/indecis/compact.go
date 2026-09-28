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

// runCompact reduces the size of a model and measures the effect on
// texts held out:
//
//   - -corpus reduces the vocabulary to the tokens used by a
//     representative corpus (Model.PruneVocabulary): no loss on the
//     texts it covers, with possible loss on unforeseen texts;
//   - -int8-embeddings writes the embeddings table in int8
//     (WithInt8Embeddings): twice as small, with no measured loss.
//
// The corpus should resemble the texts the model will judge; evaluation
// sets must not be part of it.
func runCompact(args []string) error {
	fs := flag.NewFlagSet("compact", flag.ExitOnError)
	modelDir := fs.String("model", "", "model to reduce")
	out := fs.String("out", "", "directory of the reduced model")
	corpusGlobs := fs.String("corpus", "", "prune the vocabulary on this corpus (JSONL, comma-separated patterns)")
	evalGlobs := fs.String("eval", "", "held-out examples, to measure the effect (optional)")
	minCount := fs.Int("min-count", 1, "minimum occurrences of a token in the corpus")
	int8Emb := fs.Bool("int8-embeddings", false, "write the embeddings table in int8")
	fs.Parse(args)
	if err := required(fs, "model", "out"); err != nil {
		return err
	}
	if *corpusGlobs == "" && !*int8Emb {
		return fmt.Errorf("nothing to do: -corpus and/or -int8-embeddings")
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
		fmt.Println("before:")
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
		log.Printf("vocabulary: %d -> %d tokens (%.1f%%), corpus of %d texts", st.Before, st.After, 100*float64(st.After)/float64(st.Before), len(corpus))
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
			log.Printf("%-18s %7.1f MB -> %7.1f MB", name, float64(a.Size())/(1<<20), float64(b.Size())/(1<<20))
		}
	}
	if len(evalEx) > 0 {
		reduced, err := indecis.Load(*out)
		if err != nil {
			return err
		}
		fmt.Println("after:")
		return report(ctx, reduced, nil, evalEx, false)
	}
	return nil
}
