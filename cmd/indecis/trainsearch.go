package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/bornholm/indecis"
)

// trainSearch fine-tunes an embedding model for search on (query,
// document) pairs: indecis train -search -backbone dir -train pairs.jsonl
// -out model. Each line is an indecis.SearchPair: {"query", "document",
// "query_id"?, "document_id"?, "negatives"?: [...]}.
func trainSearch(backbone, globs, out string, maxLen, threads, batch, epochs int, lr float64, seed int64, int8Emb bool) error {
	pairs, err := readSearchPairs(globs)
	if err != nil {
		return err
	}
	m, err := indecis.Open(backbone, indecis.WithMaxLen(maxLen), indecis.WithThreads(threads))
	if err != nil {
		return err
	}
	batches := m.SearchBatches(pairs, batch, seed)
	if len(batches) == 0 {
		return fmt.Errorf("train -search: no batch (at least two distinct queries are needed)")
	}
	log.Printf("search mode: %d pairs, %d batches", len(pairs), len(batches))
	to := indecis.DefaultTrainOptions()
	to.Epochs, to.LR, to.Seed, to.Warmup = epochs, lr, seed, 0.1
	start := time.Now()
	to.Progress = func(p indecis.Progress) {
		if p.Step%20 == 0 || p.Step == p.Steps {
			log.Printf("step %d/%d  loss %.4f  %.0f tokens/s  %s", p.Step, p.Steps, p.Loss,
				float64(p.Tokens)/p.Elapsed.Seconds(), p.Elapsed.Round(time.Second))
		}
	}
	if err := m.FitEmbeddings(context.Background(), batches, to); err != nil {
		return err
	}
	log.Printf("training: %s", time.Since(start).Round(time.Second))
	if int8Emb {
		indecis.WithInt8Embeddings()(m)
	}
	if err := m.Save(out); err != nil {
		return err
	}
	log.Printf("model written to %s", out)
	return nil
}

func readSearchPairs(globs string) ([]indecis.SearchPair, error) {
	var out []indecis.SearchPair
	for _, g := range strings.Split(globs, ",") {
		if g = strings.TrimSpace(g); g == "" {
			continue
		}
		if strings.HasPrefix(g, "~/") {
			home, _ := os.UserHomeDir()
			g = filepath.Join(home, g[2:])
		}
		files, err := filepath.Glob(g)
		if err != nil {
			return nil, err
		}
		if len(files) == 0 {
			return nil, fmt.Errorf("no file for %q", g)
		}
		slices.Sort(files)
		for _, name := range files {
			f, err := os.Open(name)
			if err != nil {
				return nil, err
			}
			sc := bufio.NewScanner(f)
			sc.Buffer(make([]byte, 1<<20), 1<<26)
			for line := 1; sc.Scan(); line++ {
				if strings.TrimSpace(sc.Text()) == "" {
					continue
				}
				var p indecis.SearchPair
				if err := json.Unmarshal(sc.Bytes(), &p); err != nil {
					f.Close()
					return nil, fmt.Errorf("%s:%d: %w", name, line, err)
				}
				if p.Query == "" || p.Document == "" {
					f.Close()
					return nil, fmt.Errorf("%s:%d: query and document are required", name, line)
				}
				out = append(out, p)
			}
			err = sc.Err()
			f.Close()
			if err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}
