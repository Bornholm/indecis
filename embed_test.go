package indecis

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestTruncate(t *testing.T) {
	v := []float32{0.6, 0.8, 0, 0}
	if got := Truncate(v, 0); len(got) != 4 {
		t.Fatalf("dim 0: %v", got)
	}
	got := Truncate([]float32{3, 4, 12}, 2)
	if len(got) != 2 || math.Abs(float64(got[0])-0.6) > 1e-7 || math.Abs(float64(got[1])-0.8) > 1e-7 {
		t.Fatalf("%v", got)
	}
}

// bekkoVariant links the files of the bekko backbone into a new directory
// and adds extra sentence-transformers files.
func bekkoVariant(t *testing.T, extra map[string]string) string {
	t.Helper()
	src := bekkoDir(t)
	dir := t.TempDir()
	for _, f := range []string{"config.json", "model.safetensors", "tokenizer.json"} {
		if err := os.Symlink(filepath.Join(src, f), filepath.Join(dir, f)); err != nil {
			t.Fatal(err)
		}
	}
	for name, content := range extra {
		p := filepath.Join(dir, name)
		os.MkdirAll(filepath.Dir(p), 0o755)
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// Prompts and pooling come from the sentence-transformers files and
// survive Save and Load.
func TestEmbedPromptsAndPooling(t *testing.T) {
	ctx := context.Background()
	dir := bekkoVariant(t, map[string]string{
		"config_sentence_transformers.json": `{"prompts": {"query": "query: ", "document": ""}}`,
		"1_Pooling/config.json":             `{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false}`,
	})
	m, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if p := m.Info().Prompts; len(p) != 1 || p["query"] != "query: " {
		t.Fatalf("prompts %v", p)
	}
	q, _ := m.EmbedQueries(ctx, "facture impayée")
	e, _ := m.Embed(ctx, "query: facture impayée")
	d, _ := m.EmbedDocuments(ctx, "facture impayée")
	if dot(q[0], e[0]) < 1-1e-6 || dot(q[0], d[0]) > 1-1e-4 {
		t.Fatalf("query·prompted %v, query·document %v", dot(q[0], e[0]), dot(q[0], d[0]))
	}
	mean, err := Open(bekkoDir(t))
	if err != nil {
		t.Fatal(err)
	}
	mv, _ := mean.Embed(ctx, "facture impayée")
	if dot(mv[0], d[0]) > 1-1e-4 {
		t.Fatal("cls pooling gives the mean embedding")
	}
	saved := t.TempDir()
	if err := m.Save(saved); err != nil {
		t.Fatal(err)
	}
	l, err := Load(saved)
	if err != nil {
		t.Fatal(err)
	}
	lq, _ := l.EmbedQueries(ctx, "facture impayée")
	if dot(lq[0], q[0]) < 1-1e-6 {
		t.Fatalf("after Load: %v", dot(lq[0], q[0]))
	}
}

// With mean pooling, prompts that sentence-transformers leaves out of the
// mean are refused rather than embedded differently.
func TestEmbedRefusesExcludedPrompts(t *testing.T) {
	dir := bekkoVariant(t, map[string]string{
		"config_sentence_transformers.json": `{"prompts": {"query": "query: "}}`,
		"1_Pooling/config.json":             `{"pooling_mode_mean_tokens": true, "include_prompt": false}`,
	})
	if _, err := Open(dir); err == nil {
		t.Fatal("accepted")
	}
}

func TestSearchBatches(t *testing.T) {
	m := &Model{info: Info{Prompts: map[string]string{"query": "q: "}}}
	pairs := []SearchPair{
		{QueryID: "a", Query: "A", DocumentID: "1", Document: "doc 1", Negatives: []string{"doc 3"}},
		{QueryID: "a", Query: "A", DocumentID: "2", Document: "doc 2"},
		{QueryID: "b", Query: "B", DocumentID: "3", Document: "doc 3"},
		{QueryID: "c", Query: "C", DocumentID: "1", Document: "doc 1"},
		{QueryID: "d", Query: "D", DocumentID: "4", Document: "doc 4"},
	}
	for seed := int64(0); seed < 20; seed++ {
		checkSearchBatches(t, m.SearchBatches(pairs, 3, seed))
	}
}

func checkSearchBatches(t *testing.T, batches []ChoiceBatch) {
	t.Helper()
	used := 0
	for _, b := range batches {
		seen := map[string]bool{}
		for i, q := range b.Texts {
			if seen[q] {
				t.Fatalf("query %q twice in a batch", q)
			}
			seen[q] = true
			if q[:3] != "q: " {
				t.Fatalf("query prompt missing: %q", q)
			}
			// Every masked option is a relevant document of the query.
			for _, j := range b.Correct[i][1:] {
				d := b.Candidates[j].Name
				ok := (q == "q: A" && (d == "doc 1" || d == "doc 2")) || (q == "q: C" && d == "doc 1") || (q == "q: B" && d == "doc 3")
				if !ok {
					t.Fatalf("%q: option %q masked", q, d)
				}
			}
			doc3 := 0
			for _, c := range b.Candidates {
				if c.Name == "doc 3" {
					doc3++
				}
			}
			if q == "q: B" && doc3 == 2 {
				// doc 3, B's target, is also A's hard negative: it must not
				// count as a negative for B.
				if len(b.Correct[i]) != 2 {
					t.Fatalf("B: %v", b.Correct[i])
				}
			}
		}
		used += len(b.Texts)
	}
	if used != 5 {
		t.Fatalf("%d pairs used out of 5", used)
	}
}
