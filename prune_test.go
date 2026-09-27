package indecis

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestPruneVocabulary(t *testing.T) {
	ctx := context.Background()
	m, err := New(bekkoDir(t), toySchema, 1, WithMaxLen(64), WithPairs())
	if err != nil {
		t.Fatal(err)
	}
	var corpus []Input
	var texts []string
	for _, e := range toyData {
		corpus = append(corpus, Input{Context: "You are a billing assistant.", Text: e.Text})
		texts = append(texts, e.Text)
	}
	before, err := m.DecideInputs(ctx, corpus...)
	if err != nil {
		t.Fatal(err)
	}
	embBefore, _ := m.Embed(ctx, texts...)
	st, err := m.PruneVocabulary(corpus, 1)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("vocabulaire %d → %d", st.Before, st.After)
	if st.After >= st.Before/10 {
		t.Fatalf("élagage trop faible : %+v", st)
	}
	dir := t.TempDir()
	if err := m.Save(dir); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, mm := range []*Model{m, loaded} {
		after, err := mm.DecideInputs(ctx, corpus...)
		if err != nil {
			t.Fatal(err)
		}
		for i := range corpus {
			for q, a := range before[i] {
				if d := a.P - after[i][q].P; d > 1e-6 || d < -1e-6 || a.Choice != after[i][q].Choice {
					t.Fatalf("%q/%s : %+v puis %+v", corpus[i].Text, q, a, after[i][q])
				}
			}
		}
		embAfter, _ := mm.Embed(ctx, texts...)
		for i := range texts {
			for k := range embAfter[i] {
				if d := embAfter[i][k] - embBefore[i][k]; d > 1e-5 || d < -1e-5 {
					t.Fatalf("plongement %d changé", i)
				}
			}
		}
	}
	// Un texte hors corpus reste utilisable.
	if _, err := loaded.DecideInputs(ctx, Input{Text: "Ein völlig anderer Text mit unbekannten Wörtern: Donaudampfschifffahrt"}); err != nil {
		t.Fatalf("texte hors corpus refusé : %v", err)
	}
	fi, _ := os.Stat(filepath.Join(dir, "model.safetensors"))
	t.Logf("model.safetensors : %.1f Mo", float64(fi.Size())/(1<<20))
}

func TestInt8Embeddings(t *testing.T) {
	ctx := context.Background()
	m, err := New(bekkoDir(t), toySchema, 1, WithMaxLen(64))
	if err != nil {
		t.Fatal(err)
	}
	var texts []string
	for _, e := range toyData {
		texts = append(texts, e.Text)
	}
	want, _ := m.Embed(ctx, texts...)
	dirs := []string{t.TempDir(), t.TempDir()}
	WithInt8Embeddings()(m)
	if err := m.Save(dirs[0]); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(dirs[0])
	if err != nil {
		t.Fatal(err)
	}
	got, _ := loaded.Embed(ctx, texts...)
	for i := range texts {
		var dot float64
		for k := range got[i] {
			dot += float64(got[i][k]) * float64(want[i][k])
		}
		if dot < 0.999 {
			t.Errorf("%q : cosinus %.5f", texts[i], dot)
		}
	}
	// Le format se conserve d'une sauvegarde à l'autre.
	if err := loaded.Save(dirs[1]); err != nil {
		t.Fatal(err)
	}
	a, _ := os.Stat(filepath.Join(dirs[0], "model.safetensors"))
	b, _ := os.Stat(filepath.Join(dirs[1], "model.safetensors"))
	if a.Size() != b.Size() || a.Size() > 150<<20 {
		t.Fatalf("tailles %d puis %d", a.Size(), b.Size())
	}
	t.Logf("model.safetensors en int8 : %.1f Mo", float64(a.Size())/(1<<20))
}
