package indecis

import (
	"context"
	"testing"
)

// A model pruned to a sixteenth of its MLP, then trained on its teacher's
// tags alone (Distill 1: the labels weigh nothing), tags unseen names as
// its teacher does; saved and loaded, it keeps its size.
func TestPruneAndDistill(t *testing.T) {
	ctx := context.Background()
	schema := Schema{NewSpans("entities", "Personnes et lieux", "PER", "LOC"), NewNoul("has_city", "")}
	teacher, err := New(bekkoDir(t), schema, 1, WithMaxLen(24))
	if err != nil {
		t.Fatal(err)
	}
	first := []string{"Jean", "Marie", "Ahmed", "Sophie", "Lucas", "Fatou", "Hugo", "Léa", "Yann", "Inès", "Omar", "Chloé", "Théo", "Nadia", "Louis", "Emma"}
	last := []string{"Dupont", "Curie", "Benali", "Martin", "Bernard", "Diallo", "Petit", "Roux", "Garnier", "Nguyen", "Faure", "Blanc", "Mercier", "Gauthier", "Henry", "Perrin"}
	var people []string
	for i, f := range first {
		for j, l := range last {
			if (i+j)%3 == 0 {
				people = append(people, f+" "+l)
			}
		}
	}
	cities := []string{"Paris", "Lyon", "Marseille", "Toulouse", "Nantes", "Nice", "Strasbourg", "Montpellier", "Brest", "Dijon"}
	train := toyEntities(300, 1, people, cities)
	opts := DefaultTrainOptions()
	opts.Epochs, opts.BatchSize, opts.LR, opts.HeadLR = 2, 8, 2e-4, 3e-3
	if err := teacher.Fit(ctx, train, opts); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := teacher.Save(dir); err != nil {
		t.Fatal(err)
	}
	student, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	var texts []string
	for _, e := range train[:100] {
		texts = append(texts, e.Text)
	}
	I := student.enc.Cfg.Intermediate
	if err := student.PruneMLP(ctx, texts, I/16); err != nil {
		t.Fatal(err)
	}
	test := toyEntities(30, 2, []string{"Paul Lefèvre", "Claire Moreau", "Karim Haddad"}, []string{"Bordeaux", "Lille", "Rennes"})
	pruned, err := student.Evaluate(ctx, test)
	if err != nil {
		t.Fatal(err)
	}
	t.Log("pruned:", pruned[0])

	other, _ := New(bekkoDir(t), Schema{NewSpans("entities", "", "PER")}, 1, WithMaxLen(24))
	opts.Teacher = other
	if err := student.Fit(ctx, train[:8], opts); err == nil {
		t.Fatal("a teacher with other questions was accepted")
	}

	opts.Teacher, opts.Distill, opts.DistillTemp, opts.DistillHidden = teacher, 1, 2, 1
	opts.Epochs = 2
	if err := student.Fit(ctx, train, opts); err != nil {
		t.Fatal(err)
	}
	mt, err := student.Evaluate(ctx, test)
	if err != nil {
		t.Fatal(err)
	}
	t.Log("distilled:", mt[0])
	if mt[0].F1 < 0.8 || mt[0].F1 <= pruned[0].F1 {
		t.Fatalf("pruned and distilled: F1 %.3f, %.3f before training", mt[0].F1, pruned[0].F1)
	}
	if err := student.SetTemperatures(map[string]float64{"entities": 0.8}); err != nil || student.Temperatures()["entities"] != 0.8 {
		t.Fatalf("SetTemperatures: %v %v", student.Temperatures(), err)
	}
	if student.SetTemperatures(map[string]float64{"cities": 1}) == nil || student.SetTemperatures(map[string]float64{"entities": 0}) == nil {
		t.Fatal("SetTemperatures accepted an unknown question or a zero temperature")
	}
	small := t.TempDir()
	if err := student.Save(small); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(small, WithInt8())
	if err != nil {
		t.Fatal(err)
	}
	if loaded.enc.Cfg.Intermediate != I/16 {
		t.Fatalf("loaded with %d MLP neurons, want %d", loaded.enc.Cfg.Intermediate, I/16)
	}
	if mt, err := loaded.Evaluate(ctx, test); err != nil || mt[0].F1 < 0.8 {
		t.Fatalf("loaded: %v %v", mt, err)
	}
}
