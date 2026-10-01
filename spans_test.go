package indecis

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/bornholm/indecis/dataset"
	"github.com/bornholm/indecis/tokenizer"
)

// Every token is kept by exactly one window, no window exceeds the size,
// and a kept token is at least an eighth of a window from an inner edge.
func TestPlanWindows(t *testing.T) {
	for _, size := range []int{1, 2, 3, 7, 16, 254} {
		for n := 0; n < 4*size+5; n++ {
			ws := planWindows(n, size)
			next := 0
			for i, w := range ws {
				if w.end-w.start > size || w.start < 0 || w.end > n {
					t.Fatalf("n=%d size=%d: window %d = %+v", n, size, i, w)
				}
				if w.keepFrom != next || w.keepFrom < w.start || w.keepTo > w.end || w.keepTo < w.keepFrom {
					t.Fatalf("n=%d size=%d: window %d = %+v, expected to keep from %d", n, size, i, w, next)
				}
				margin := size / 8
				if i > 0 && w.keepFrom-w.start < margin || i < len(ws)-1 && w.end-w.keepTo < margin {
					t.Fatalf("n=%d size=%d: window %d = %+v keeps tokens near an edge", n, size, i, w)
				}
				next = w.keepTo
			}
			if next != n {
				t.Fatalf("n=%d size=%d: kept up to %d", n, size, next)
			}
		}
	}
}

func TestSpansLabel(t *testing.T) {
	q := NewSpans("entities", "", "PER", "LOC")
	text := "Jean Dupont habite à Paris."
	list := func(items ...map[string]any) []any {
		var out []any
		for _, it := range items {
			out = append(out, it)
		}
		return out
	}
	got, ok, err := q.spans(list(
		map[string]any{"start": 22.0, "end": 27.0, "type": "LOC"},
		map[string]any{"start": 0.0, "end": 11.0, "type": "PER"},
	), text)
	if err != nil || !ok || !slices.Equal(got, []goldSpan{{0, 11, 0}, {22, 27, 1}}) {
		t.Fatalf("got %v %v %v", got, ok, err)
	}
	if got, ok, err := q.spans([]any{}, text); err != nil || !ok || len(got) != 0 {
		t.Fatalf("empty list: %v %v %v", got, ok, err)
	}
	if _, ok, _ := q.spans(nil, text); ok {
		t.Fatal("absent label read as present")
	}
	for name, v := range map[string]any{
		"overlap":      list(map[string]any{"start": 0.0, "end": 11.0, "type": "PER"}, map[string]any{"start": 5.0, "end": 15.0, "type": "LOC"}),
		"unknown type": list(map[string]any{"start": 0.0, "end": 4.0, "type": "ORG"}),
		"out of text":  list(map[string]any{"start": 20.0, "end": 99.0, "type": "LOC"}),
		"empty":        list(map[string]any{"start": 4.0, "end": 4.0, "type": "LOC"}),
		"not integer":  list(map[string]any{"start": 0.5, "end": 4.0, "type": "LOC"}),
		"not a list":   "PER",
	} {
		if _, _, err := q.spans(v, text); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func bekkoTokenizer(t *testing.T) *tokenizer.Tokenizer {
	tok, err := tokenizer.LoadShared(filepath.Join(bekkoDir(t), "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// Tagging the labeled passages then decoding certain tags gives the
// passages back, when their bounds fall between tokens.
func TestTagDecodeRoundTrip(t *testing.T) {
	tok := bekkoTokenizer(t)
	q := NewSpans("entities", "", "PER", "LOC", "ORG")
	cases := []struct {
		text  string
		spans []goldSpan
	}{
		{"Jean Dupont habite à Paris.", []goldSpan{{0, 11, 0}, {22, 27, 1}}},
		// "C," and "y," are single tokens: the comma is trimmed off.
		{"Le directeur de l'OMC, Pascal Lamy, est à Genève", []goldSpan{{18, 21, 2}, {23, 34, 0}, {43, 50, 1}}},
		{"(Lyon) « Marseille »", []goldSpan{{1, 5, 1}, {10, 19, 1}}},
		{"Marie Curie Marie Curie", []goldSpan{{0, 11, 0}, {12, 23, 0}}},
		{"  Lyon  ", []goldSpan{{2, 6, 1}}},
		{"rien à signaler", nil},
		{"", nil},
	}
	for _, c := range cases {
		ids, offs := tok.AppendOffsets(nil, nil, c.text)
		if len(ids) != len(offs) {
			t.Fatal("ids and offsets differ in length")
		}
		tags := tagTokens(c.text, offs, c.spans)
		C := spanTagCount(len(q.Options))
		logp := make([][]float64, len(tags))
		for i, tag := range tags {
			logp[i] = make([]float64, C)
			for k := range logp[i] {
				logp[i][k] = -30
			}
			logp[i][tag] = 0
		}
		got := decodeSpans(q, c.text, offs, logp, 0)
		var want []Span
		for _, g := range c.spans {
			want = append(want, Span{Type: q.Options[g.k], Start: g.start, End: g.end, Text: c.text[g.start:g.end], Confidence: 1})
		}
		if !spansEqual(got, want) {
			t.Errorf("%q: tags %v\n got  %v\n want %v", c.text, tags, got, want)
		}
	}
}

func spansEqual(a, b []Span) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Type != b[i].Type || a[i].Start != b[i].Start || a[i].End != b[i].End || a[i].Text != b[i].Text ||
			math.Abs(a[i].Confidence-b[i].Confidence) > 1e-6 {
			return false
		}
	}
	return true
}

// An inside tag never starts a passage nor follows another type, even
// when it is the most probable tag of its token.
func TestDecodeSpansConsistent(t *testing.T) {
	q := NewSpans("entities", "", "PER", "LOC")
	text := "a b c"
	offs := []tokenizer.Offset{{Start: 0, End: 1}, {Start: 1, End: 3}, {Start: 3, End: 5}}
	lp := func(p ...float64) []float64 {
		out := make([]float64, len(p))
		for i, v := range p {
			out[i] = math.Log(v)
		}
		return out
	}
	// tags: O, B-PER, I-PER, B-LOC, I-LOC
	logp := [][]float64{
		lp(0.30, 0.25, 0.40, 0.03, 0.02), // I-PER first: not allowed
		lp(0.10, 0.10, 0.10, 0.10, 0.60), // I-LOC after PER: not allowed
		lp(0.90, 0.025, 0.025, 0.025, 0.025),
	}
	for _, s := range decodeSpans(q, text, offs, logp, 0) {
		if s.Start == 0 && s.Type == "PER" && s.End == 1 {
			continue // B-PER on the first token is a valid reading
		}
		if s.Start != 0 && s.Start != 2 {
			t.Errorf("passage %+v does not start on a token", s)
		}
	}
	// The best valid sequence: B-PER I-PER O beats O B-LOC O? Check the
	// Viterbi score against brute force.
	best, bestTags := math.Inf(-1), []int(nil)
	for a := range 5 {
		for b := range 5 {
			for c := range 5 {
				tags := []int{a, b, c}
				ok := !isInside(a)
				for i := 1; i < 3 && ok; i++ {
					ok = !isInside(tags[i]) || tags[i-1] > 0 && tagType(tags[i-1]) == tagType(tags[i])
				}
				if s := logp[0][a] + logp[1][b] + logp[2][c]; ok && s > best {
					best, bestTags = s, tags
				}
			}
		}
	}
	got := decodeSpans(q, text, offs, logp, 0)
	t.Logf("brute force: %v, decoded: %+v", bestTags, got)
	var want []Span
	for i := 0; i < 3; {
		if bestTags[i] == 0 {
			i++
			continue
		}
		j := i + 1
		for j < 3 && isInside(bestTags[j]) {
			j++
		}
		start, _ := tokenCore(text, offs[i])
		want = append(want, Span{Type: q.Options[tagType(bestTags[i])], Start: start, End: offs[j-1].End})
		i = j
	}
	if len(got) != len(want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
	for i := range got {
		if got[i].Type != want[i].Type || got[i].Start != want[i].Start || got[i].End != want[i].End {
			t.Fatalf("got %+v, want %+v", got, want)
		}
	}
}

// toyEntities writes sentences that name a person and a city, labeled.
func toyEntities(n int, seed int64, people, cities []string) []dataset.Example {
	rng := rand.New(rand.NewSource(seed))
	frames := []string{"%s habite à %s.", "Hier, %s est parti pour %s.", "%s a écrit depuis %s une longue lettre.", "On a vu %s marcher dans %s ce matin."}
	var out []dataset.Example
	for range n {
		p, c := people[rng.Intn(len(people))], cities[rng.Intn(len(cities))]
		f := frames[rng.Intn(len(frames))]
		text := fmt.Sprintf(f, p, c)
		ps := strings.Index(text, p)
		cs := strings.LastIndex(text, c)
		out = append(out, dataset.Example{Text: text, Labels: map[string]any{
			"entities": []any{
				map[string]any{"start": float64(ps), "end": float64(ps + len(p)), "type": "PER"},
				map[string]any{"start": float64(cs), "end": float64(cs + len(c)), "type": "LOC"},
			},
			"has_city": true,
		}})
	}
	return out
}

// A Spans question learns to find unseen names, alongside a question on
// the pooled vector, in texts longer than one window, and survives a save.
func TestFitSpans(t *testing.T) {
	ctx := context.Background()
	schema := Schema{NewSpans("entities", "Personnes et lieux", "PER", "LOC"), NewNoul("has_city", "")}
	// 24 tokens per window: the joined sentences need several.
	m, err := New(bekkoDir(t), schema, 1, WithMaxLen(24))
	if err != nil {
		t.Fatal(err)
	}
	// Enough names that the model learns where a name is, not which
	// names exist.
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
	train := toyEntities(300, 1, people, []string{"Paris", "Lyon", "Marseille", "Toulouse", "Nantes", "Nice", "Strasbourg", "Montpellier", "Brest", "Dijon"})
	opts := DefaultTrainOptions()
	opts.Epochs, opts.BatchSize, opts.LR, opts.HeadLR = 2, 8, 2e-4, 3e-3
	if err := m.Fit(ctx, train, opts); err != nil {
		t.Fatal(err)
	}

	test := toyEntities(30, 2, []string{"Paul Lefèvre", "Claire Moreau", "Karim Haddad"}, []string{"Bordeaux", "Lille", "Rennes"})
	metrics, err := m.Evaluate(ctx, test)
	if err != nil {
		t.Fatal(err)
	}
	for _, mt := range metrics {
		t.Log(mt)
	}
	if f1 := metrics[0].F1; f1 < 0.8 {
		t.Fatalf("unseen names: F1 %.3f", f1)
	}

	// A long text: three sentences, read in several windows.
	var texts []string
	for _, e := range test[:3] {
		texts = append(texts, e.Text)
	}
	long := strings.Join(texts, " ")
	if n := len(m.tok.Encode(long)); n <= 24 {
		t.Fatalf("%d tokens: the text fits in one window", n)
	}
	d, err := m.Decide(ctx, long)
	if err != nil {
		t.Fatal(err)
	}
	found := d[0]["entities"].Spans
	t.Logf("%q: %+v", long, found)
	if len(found) != 6 {
		t.Fatalf("%d passages in the long text, want 6", len(found))
	}
	for _, s := range found {
		if long[s.Start:s.End] != s.Text {
			t.Fatalf("passage %+v does not match the text", s)
		}
	}

	if _, err := m.Calibrate(ctx, test); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := m.Save(dir); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	d2, err := loaded.Decide(ctx, long)
	if err != nil {
		t.Fatal(err)
	}
	a, b := d[0]["entities"].Spans, d2[0]["entities"].Spans
	if len(a) != len(b) {
		t.Fatalf("after loading: %+v, before %+v", b, a)
	}
	for i := range a {
		if a[i].Start != b[i].Start || a[i].End != b[i].End || a[i].Type != b[i].Type {
			t.Fatalf("after loading: %+v, before %+v", b, a)
		}
	}
	if logits, err := loaded.Logits(ctx, long); err != nil || logits[0]["entities"] != nil || logits[0]["has_city"] == nil {
		t.Fatalf("Logits: %v %v", logits, err)
	}
}

func TestSpansRefusePairs(t *testing.T) {
	if _, err := New(bekkoDir(t), Schema{NewSpans("entities", "", "PER")}, 1, WithPairs()); err == nil {
		t.Fatal("Spans question accepted with WithPairs")
	}
}

// A positive bias turns a hesitant token into a passage, without changing
// its confidence.
func TestDecodeSpansBias(t *testing.T) {
	q := NewSpans("entities", "", "PER")
	offs := []tokenizer.Offset{{Start: 0, End: 4}}
	logp := [][]float64{{math.Log(0.6), math.Log(0.3), math.Log(0.1)}} // O, B-PER, I-PER
	if got := decodeSpans(q, "Jean", offs, logp, 0); len(got) != 0 {
		t.Fatalf("without bias: %+v", got)
	}
	got := decodeSpans(q, "Jean", offs, logp, 1)
	if len(got) != 1 || got[0].Text != "Jean" || math.Abs(got[0].Confidence-0.3) > 1e-9 {
		t.Fatalf("with bias: %+v", got)
	}
}
