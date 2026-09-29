package tokenizer

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// A tokenizer pruned on a corpus splits that corpus exactly like the
// original (aside from the ids), and splits any other text into valid ids
// that describe the same text.
func TestPrune(t *testing.T) {
	tok := bekko(t)
	src, err := os.ReadFile(filepath.Join(bekkoDir(t), "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	fx := readFixtures(t, "../testdata/bekko/tokenizer_cases.jsonl")
	var corpus, other []string
	for i, f := range fx {
		if f.Pair {
			continue
		}
		if i%2 == 0 {
			corpus = append(corpus, f.Text)
		} else {
			other = append(other, f.Text)
		}
	}
	used := map[int32]bool{}
	for _, s := range corpus {
		tok.Trace(s, func(id int32) { used[id] = true })
	}
	out, newID, err := Prune(src, func(id int32) bool { return used[id] })
	if err != nil {
		t.Fatal(err)
	}
	pruned, err := Parse(out)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d tokens kept out of %d (%d seen in the corpus)", pruned.VocabSize(), tok.VocabSize(), len(used))
	for _, s := range corpus {
		want := tok.Encode(s)
		for i, id := range want {
			want[i] = newID[id]
		}
		if got := pruned.Encode(s); !slices.Equal(got, want) {
			t.Fatalf("%.40q: split changed", s)
		}
	}
	changed := 0
	for _, s := range other {
		ids := pruned.Encode(s)
		var text, ref string
		for _, id := range ids {
			if id < 0 || int(id) >= pruned.VocabSize() {
				t.Fatalf("id %d out of vocabulary", id)
			}
			text += pruned.Token(id)
		}
		for _, id := range tok.Encode(s) {
			ref += tok.Token(id)
		}
		if text != ref && !hasByteTokens(pruned, ids) {
			t.Fatalf("%.40q: split no longer describes the same text", s)
		}
		if len(ids) != len(tok.Encode(s)) {
			changed++
		}
	}
	t.Logf("%d texts out of corpus split differently out of %d", changed, len(other))
	if pruned.BosID() != newID[tok.BosID()] || pruned.EosID() != newID[tok.EosID()] {
		t.Fatal("special tokens mis-renumbered")
	}
}

func hasByteTokens(t *Tokenizer, ids []int32) bool {
	for _, id := range ids {
		if isByteToken(t.Token(id)) {
			return true
		}
	}
	return false
}

func TestLoadShared(t *testing.T) {
	path := filepath.Join(bekkoDir(t), "tokenizer.json")
	copyPath := filepath.Join(t.TempDir(), "tokenizer.json")
	b, _ := os.ReadFile(path)
	os.WriteFile(copyPath, b, 0o644)
	a, err := LoadShared(path)
	if err != nil {
		t.Fatal(err)
	}
	c, err := LoadShared(copyPath)
	if err != nil {
		t.Fatal(err)
	}
	if a != c {
		t.Fatal("two identical files should give the same tokenizer")
	}
}
