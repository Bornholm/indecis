package tokenizer

import (
	"crypto/sha256"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// binaryCopy copies a tokenizer.json into a temporary directory, writes
// its tokenizer.bin and returns the path of the copy.
func binaryCopy(t *testing.T, src string) string {
	t.Helper()
	data, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := WriteBinary(path); err != nil {
		t.Fatal(err)
	}
	if _, err := loadBinary(BinaryPath(path), sha256.Sum256(data)); err != nil {
		t.Fatalf("tokenizer.bin not loadable: %v", err)
	}
	return path
}

// A mapped tokenizer encodes like the parsed one.
func TestBinaryParity(t *testing.T) {
	path := binaryCopy(t, "../testdata/siglip2-tiny/tokenizer.json")
	for _, load := range []func(string) (*Tokenizer, error){Load, LoadShared} {
		tok, err := load(path)
		if err != nil {
			t.Fatal(err)
		}
		checkParity(t, tok, "../testdata/siglip2-tiny/tokenizer_cases.jsonl")
		checkOffsets(t, tok, "../testdata/siglip2-tiny/offset_cases.jsonl")
	}

	path = binaryCopy(t, filepath.Join(bekkoDir(t), "tokenizer.json"))
	tok, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	ref := bekko(t)
	if tok.VocabSize() != ref.VocabSize() || tok.PadID() != ref.PadID() || tok.BosID() != ref.BosID() || tok.EosID() != ref.EosID() {
		t.Fatal("special ids differ")
	}
	for id := int32(0); id < int32(ref.VocabSize()); id++ {
		if tok.Token(id) != ref.Token(id) {
			t.Fatalf("Token(%d) = %q, want %q", id, tok.Token(id), ref.Token(id))
		}
	}
	checkParity(t, tok, "../testdata/bekko/tokenizer_cases.jsonl")
	checkOffsets(t, tok, "../testdata/bekko/offset_cases.jsonl")
}

// A tokenizer.bin written from another JSON is ignored: the JSON is read.
func TestBinaryStale(t *testing.T) {
	path := binaryCopy(t, "../testdata/siglip2-tiny/tokenizer.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadBinary(BinaryPath(path), sha256.Sum256(data)); !errors.Is(err, errStale) {
		t.Fatalf("loadBinary: %v, want errStale", err)
	}
	tok, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	checkParity(t, tok, "../testdata/siglip2-tiny/tokenizer_cases.jsonl")
}
