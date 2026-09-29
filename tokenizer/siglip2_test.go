package tokenizer

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// siglip2 loads the tokenizer of SigLIP 2, the original Gemma pipeline
// (no leading ▁, no split, no <bos>). Skipped if the model is absent.
func siglip2(t testing.TB) *Tokenizer {
	t.Helper()
	dir := os.Getenv("INDECIS_SIGLIP2_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".cache/indecis/models/siglip2-base-patch32-256")
	}
	path := filepath.Join(dir, "tokenizer.json")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("SigLIP 2 model absent (%s): set INDECIS_SIGLIP2_DIR", dir)
	}
	tok, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// Parity with the Hugging Face tokenizers library for the raw pipeline.
func TestParitySiglip2(t *testing.T) {
	checkParity(t, siglip2(t), "../testdata/siglip2/tokenizer_cases.jsonl")
}

func TestSiglip2Template(t *testing.T) {
	tok := siglip2(t)
	// "a photo of a cat": no <bos>, <eos> at the end, as the reference.
	if got, want := tok.Encode("a photo of a cat"), []int32{235250, 2686, 576, 476, 4401, 1}; !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}
