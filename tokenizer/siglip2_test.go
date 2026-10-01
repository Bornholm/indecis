package tokenizer

import (
	"encoding/json"
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

// The raw pipeline without the checkpoint: a small tokenizer cut from
// SigLIP 2's (tools/oracle/tiny_tokenizer.py), against the tokenizers
// library.
func TestParitySiglip2Tiny(t *testing.T) {
	tok, err := Load("../testdata/siglip2-tiny/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Encode("a"); got[0] == tok.BosID() {
		t.Fatalf("Encode(a) = %v: the raw pipeline starts without <bos>", got)
	}
	checkParity(t, tok, "../testdata/siglip2-tiny/tokenizer_cases.jsonl")
}

// Near misses of the two known pipelines are refused, not misread.
func TestPipelineNearMisses(t *testing.T) {
	b, err := os.ReadFile("../testdata/siglip2-tiny/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name   string
		mutate func(j map[string]any)
	}{
		{"inverted split", func(j map[string]any) { j["pre_tokenizer"].(map[string]any)["invert"] = true }},
		{"other split behavior", func(j map[string]any) { j["pre_tokenizer"].(map[string]any)["behavior"] = "Isolated" }},
		{"no normalizer", func(j map[string]any) { j["normalizer"] = nil }},
		{"template with <bos>", func(j map[string]any) {
			pp := j["post_processor"].(map[string]any)
			pp["single"] = append([]any{map[string]any{"SpecialToken": map[string]any{"id": "<bos>", "type_id": 0}}}, pp["single"].([]any)...)
		}},
		{"no post-processor", func(j map[string]any) { j["post_processor"] = nil }},
	} {
		var j map[string]any
		if err := json.Unmarshal(b, &j); err != nil {
			t.Fatal(err)
		}
		c.mutate(j)
		out, _ := json.Marshal(j)
		path := filepath.Join(t.TempDir(), "tokenizer.json")
		if err := os.WriteFile(path, out, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(path); err == nil {
			t.Errorf("%s: accepted", c.name)
		}
	}
}

// Offsets of the raw pipeline, on the small tokenizer.
func TestOffsetsParitySiglip2Tiny(t *testing.T) {
	tok, err := Load("../testdata/siglip2-tiny/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	checkOffsets(t, tok, "../testdata/siglip2-tiny/offset_cases.jsonl")
}
