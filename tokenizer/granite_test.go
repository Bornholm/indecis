package tokenizer

import (
	"os"
	"path/filepath"
	"testing"
)

// graniteDir is the directory of granite-embedding-311m-multilingual-r2
// (Gemma 3 vocabulary, raw pipeline, template <bos> A). Tests are skipped
// if it is absent.
func graniteDir(t testing.TB) string {
	t.Helper()
	dir := os.Getenv("INDECIS_GRANITE_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".cache/indecis/models/granite-embedding-311m-multilingual-r2")
	}
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err != nil {
		t.Skipf("granite model absent (%s): set INDECIS_GRANITE_DIR", dir)
	}
	return dir
}

func TestParityGranite(t *testing.T) {
	tok, err := Parse(mustRead(t, filepath.Join(graniteDir(t), "tokenizer.json")))
	if err != nil {
		t.Fatal(err)
	}
	checkParity(t, tok, "../testdata/granite311m/tokenizer_cases.jsonl")
	checkOffsets(t, tok, "../testdata/granite311m/offset_cases.jsonl")
}

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
