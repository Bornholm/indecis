package ocr

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bornholm/indecis/internal/linerec"
)

func pageLines(t *testing.T) []ShardLine {
	page, cases := loadPage(t)
	var lines []ShardLine
	for _, c := range cases {
		cr := CropLine(page, c.box(), DefaultCrop())
		lines = append(lines, ShardLine{Text: NormalizeText(c.Text), W: cr.W, Rows: len(cr.Ink) / cr.W, Ink: cr.Ink})
	}
	return lines
}

// TestTrainLearnsLines: a small model learns the page's five lines by
// heart.
func TestTrainLearnsLines(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a model")
	}
	v, err := NewVocab(DefaultCharset)
	if err != nil {
		t.Fatal(err)
	}
	cfg := linerec.Base(len(v.Chars) + 1)
	cfg.Dim, cfg.Expand, cfg.SE, cfg.Blocks = 48, 96, 16, 3
	cfg.CtxChannels, cfg.CtxDim = 8, 16
	m, err := linerec.New(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	lines := pageLines(t)
	var log strings.Builder
	res, err := Train(context.Background(), m, v, NewMemorySource(lines, 1), TrainOptions{
		Steps: 400, Batch: 5, LR: 3e-3, Warmup: 20, Clip: 5, Eval: lines, EvalEvery: 100, LogEvery: 50, Log: &log,
	})
	t.Log("\n" + log.String())
	if err != nil {
		t.Fatal(err)
	}
	if res.Skipped > 0 || res.BestCER > 0 {
		t.Fatalf("skipped %d, best CER %.2f%%", res.Skipped, 100*res.BestCER)
	}
}

func TestShardRoundTrip(t *testing.T) {
	lines := pageLines(t)
	path := filepath.Join(t.TempDir(), "a.lines")
	w, err := CreateShard(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range lines {
		if err := w.Write(l.Text, Crop{Ink: l.Ink, W: l.W}, map[string]string{"src": "page"}); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	r, err := OpenShard(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	for i := 0; ; i++ {
		l, err := r.Next()
		if err == io.EOF {
			if i != len(lines) {
				t.Fatalf("%d lines read, want %d", i, len(lines))
			}
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		want := lines[i]
		if l.Text != want.Text || l.W != want.W || l.Meta["src"] != "page" {
			t.Fatalf("line %d: %q %d", i, l.Text, l.W)
		}
		for j, v := range l.Ink {
			if d := v - want.Ink[j]; d > 0.5/255+1e-6 || d < -0.5/255-1e-6 {
				t.Fatalf("line %d: ink %d: %v vs %v", i, j, v, want.Ink[j])
			}
		}
	}
	src, err := NewShardSource([]string{path}, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]int{}
	for range 3 * len(lines) {
		l, err := src.Next()
		if err != nil {
			t.Fatal(err)
		}
		seen[l.Text]++
	}
	if len(seen) != len(lines) {
		t.Fatalf("shard source saw %d distinct lines", len(seen))
	}
	_ = os.Remove(path)
}

func TestVocab(t *testing.T) {
	v, err := NewVocab(DefaultCharset)
	if err != nil {
		t.Fatal(err)
	}
	e := v.Encoder()
	ids, _, ok := e.Encode("Œuvre à 12 €")
	if !ok || len(ids) != 12 {
		t.Fatalf("ids %v ok %v", ids, ok)
	}
	if _, r, ok := e.Encode("日本"); ok || r != '日' {
		t.Fatalf("unknown character not reported")
	}
	if got := NormalizeText("  la\u00a0ﬁn\u202f!  ne\u0301ces¬  "); got != "la fin ! néces-" {
		t.Fatalf("NormalizeText: %q", got)
	}
	if _, err := NewVocab("aba"); err == nil {
		t.Fatal("duplicate accepted")
	}
}
