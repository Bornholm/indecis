package ocr

import (
	"os"
	"path/filepath"
	"testing"
)

// indraDir holds Indra OCR's recognizer converted by
// tools/oracle/ocr_indra_check.py; its weights are not distributable.
func indraDir(t testing.TB) string {
	dir := os.Getenv("INDECIS_INDRA_REC_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".cache/indecis/models/indra-rec")
	}
	if _, err := os.Stat(filepath.Join(dir, "vocab.json")); err != nil {
		t.Skip("Indra recognizer missing (tools/oracle/ocr_indra_check.py)")
	}
	return dir
}

func TestReadPageWithIndra(t *testing.T) {
	r, err := Load(indraDir(t))
	if err != nil {
		t.Fatal(err)
	}
	page, cases := loadPage(t)
	boxes := make([]Box, len(cases))
	for i, c := range cases {
		boxes[i] = c.box()
	}
	var e ErrorRate
	for i, l := range r.ReadLines(page, boxes) {
		e.Add(l.Text, cases[i].Text)
		t.Logf("%q (confidence %.2f)", l.Text, l.Confidence)
		if len(l.Chars) > 0 {
			first, last := l.Chars[0], l.Chars[len(l.Chars)-1]
			if first.X0 < boxes[i].X0-30 || last.X1 > boxes[i].X1+30 {
				t.Errorf("line %d: characters span %.0f–%.0f, box %.0f–%.0f", i, first.X0, last.X1, boxes[i].X0, boxes[i].X1)
			}
		}
	}
	t.Logf("CER %.2f%%, WER %.2f%%, %d/%d lines exact", 100*e.CER(), 100*e.WER(), e.Exact, e.Lines)
	if e.CER() > 0.02 {
		t.Errorf("CER %.2f%%", 100*e.CER())
	}
}
