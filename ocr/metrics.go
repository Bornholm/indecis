package ocr

import (
	"regexp"
	"strings"
)

// ErrorRate accumulates edit distances against references: the character
// and word error rates of a set of lines (CER, WER), totals over totals.
type ErrorRate struct {
	CharEdits, Chars int
	WordEdits, Words int
	Lines, Exact     int
}

// Add counts one line: got read for want, both in CompareForm.
func (e *ErrorRate) Add(got, want string) {
	got, want = CompareForm(got), CompareForm(want)
	g, w := []rune(got), []rune(want)
	e.CharEdits += levenshtein(g, w)
	e.Chars += len(w)
	gw, ww := strings.Fields(got), strings.Fields(want)
	e.WordEdits += levenshtein(gw, ww)
	e.Words += len(ww)
	e.Lines++
	if got == want {
		e.Exact++
	}
}

// Merge adds the counts of another accumulator.
func (e *ErrorRate) Merge(o ErrorRate) {
	e.CharEdits += o.CharEdits
	e.Chars += o.Chars
	e.WordEdits += o.WordEdits
	e.Words += o.Words
	e.Lines += o.Lines
	e.Exact += o.Exact
}

// CER is the character error rate.
func (e ErrorRate) CER() float64 { return rate(e.CharEdits, e.Chars) }

// WER is the word error rate.
func (e ErrorRate) WER() float64 { return rate(e.WordEdits, e.Words) }

func rate(edits, n int) float64 {
	if n == 0 {
		if edits == 0 {
			return 0
		}
		return 1
	}
	return float64(edits) / float64(n)
}

// levenshtein is the edit distance (insertions, deletions, substitutions).
func levenshtein[T comparable](a, b []T) int {
	prev := make([]int, len(b)+1)
	cur := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := range a {
		cur[0] = i + 1
		for j := range b {
			c := prev[j]
			if a[i] != b[j] {
				c = 1 + min(prev[j], prev[j+1], cur[j])
			}
			cur[j+1] = c
		}
		prev, cur = cur, prev
	}
	return prev[len(b)]
}

// leaders matches a run of dot leaders (a table of contents' "......"),
// spaces included.
var leaders = regexp.MustCompile(`\.(\s*\.){3,}`)

// CompareForm is the form texts are compared in: dot leaders cut to
// "...". A recognizer reading one step every few pixels cannot count dots
// spaced closer than that; neither can the references' readers agree on
// their number.
func CompareForm(s string) string {
	return leaders.ReplaceAllString(s, "...")
}
