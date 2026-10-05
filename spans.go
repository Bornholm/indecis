package indecis

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/bornholm/indecis/tokenizer"
)

// A Spans question finds passages of the text and gives each one a type:
// the named entities of a sentence, the personal data of a document. Its
// options are the types. The head reads every token, not the pooled
// vector, and tags it in BIO: outside, beginning or inside of a passage
// of type k. The best consistent sequence of tags (Viterbi: an inside tag
// follows a tag of the same type) gives the passages.
//
// The label of an example is the list of its passages, as byte offsets in
// the text:
//
//	{"text": "Jean Dupont habite à Paris.", "labels": {"entities": [
//	  {"start": 0, "end": 11, "type": "PER"}, {"start": 22, "end": 27, "type": "LOC"}]}}
//
// An empty list says the text has none. Passages must not overlap.
//
// Texts longer than the model (WithMaxLen) are read in overlapping
// windows, in training as in inference: each token is tagged by the window
// where it is furthest from an edge. The other questions of the schema
// read the first window, the truncation they always had.

// Span is a passage found by a Spans question: text[Start:End].
type Span struct {
	Type  string `json:"type"`
	Start int    `json:"start"`
	End   int    `json:"end"`
	Text  string `json:"text"`
	// Confidence is the mean probability of the tags of the passage.
	Confidence float64 `json:"confidence"`
}

// NewSpans builds a question that finds passages of the given types.
func NewSpans(name, instructions string, types ...string) Question {
	return Question{Name: name, Kind: Spans, Instructions: instructions, Options: types}
}

// goldSpan is a labeled passage, k the index of its type.
type goldSpan struct{ start, end, k int }

// spans converts a Spans label into passages sorted by start. ok is false
// if the label is absent.
func (q Question) spans(v any, text string) (out []goldSpan, ok bool, err error) {
	if v == nil {
		return nil, false, nil
	}
	list, isList := v.([]any)
	if !isList {
		return nil, false, fmt.Errorf("%s: expected a list of passages, got %T", q.Name, v)
	}
	for _, item := range list {
		obj, isObj := item.(map[string]any)
		if !isObj {
			return nil, false, fmt.Errorf("%s: passage %v is not an object", q.Name, item)
		}
		start, okS := intField(obj["start"])
		end, okE := intField(obj["end"])
		typ, _ := obj["type"].(string)
		k := indexOf(q.Options, typ)
		switch {
		case !okS || !okE:
			return nil, false, fmt.Errorf("%s: passage %v: start and end must be integers", q.Name, obj)
		case k < 0:
			return nil, false, fmt.Errorf("%s: unknown type %q", q.Name, typ)
		case start < 0 || end <= start || end > len(text):
			return nil, false, fmt.Errorf("%s: passage [%d, %d) out of a text of %d bytes", q.Name, start, end, len(text))
		}
		out = append(out, goldSpan{start, end, k})
	}
	sort.Slice(out, func(a, b int) bool { return out[a].start < out[b].start })
	for i := 1; i < len(out); i++ {
		if out[i].start < out[i-1].end {
			return nil, false, fmt.Errorf("%s: passages [%d, %d) and [%d, %d) overlap", q.Name,
				out[i-1].start, out[i-1].end, out[i].start, out[i].end)
		}
	}
	return out, true, nil
}

func intField(v any) (int, bool) {
	f, ok := v.(float64)
	if !ok || f != math.Trunc(f) {
		return 0, false
	}
	return int(f), true
}

// Tags: 0 is outside, 1+2k the beginning of a passage of type k, 2+2k its
// inside.
func spanTagCount(types int) int { return 1 + 2*types }
func tagType(tag int) int        { return (tag - 1) / 2 }
func isInside(tag int) bool      { return tag > 0 && tag%2 == 0 }

// tokenCore is the span of a token without the space it carries in front
// ("▁world" covers " world"). A token made only of spaces keeps its span.
func tokenCore(text string, o tokenizer.Offset) (int, int) {
	s := o.Start
	for s < o.End {
		r, size := utf8.DecodeRuneInString(text[s:o.End])
		if !unicode.IsSpace(r) {
			break
		}
		s += size
	}
	if s == o.End {
		return o.Start, o.End
	}
	return s, o.End
}

// tagTokens gives each token its BIO tag: a token belongs to a passage if
// its core overlaps it, and the first one begins it.
func tagTokens(text string, offs []tokenizer.Offset, spans []goldSpan) []int8 {
	tags := make([]int8, len(offs))
	j, prev := 0, -1
	for i, o := range offs {
		s, e := tokenCore(text, o)
		for j < len(spans) && spans[j].end <= s {
			j++
		}
		if j < len(spans) && spans[j].start < e && s < e {
			if prev == j {
				tags[i] = int8(2 + 2*spans[j].k)
			} else {
				tags[i] = int8(1 + 2*spans[j].k)
			}
			prev = j
			continue
		}
		prev = -1
	}
	return tags
}

// decodeSpans picks the most probable consistent tag sequence from the
// per-token log-probabilities and returns its passages.
func decodeSpans(q Question, text string, offs []tokenizer.Offset, logp [][]float64, bias float64) []Span {
	n := len(logp)
	if n == 0 {
		return nil
	}
	C := len(logp[0])
	// The path is chosen on the biased scores, the confidences read the
	// model's probabilities.
	score := func(t, c int) float64 {
		if c > 0 {
			return logp[t][c] + bias
		}
		return logp[t][c]
	}
	allowed := func(prev, cur int) bool {
		if !isInside(cur) {
			return true
		}
		return prev > 0 && tagType(prev) == tagType(cur)
	}
	cur := make([]float64, C)
	next := make([]float64, C)
	back := make([][]int16, n)
	for c := range C {
		cur[c] = score(0, c)
		if isInside(c) {
			cur[c] = math.Inf(-1)
		}
	}
	for t := 1; t < n; t++ {
		back[t] = make([]int16, C)
		for c := range C {
			best, arg := math.Inf(-1), 0
			for p := range C {
				if allowed(p, c) && cur[p] > best {
					best, arg = cur[p], p
				}
			}
			next[c] = best + score(t, c)
			back[t][c] = int16(arg)
		}
		cur, next = next, cur
	}
	tags := make([]int, n)
	for c := range C {
		if cur[c] > cur[tags[n-1]] {
			tags[n-1] = c
		}
	}
	for t := n - 1; t > 0; t-- {
		tags[t-1] = int(back[t][tags[t]])
	}

	var out []Span
	for t := 0; t < n; {
		if tags[t] == 0 {
			t++
			continue
		}
		k := tagType(tags[t])
		start, _ := tokenCore(text, offs[t])
		end := offs[t].End
		conf := math.Exp(logp[t][tags[t]])
		u := t + 1
		for u < n && isInside(tags[u]) {
			end = offs[u].End
			conf += math.Exp(logp[u][tags[u]])
			u++
		}
		start, end = trimPunct(text, start, end)
		out = append(out, Span{Type: q.Options[k], Start: start, End: end, Text: text[start:end], Confidence: conf / float64(u-t)})
		t = u
	}
	return out
}

// trimPunct removes from a passage the opening and closing punctuation
// that BPE glues to its edge tokens: "Lamy," is one token, the comma is
// not part of the name. Dots stay ("J.-P.", "Inc."), so does a closing
// bracket opened inside the passage, and so does a passage made only of
// punctuation.
func trimPunct(text string, start, end int) (int, int) {
	s, e := start, end
	for s < e {
		r, size := utf8.DecodeRuneInString(text[s:e])
		if !strings.ContainsRune(openPunct, r) && !unicode.IsSpace(r) {
			break
		}
		s += size
	}
	for s < e {
		r, size := utf8.DecodeLastRuneInString(text[s:e])
		if !unicode.IsSpace(r) && (!strings.ContainsRune(closePunct, r) || paired(text[s:e-size], r)) {
			break
		}
		e -= size
	}
	if s == e {
		return start, end
	}
	return s, e
}

// paired reports whether a closing bracket ends a pair opened in the
// passage, as in "Mme Martin (née Roux)": it then stays.
func paired(span string, closing rune) bool {
	open, ok := pairs[closing]
	if !ok {
		return false
	}
	return strings.Count(span, string(open)) > strings.Count(span, string(closing))
}

var pairs = map[rune]rune{')': '(', ']': '[', '}': '{', '»': '«', '›': '‹', '”': '“'}

const (
	openPunct  = "([{«‹“‘\"'¿¡"
	closePunct = ",;:!?)]}»›”’\"'"
)

// logSoftmax divides z by the temperature and normalizes it.
func logSoftmax(z []float64, temperature float64) []float64 {
	if temperature <= 0 {
		temperature = 1
	}
	mx := math.Inf(-1)
	for _, v := range z {
		mx = max(mx, v/temperature)
	}
	var sum float64
	for _, v := range z {
		sum += math.Exp(v/temperature - mx)
	}
	out := make([]float64, len(z))
	lse := mx + math.Log(sum)
	for i, v := range z {
		out[i] = v/temperature - lse
	}
	return out
}

// window is a slice [start, end) of a text's tokens read in one pass;
// the tokens [keepFrom, keepTo) are tagged from it.
type window struct{ start, end, keepFrom, keepTo int }

// planWindows cuts n tokens into windows of at most size tokens that
// overlap by a quarter. Each token is kept by the window where it is
// furthest from an edge: the boundaries fall in the middle of the
// overlaps. The last window ends on the last token.
func planWindows(n, size int) []window {
	if n <= size {
		return []window{{0, n, 0, n}}
	}
	stride := max(size-size/4, 1)
	var ws []window
	for start := 0; ; start += stride {
		if start+size >= n {
			ws = append(ws, window{start: max(n-size, 0), end: n})
			break
		}
		ws = append(ws, window{start: start, end: start + size})
	}
	ws[0].keepFrom = 0
	for i := 1; i < len(ws); i++ {
		mid := (ws[i].start + ws[i-1].end) / 2
		ws[i-1].keepTo, ws[i].keepFrom = mid, mid
	}
	ws[len(ws)-1].keepTo = n
	return ws
}

// hasSpans reports whether the schema has a Spans question.
func (m *Model) hasSpans() bool {
	for _, q := range m.schema {
		if q.Kind == Spans {
			return true
		}
	}
	return false
}

// frame returns the special tokens the template puts around a text.
func (m *Model) frame() (prefix, suffix []int32) {
	ids := m.tok.Encode("")
	return ids[:len(ids)-1], ids[len(ids)-1:]
}

// windowSize is the number of text tokens a window holds.
func (m *Model) windowSize() int {
	prefix, suffix := m.frame()
	return max(m.maxLen-len(prefix)-len(suffix), 1)
}

// inference holds what the model computed for one input: the logits of
// the questions on the pooled vector (nil for Spans questions) and, for
// Spans questions, the logits of every token of the text, whose spans are
// offs.
type inference struct {
	pooled [][]float64
	tokens [][][]float64
	offs   []tokenizer.Offset
}

// infer computes the logits of every question for each input.
func (m *Model) infer(ctx context.Context, inputs []Input, batchSize int) ([]inference, error) {
	if !m.hasSpans() {
		ids, err := m.tokenizeAll(inputs)
		if err != nil {
			return nil, err
		}
		out := make([]inference, len(inputs))
		err = m.forEachPooled(ctx, ids, batchSize, func(i int, x []float32) {
			out[i].pooled = make([][]float64, len(m.heads))
			for qi, h := range m.heads {
				out[i].pooled[qi] = h.logits(x)
			}
		})
		return out, err
	}

	type job struct {
		in  int
		w   window
		seq []int32
	}
	prefix, suffix := m.frame()
	size := m.windowSize()
	out := make([]inference, len(inputs))
	var jobs []job
	for i, in := range inputs {
		if in.Context != "" {
			return nil, fmt.Errorf("indecis: a model with Spans questions reads no context")
		}
		ids, offs := m.tok.AppendOffsets(nil, nil, in.Text)
		out[i].offs = offs
		out[i].pooled = make([][]float64, len(m.heads))
		out[i].tokens = make([][][]float64, len(m.heads))
		for qi, h := range m.heads {
			if h.q.Kind == Spans {
				out[i].tokens[qi] = make([][]float64, len(ids))
			}
		}
		for _, w := range planWindows(len(ids), size) {
			seq := append(append(append([]int32(nil), prefix...), ids[w.start:w.end]...), suffix...)
			jobs = append(jobs, job{i, w, seq})
		}
	}

	H := m.enc.Cfg.Hidden
	for start := 0; start < len(jobs); start += batchSize {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		group := jobs[start:min(start+batchSize, len(jobs))]
		seqs := make([][]int32, len(group))
		for j, jb := range group {
			seqs[j] = jb.seq
		}
		states, err := m.enc.EncodeTokens(seqs)
		if err != nil {
			return nil, err
		}
		row := 0
		for _, jb := range group {
			res := &out[jb.in]
			rows := states[row*H : (row+len(jb.seq))*H]
			row += len(jb.seq)
			if jb.w.start == 0 {
				x := meanRows(rows, len(jb.seq), H)
				for qi, h := range m.heads {
					if h.q.Kind != Spans {
						res.pooled[qi] = h.logits(x)
					}
				}
			}
			for t := jb.w.keepFrom; t < jb.w.keepTo; t++ {
				r := len(prefix) + t - jb.w.start
				x := rows[r*H : (r+1)*H]
				for qi, h := range m.heads {
					if h.q.Kind == Spans {
						res.tokens[qi][t] = h.logits(x)
					}
				}
			}
		}
	}
	return out, nil
}

// meanRows averages n rows of width H, like the encoder's pooling.
func meanRows(rows []float32, n, H int) []float32 {
	acc := make([]float64, H)
	for t := range n {
		for k, v := range rows[t*H : (t+1)*H] {
			acc[k] += float64(v)
		}
	}
	x := make([]float32, H)
	for k := range acc {
		x[k] = float32(acc[k] / float64(max(n, 1)))
	}
	return x
}

// answerSpans decodes the passages of a Spans question.
func (m *Model) answerSpans(qi int, text string, r inference) Answer {
	q := m.schema[qi]
	logp := make([][]float64, len(r.tokens[qi]))
	for t, z := range r.tokens[qi] {
		logp[t] = logSoftmax(z, m.temps[qi])
	}
	spans := decodeSpans(q, text, r.offs, logp, m.spanBias[q.Name])
	if spans == nil {
		spans = []Span{}
	}
	return Answer{Question: q.Name, Kind: Spans, Spans: spans}
}

// spanWindows cuts the encoded examples of a model with Spans questions
// into training windows. The other questions train on the first window
// only, the one they read in inference.
func (m *Model) spanWindows(data []encoded) []encoded {
	prefix, suffix := m.frame()
	size := m.windowSize()
	var out []encoded
	for _, e := range data {
		for wi, w := range planWindows(len(e.ids), size) {
			win := encoded{targets: make([][]float64, len(m.heads)), tags: make([][]int8, len(m.heads))}
			win.ids = append(append(append([]int32(nil), prefix...), e.ids[w.start:w.end]...), suffix...)
			if wi == 0 {
				copy(win.targets, e.targets)
			}
			for qi, tags := range e.tags {
				if tags == nil {
					continue
				}
				t := make([]int8, len(win.ids))
				for i := range t {
					t[i] = -1
				}
				copy(t[len(prefix):], tags[w.start:w.end])
				win.tags[qi] = t
			}
			out = append(out, win)
		}
	}
	return out
}

// spanLossGrad adds to dHidden the gradient of the tagging loss of the
// Spans heads, each averaged over the tagged tokens of the batch, and
// returns the loss. hidden is [B·T, H], the batch padded to T tokens.
func (m *Model) spanLossGrad(batch []*encoded, hidden, dHidden []float32, T int, dropout float64, rng interface{ Float64() float64 }) float64 {
	H := m.enc.Cfg.Hidden
	var loss float64
	keep := float32(1 / (1 - dropout))
	x := make([]float32, H)
	mask := make([]float32, H)
	dx := make([]float32, H)
	for qi, h := range m.heads {
		if h.q.Kind != Spans {
			continue
		}
		count := 0
		for _, e := range batch {
			for _, tag := range e.tags[qi] {
				if tag >= 0 {
					count++
				}
			}
		}
		if count == 0 {
			continue
		}
		scale := 1 / float64(count)
		for bi, e := range batch {
			for t, tag := range e.tags[qi] {
				if tag < 0 {
					continue
				}
				row := hidden[(bi*T+t)*H : (bi*T+t+1)*H]
				for i := range x {
					mask[i] = 1
					if dropout > 0 {
						mask[i] = 0
						if rng.Float64() >= dropout {
							mask[i] = keep
						}
					}
					x[i] = row[i] * mask[i]
				}
				z := h.logits(x)
				p := softmax(z)
				loss -= math.Log(max(p[tag], 1e-300)) * scale
				dz := p
				dz[tag]--
				for k := range dz {
					dz[k] *= scale
				}
				clear(dx)
				h.backward(x, dz, dx)
				d := dHidden[(bi*T+t)*H : (bi*T+t+1)*H]
				for i := range d {
					d[i] += dx[i] * mask[i]
				}
			}
		}
	}
	return loss
}

// SpanMetrics measures the passages of one type, or of all types.
type SpanMetrics struct {
	Type string `json:"type,omitempty"`
	// Gold is the number of labeled passages, Found the number found.
	Gold      int     `json:"gold"`
	Found     int     `json:"found"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
	// F2 weighs recall twice as much as precision: for personal data, a
	// missed passage costs more than one too many.
	F2 float64 `json:"f2"`
}

func (s *SpanMetrics) finish(tp int) {
	s.Precision = safeDiv(float64(tp), float64(s.Found))
	s.Recall = safeDiv(float64(tp), float64(s.Gold))
	s.F1 = safeDiv(2*s.Precision*s.Recall, s.Precision+s.Recall)
	s.F2 = safeDiv(5*s.Precision*s.Recall, 4*s.Precision+s.Recall)
}

// spanMetrics compares the passages found with the labeled ones: a
// passage counts if its type and its bounds are exact. Accuracy and NLL
// are per token.
func (m *Model) spanMetrics(qi int, examples []encoded, texts []string, res []inference) Metrics {
	q := m.schema[qi]
	mt := Metrics{Question: q.Name, Kind: Spans}
	all := SpanMetrics{}
	per := make([]SpanMetrics, len(q.Options))
	tpAll, tp := 0, make([]int, len(q.Options))
	var conf []float64
	var correct []bool
	tokens := 0
	for i, e := range examples {
		if e.tags == nil || e.tags[qi] == nil {
			continue
		}
		mt.N++
		a := m.answerSpans(qi, texts[i], res[i])
		gold := map[[3]int]bool{}
		for _, g := range e.spans[qi] {
			gold[[3]int{g.start, g.end, g.k}] = true
			per[g.k].Gold++
			all.Gold++
		}
		for _, s := range a.Spans {
			k := indexOf(q.Options, s.Type)
			per[k].Found++
			all.Found++
			if key := [3]int{s.Start, s.End, k}; gold[key] {
				delete(gold, key) // a gold passage matches once
				tp[k]++
				tpAll++
			}
		}
		for t, tag := range e.tags[qi] {
			logp := logSoftmax(res[i].tokens[qi][t], m.temps[qi])
			best := 0
			for c, v := range logp {
				if v > logp[best] {
					best = c
				}
			}
			mt.NLL -= logp[tag]
			conf = append(conf, math.Exp(logp[best]))
			correct = append(correct, best == int(tag))
			if best == int(tag) {
				mt.Accuracy++
			}
			tokens++
		}
	}
	if tokens > 0 {
		mt.Accuracy /= float64(tokens)
		mt.NLL /= float64(tokens)
		mt.ECE = ece(conf, correct, 10)
	}
	all.finish(tpAll)
	mt.Precision, mt.Recall, mt.F1, mt.F2 = all.Precision, all.Recall, all.F1, all.F2
	for k, s := range per {
		s.Type = q.Options[k]
		s.finish(tp[k])
		mt.Types = append(mt.Types, s)
	}
	return mt
}

// calibrateSpans tunes the temperature of a Spans question on the token
// tags.
func (m *Model) calibrateSpans(qi int, examples []encoded, res []inference) {
	var zs [][]float64
	var ts []int
	for i, e := range examples {
		if e.tags == nil || e.tags[qi] == nil {
			continue
		}
		for t, tag := range e.tags[qi] {
			zs = append(zs, res[i].tokens[qi][t])
			ts = append(ts, int(tag))
		}
	}
	if len(zs) == 0 {
		return
	}
	nll := func(T float64) float64 {
		var sum float64
		for i, z := range zs {
			sum -= logSoftmax(z, T)[ts[i]]
		}
		return sum / float64(len(zs))
	}
	m.temps[qi] = goldenLog(nll, 0.05, 20)
}
