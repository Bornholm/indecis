// Package tokenizer reproduces in pure Go the BPE tokenizer of the Gemma
// model family, as a Hugging Face tokenizer.json file describes it.
//
// The contract is exact parity with the Rust `tokenizers` library: the same
// ids for the same text. A different id means the model reads another text
// than the one it was trained on. The pipeline is therefore reproduced step
// by step, quirks included:
//
//  1. added tokens (special or not) are found in the raw text, with
//     leftmost-longest matching, before any normalization;
//  2. each remaining segment is normalized (space becomes ▁), gets a leading
//     ▁ if it has none, then is split before each ▁;
//  3. each piece goes through BPE: merges by increasing rank, the leftmost
//     one on equal ranks, and byte fallback for characters missing from the
//     vocabulary;
//  4. the template adds <bos> and <eos>.
//
// Only this configuration is accepted: a tokenizer.json that differs is
// rejected at load time instead of being tokenized approximately.
package tokenizer

import (
	"bufio"
	"bytes"
	"container/heap"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

const metaspace = "▁"

// Tokenizer is safe for concurrent use.
type Tokenizer struct {
	vocab  *strTable        // BPE vocabulary strings, by id
	names  map[int32]string // added tokens, if they differ from the vocabulary
	size   int              // number of ids
	merges mergeTable
	bytes  [256]int32 // id of <0xXX>, -1 if absent
	unk    int32
	bos    int32
	eos    int32
	// raw is the original Gemma pipeline (SigLIP 2): no leading ▁, no split
	// before BPE, and templates without <bos>.
	raw     bool
	pad     int32
	added   map[byte][]addedToken // by first byte, longest to shortest
	cacheMu sync.RWMutex
	cache   map[string][]int32
}

type merge struct {
	rank int32
	id   int32
}

type addedToken struct {
	content string
	id      int32
	lstrip  bool
	rstrip  bool
}

const maxCache = 1 << 16

// Load reads a tokenizer.json.
func Load(path string) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return read(bufio.NewReaderSize(f, 1<<16))
}

var (
	sharedMu sync.Mutex
	shared   = map[[sha256.Size]byte]*Tokenizer{}
)

// LoadShared is Load, but two files with the same content give the same
// Tokenizer: several models derived from the same backbone keep only one
// in memory (~12 MB each). A Tokenizer is immutable and safe for
// concurrent access, including its word cache.
func LoadShared(path string) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	h := sha256.New()
	t, err := read(bufio.NewReaderSize(io.TeeReader(f, h), 1<<16))
	if err != nil {
		return nil, err
	}
	var key [sha256.Size]byte
	h.Sum(key[:0])
	sharedMu.Lock()
	defer sharedMu.Unlock()
	if prev, ok := shared[key]; ok {
		return prev, nil
	}
	shared[key] = t
	return t, nil
}

// Parse reads the content of a tokenizer.json.
func Parse(data []byte) (*Tokenizer, error) { return read(bytes.NewReader(data)) }

type fileJSON struct {
	AddedTokens []struct {
		ID         int32  `json:"id"`
		Content    string `json:"content"`
		SingleWord bool   `json:"single_word"`
		LStrip     bool   `json:"lstrip"`
		RStrip     bool   `json:"rstrip"`
		Normalized bool   `json:"normalized"`
		Special    bool   `json:"special"`
	} `json:"added_tokens"`
	Normalizer *struct {
		Type    string `json:"type"`
		Pattern struct {
			String string `json:"String"`
		} `json:"pattern"`
		Content string `json:"content"`
	} `json:"normalizer"`
	PreTokenizer *struct {
		Type          string `json:"type"`
		Replacement   string `json:"replacement"`
		PrependScheme string `json:"prepend_scheme"`
		Split         bool   `json:"split"`
		Pattern       struct {
			String string `json:"String"`
		} `json:"pattern"`
		Behavior string `json:"behavior"`
		Invert   bool   `json:"invert"`
	} `json:"pre_tokenizer"`
	PostProcessor *struct {
		Type   string          `json:"type"`
		Single []templatePiece `json:"single"`
		Pair   []templatePiece `json:"pair"`
	} `json:"post_processor"`
	Model struct {
		Type                    string   `json:"type"`
		Dropout                 *float64 `json:"dropout"`
		UnkToken                string   `json:"unk_token"`
		ContinuingSubwordPrefix *string  `json:"continuing_subword_prefix"`
		EndOfWordSuffix         *string  `json:"end_of_word_suffix"`
		FuseUnk                 bool     `json:"fuse_unk"`
		ByteFallback            bool     `json:"byte_fallback"`
		IgnoreMerges            bool     `json:"ignore_merges"`
	} `json:"model"` // vocab and merges are streamed (see decode)
}

type templatePiece struct {
	SpecialToken *struct {
		ID string `json:"id"`
	} `json:"SpecialToken"`
	Sequence *struct {
		ID string `json:"id"`
	} `json:"Sequence"`
}

// matches checks a template: each element is a special token (named)
// or a sequence (empty).
func matches(pieces []templatePiece, want ...string) bool {
	if len(pieces) != len(want) {
		return false
	}
	for i, w := range want {
		p := pieces[i]
		if w == "" {
			if p.Sequence == nil {
				return false
			}
		} else if p.SpecialToken == nil || p.SpecialToken.ID != w {
			return false
		}
	}
	return true
}

func read(r io.Reader) (*Tokenizer, error) {
	f, byID, pairs, err := decode(r)
	if err != nil {
		return nil, err
	}
	if err := checkSupported(&f); err != nil {
		return nil, err
	}
	raw, _ := pipeline(&f)

	t := &Tokenizer{
		raw:   raw,
		names: make(map[int32]string),
		added: make(map[byte][]addedToken),
		cache: make(map[string][]int32),
	}
	for _, a := range f.AddedTokens {
		if a.ID < 0 {
			return nil, fmt.Errorf("tokenizer: id %d negative", a.ID)
		}
		for int(a.ID) >= len(byID) {
			byID = append(byID, "")
		}
	}
	vocab := newStrTable(byID)
	t.vocab = vocab
	t.size = len(byID)

	for _, a := range f.AddedTokens {
		if a.Normalized || a.SingleWord || a.Content == "" {
			return nil, fmt.Errorf("tokenizer: added token %q: normalized/single_word not supported", a.Content)
		}
		if byID[a.ID] != a.Content {
			t.names[a.ID] = a.Content
		}
		first := a.Content[0]
		t.added[first] = append(t.added[first], addedToken{content: a.Content, id: a.ID, lstrip: a.LStrip, rstrip: a.RStrip})
	}
	for k := range t.added {
		sort.SliceStable(t.added[k], func(i, j int) bool { return len(t.added[k][i].content) > len(t.added[k][j].content) })
	}

	var ok bool
	if t.unk, ok = t.lookup(f.Model.UnkToken); !ok {
		return nil, fmt.Errorf("tokenizer: unk token %q missing", f.Model.UnkToken)
	}
	if t.bos, ok = t.lookup("<bos>"); !ok {
		return nil, fmt.Errorf("tokenizer: <bos> missing")
	}
	if t.eos, ok = t.lookup("<eos>"); !ok {
		return nil, fmt.Errorf("tokenizer: <eos> missing")
	}
	if t.pad, ok = t.lookup("<pad>"); !ok {
		t.pad = 0
	}
	for i := range t.bytes {
		id, ok := t.vocab.lookup(fmt.Sprintf("<0x%02X>", i))
		if !ok {
			id = -1
		}
		t.bytes[i] = id
	}

	entries := make([]mergeEntry, 0, len(pairs))
	for rank, p := range pairs {
		a, okA := vocab.lookup(p[0])
		b, okB := vocab.lookup(p[1])
		id, okM := vocab.lookup(p[0] + p[1])
		if !okA || !okB || !okM {
			return nil, fmt.Errorf("tokenizer: merge %q %q out of vocabulary", p[0], p[1])
		}
		entries = append(entries, mergeEntry{key: pairKey(a, b), m: merge{rank: int32(rank), id: id}})
	}
	t.merges = newMergeTable(entries)
	return t, nil
}

func checkSupported(f *fileJSON) error {
	m := f.Model
	switch {
	case m.Type != "BPE":
		return fmt.Errorf("tokenizer: model %q not supported", m.Type)
	case m.Dropout != nil && *m.Dropout != 0:
		return fmt.Errorf("tokenizer: BPE dropout not supported")
	case m.ContinuingSubwordPrefix != nil && *m.ContinuingSubwordPrefix != "",
		m.EndOfWordSuffix != nil && *m.EndOfWordSuffix != "":
		return fmt.Errorf("tokenizer: subword prefix/suffix not supported")
	case !m.ByteFallback || !m.FuseUnk || m.IgnoreMerges:
		return fmt.Errorf("tokenizer: expected byte_fallback=true, fuse_unk=true, ignore_merges=false")
	}
	if n := f.Normalizer; n != nil && (n.Type != "Replace" || n.Pattern.String != " " || n.Content != metaspace) {
		return fmt.Errorf("tokenizer: normalizer not supported")
	}
	_, err := pipeline(f)
	return err
}

// pipeline recognizes the two supported pipelines and reports whether it
// is the raw one:
//   - Metaspace(▁, always, split), templates <bos> A <eos> and
//     <bos> A <eos> B <eos> (Gemma 3, bekko);
//   - raw: Split on " " (a no-op, since the normalizer already replaced
//     spaces), templates A <eos> and A <eos> B <eos> (the original Gemma
//     tokenizer, SigLIP 2).
func pipeline(f *fileJSON) (raw bool, err error) {
	p, pp := f.PreTokenizer, f.PostProcessor
	if pp == nil || pp.Type != "TemplateProcessing" {
		return false, fmt.Errorf("tokenizer: expected a TemplateProcessing post-processor")
	}
	switch {
	case p != nil && p.Type == "Metaspace" && p.Replacement == metaspace && p.PrependScheme == "always" && p.Split:
		if !matches(pp.Single, "<bos>", "", "<eos>") || !matches(pp.Pair, "<bos>", "", "<eos>", "", "<eos>") {
			return false, fmt.Errorf("tokenizer: expected templates: <bos> A <eos> and <bos> A <eos> B <eos>")
		}
		return false, nil
	case p != nil && p.Type == "Split" && p.Pattern.String == " " && p.Behavior == "MergedWithPrevious" && !p.Invert &&
		f.Normalizer != nil:
		if !matches(pp.Single, "", "<eos>") || !matches(pp.Pair, "", "<eos>", "", "<eos>") {
			return false, fmt.Errorf("tokenizer: expected templates: A <eos> and A <eos> B <eos>")
		}
		return true, nil
	}
	return false, fmt.Errorf("tokenizer: expected pre-tokenizer: Metaspace(▁, always, split) or Split(\" \", merged with previous)")
}

func pairKey(a, b int32) uint64 { return uint64(uint32(a))<<32 | uint64(uint32(b)) }

func (t *Tokenizer) lookup(s string) (int32, bool) {
	if id, ok := t.vocab.lookup(s); ok {
		return id, true
	}
	for _, list := range t.added {
		for _, a := range list {
			if a.content == s {
				return a.id, true
			}
		}
	}
	return 0, false
}

// VocabSize is the number of possible ids.
func (t *Tokenizer) VocabSize() int { return t.size }

// Token returns the string for an id.
func (t *Tokenizer) Token(id int32) string {
	if id < 0 || int(id) >= t.size {
		return ""
	}
	if s, ok := t.names[id]; ok {
		return s
	}
	return t.vocab.str(id)
}

// PadID, BosID, EosID expose the special ids.
func (t *Tokenizer) PadID() int32 { return t.pad }
func (t *Tokenizer) BosID() int32 { return t.bos }
func (t *Tokenizer) EosID() int32 { return t.eos }

// Encode tokenizes text and ends the result with <eos>, preceded by <bos>
// except for the raw pipeline.
func (t *Tokenizer) Encode(text string) []int32 {
	var ids []int32
	if !t.raw {
		ids = append(ids, t.bos)
	}
	ids = t.appendText(ids, text)
	return append(ids, t.eos)
}

// EncodeMax tokenizes like Encode and truncates on the right to not
// exceed maxLen ids, keeping the final <eos>. maxLen <= 2 gives
// [<bos>, <eos>].
func (t *Tokenizer) EncodeMax(text string, maxLen int) []int32 {
	ids := t.Encode(text)
	if maxLen < 2 {
		maxLen = 2
	}
	if len(ids) <= maxLen {
		return ids
	}
	ids = ids[:maxLen]
	ids[maxLen-1] = t.eos
	return ids
}

// EncodePair tokenizes a (context, text) pair with the template
// <bos> context <eos> text <eos>, the one used by the reference library.
//
// Beyond maxLen ids (0: no limit), truncation sacrifices the context
// before the text: the text is what is being judged, the context is what
// sheds light on it. The context keeps at least a third of the budget if
// it is long enough, and it is its beginning that is kept: a system
// prompt sets the assistant's role up front. The framing is Encode's, with
// <eos> between the two.
func (t *Tokenizer) EncodePair(context, text string, maxLen int) []int32 {
	a := t.appendText(nil, context)
	b := t.appendText(nil, text)
	if maxLen > 0 {
		budget := max(maxLen-3, 0)
		if len(a)+len(b) > budget {
			keepA := min(len(a), budget/3)
			keepB := min(len(b), budget-keepA)
			keepA = min(len(a), budget-keepB)
			a, b = a[:keepA], b[:keepB]
		}
	}
	ids := make([]int32, 0, len(a)+len(b)+3)
	if !t.raw {
		ids = append(ids, t.bos)
	}
	ids = append(ids, a...)
	ids = append(ids, t.eos)
	ids = append(ids, b...)
	return append(ids, t.eos)
}

// appendText splits text around added tokens and tokenizes the segments.
func (t *Tokenizer) appendText(ids []int32, text string) []int32 {
	t.walk(text, func(id int32) { ids = append(ids, id) }, func(w string) { ids = t.appendWord(ids, w) })
	return ids
}

// walk splits text: added receives each added token found in the raw
// text, word each piece to pass to BPE, in order.
func (t *Tokenizer) walk(text string, added func(int32), word func(string)) {
	start := 0 // start of the current segment
	i := 0
	for i < len(text) {
		a, ok := t.matchAdded(text, i)
		if !ok {
			i++
			continue
		}
		mStart, mEnd := i, i+len(a.content)
		if a.lstrip {
			for mStart > start {
				r, size := utf8.DecodeLastRuneInString(text[start:mStart])
				if !unicode.IsSpace(r) {
					break
				}
				mStart -= size
			}
		}
		if a.rstrip {
			for mEnd < len(text) {
				r, size := utf8.DecodeRuneInString(text[mEnd:])
				if !unicode.IsSpace(r) {
					break
				}
				mEnd += size
			}
		}
		t.walkSegment(text[start:mStart], word)
		added(a.id)
		start, i = mEnd, mEnd
	}
	t.walkSegment(text[start:], word)
}

// matchAdded returns the longest added token that starts at text[i].
func (t *Tokenizer) matchAdded(text string, i int) (addedToken, bool) {
	for _, a := range t.added[text[i]] {
		if strings.HasPrefix(text[i:], a.content) {
			return a, true
		}
	}
	return addedToken{}, false
}

// walkSegment normalizes, pre-tokenizes and passes each piece to word.
func (t *Tokenizer) walkSegment(seg string, word func(string)) {
	if seg == "" {
		return
	}
	s := strings.ReplaceAll(seg, " ", metaspace)
	if t.raw {
		word(s) // no leading ▁, no split: the whole segment goes to BPE
		return
	}
	if !strings.HasPrefix(s, metaspace) {
		s = metaspace + s
	}
	// Split before each ▁, the ▁ staying attached to the next piece.
	for len(s) > 0 {
		next := strings.Index(s[len(metaspace):], metaspace)
		if next < 0 {
			word(s)
			break
		}
		cut := next + len(metaspace)
		if !strings.HasPrefix(s, metaspace) {
			// Only the first piece can start without ▁; that is not the
			// case here since a ▁ was prepended.
			cut = strings.Index(s, metaspace)
		}
		word(s[:cut])
		s = s[cut:]
	}
}

// Trace calls visit for every token that splitting text produces or
// passes through: added tokens, BPE starting symbols (characters, bytes)
// and each intermediate merge. A vocabulary that keeps all these tokens
// splits text exactly like this one does (see Prune).
func (t *Tokenizer) Trace(text string, visit func(id int32)) {
	t.walk(text, visit, func(w string) { t.bpeVisit(w, visit) })
}

func (t *Tokenizer) appendWord(ids []int32, word string) []int32 {
	t.cacheMu.RLock()
	cached, ok := t.cache[word]
	t.cacheMu.RUnlock()
	if ok {
		return append(ids, cached...)
	}
	out := t.bpe(word)
	t.cacheMu.Lock()
	if len(t.cache) >= maxCache {
		clear(t.cache)
	}
	t.cache[word] = out
	t.cacheMu.Unlock()
	return append(ids, out...)
}

type symbol struct {
	id         int32
	prev, next int
	merged     bool // absorbed by its left neighbor
}

// bpe applies merges to a word, in the order used by the reference
// library: increasing rank, then increasing position.
func (t *Tokenizer) bpe(word string) []int32 { return t.bpeVisit(word, nil) }

// bpeVisit is bpe, reporting to visit (if provided) each starting symbol
// and each merge applied.
func (t *Tokenizer) bpeVisit(word string, visit func(int32)) []int32 {
	syms := make([]symbol, 0, len(word))
	lastUnk := false
	for _, r := range word {
		var rb [utf8.UTFMax]byte
		if id, ok := t.vocab.lookupBytes(rb[:utf8.EncodeRune(rb[:], r)]); ok {
			syms = append(syms, symbol{id: id})
			lastUnk = false
			continue
		}
		var buf [utf8.UTFMax]byte
		n := utf8.EncodeRune(buf[:], r)
		fallback := true
		for _, b := range buf[:n] {
			if t.bytes[b] < 0 {
				fallback = false
				break
			}
		}
		if fallback {
			for _, b := range buf[:n] {
				syms = append(syms, symbol{id: t.bytes[b]})
			}
			lastUnk = false
			continue
		}
		if !lastUnk { // fuse_unk
			syms = append(syms, symbol{id: t.unk})
		}
		lastUnk = true
	}
	for i := range syms {
		syms[i].prev, syms[i].next = i-1, i+1
		if visit != nil {
			visit(syms[i].id)
		}
	}
	if n := len(syms); n > 0 {
		syms[n-1].next = -1
	}

	q := make(mergeQueue, 0, len(syms))
	for i := 0; i+1 < len(syms); i++ {
		if m, ok := t.merges.get(pairKey(syms[i].id, syms[i+1].id)); ok {
			q = append(q, candidate{pos: i, rank: m.rank, id: m.id})
		}
	}
	heap.Init(&q)
	for q.Len() > 0 {
		top := heap.Pop(&q).(candidate)
		cur := &syms[top.pos]
		if cur.merged || cur.next < 0 {
			continue
		}
		right := syms[cur.next]
		// Stale entry: the pair changed since it was added.
		if m, ok := t.merges.get(pairKey(cur.id, right.id)); !ok || m.id != top.id {
			continue
		}
		cur.id = top.id
		if visit != nil {
			visit(top.id)
		}
		syms[cur.next].merged = true
		cur.next = right.next
		if right.next >= 0 {
			syms[right.next].prev = top.pos
		}
		if cur.prev >= 0 {
			if m, ok := t.merges.get(pairKey(syms[cur.prev].id, cur.id)); ok {
				heap.Push(&q, candidate{pos: cur.prev, rank: m.rank, id: m.id})
			}
		}
		if cur.next >= 0 {
			if m, ok := t.merges.get(pairKey(cur.id, syms[cur.next].id)); ok {
				heap.Push(&q, candidate{pos: top.pos, rank: m.rank, id: m.id})
			}
		}
	}

	out := make([]int32, 0, len(syms))
	for i := 0; i >= 0 && i < len(syms); i = syms[i].next {
		out = append(out, syms[i].id)
	}
	return out
}

type candidate struct {
	pos  int
	rank int32
	id   int32
}

type mergeQueue []candidate

func (q mergeQueue) Len() int { return len(q) }
func (q mergeQueue) Less(i, j int) bool {
	if q[i].rank != q[j].rank {
		return q[i].rank < q[j].rank
	}
	return q[i].pos < q[j].pos
}
func (q mergeQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *mergeQueue) Push(x any)   { *q = append(*q, x.(candidate)) }
func (q *mergeQueue) Pop() any {
	old := *q
	x := old[len(old)-1]
	*q = old[:len(old)-1]
	return x
}
