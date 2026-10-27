// Package charlm is a character n-gram language model, to guide the OCR's
// decoding: the probability of a character given the few before it.
//
// Smoothing is Witten-Bell interpolation: P(c|h) = (C(hc) + T(h)·P(c|h'))
// / (C(h) + T(h)), with C the counts, T(h) the number of distinct
// characters seen after h, and h' the context h without its first
// character; the recursion ends on the uniform distribution over the
// characters seen. Witten-Bell needs only counts, and suits characters
// well.
//
// The model sees characters through Fold: variants that only their shape
// tells apart (straight and curly quotes, dashes, bullets) and digits are
// one class each. It then has no say in what the image alone can decide.
//
// The counts are kept in sorted arrays of 64-bit hashes (12 bytes an
// n-gram on disk), indexed at load by open-addressing tables.
package charlm

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
)

// Model is a character n-gram model.
type Model struct {
	Order int // longest n-gram, context included

	// ngrams: hash of (order, runes) → count, for n-grams of every order.
	ngramKeys []uint64
	ngramVals []uint32
	// contexts: hash of a context → total count after it and number of
	// distinct characters after it.
	ctxKeys  []uint64
	ctxTotal []uint32
	ctxTypes []uint32
	vocab    int // distinct characters: the uniform distribution's support

	// Open-addressing indexes over the sorted keys, built at load: a
	// bisection over millions of keys misses the cache at every step.
	ngramIdx, ctxIdx index
}

// index maps a key to its position in a sorted key array, by linear
// probing in a table twice the array's size.
type index struct {
	slots []uint32 // position+1, 0 for an empty slot
	mask  uint64
}

func newIndex(keys []uint64) index {
	size := uint64(1)
	for size < uint64(2*len(keys)+1) {
		size <<= 1
	}
	ix := index{slots: make([]uint32, size), mask: size - 1}
	for i, k := range keys {
		for s := mix(k) & ix.mask; ; s = (s + 1) & ix.mask {
			if ix.slots[s] == 0 {
				ix.slots[s] = uint32(i + 1)
				break
			}
		}
	}
	return ix
}

func (ix index) find(keys []uint64, k uint64) (int, bool) {
	for s := mix(k) & ix.mask; ; s = (s + 1) & ix.mask {
		p := ix.slots[s]
		if p == 0 {
			return 0, false
		}
		if keys[p-1] == k {
			return int(p - 1), true
		}
	}
}

// mix spreads a key's bits over the table's low bits.
func mix(k uint64) uint64 {
	k ^= k >> 33
	k *= 0xff51afd7ed558ccd
	k ^= k >> 33
	return k
}

func (m *Model) buildIndexes() {
	m.ngramIdx = newIndex(m.ngramKeys)
	m.ctxIdx = newIndex(m.ctxKeys)
}

// Fold maps a character to its class: one per quote kind, dash, bullet,
// and one for all digits (superscripts included); others are themselves.
func Fold(r rune) rune {
	switch r {
	case '\'', '’', '‘', '`', '´':
		return '\''
	case '"', '“', '”', '„', '«', '»':
		return '"'
	case '-', '‐', '–', '—', '−':
		return '-'
	case '•', '·', '*', '◦', '▪', '■', '□', '●', '○', '◆', '◇', '►', '▶', '✓', '✔', '✗', '☐', '☑':
		return '•'
	case '²', '³', '¹', '⁰', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹', '₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉':
		return '0'
	}
	if r >= '0' && r <= '9' {
		return '0'
	}
	return r
}

// hash folds runes into a 64-bit key, its length mixed in: FNV-1a.
func hash(rs []rune) uint64 {
	h := uint64(14695981039346656037) ^ uint64(len(rs))*0x9E3779B97F4A7C15
	for _, r := range rs {
		h ^= uint64(r)
		h *= 1099511628211
	}
	return h
}

// Builder counts n-grams from text.
type Builder struct {
	order  int
	ngrams map[uint64]uint32
	ctx    map[uint64][2]uint32 // total, types
	vocab  map[rune]bool
}

// NewBuilder starts a model of the given order (2 to 8).
func NewBuilder(order int) *Builder {
	return &Builder{order: order, ngrams: map[uint64]uint32{}, ctx: map[uint64][2]uint32{}, vocab: map[rune]bool{}}
}

// Add counts the n-grams of a text (one line or paragraph; its start
// counts as a context of spaces).
func (b *Builder) Add(text string) {
	rs := make([]rune, 0, len(text)+b.order)
	for range b.order - 1 {
		rs = append(rs, ' ')
	}
	for _, r := range text {
		r = Fold(r)
		rs = append(rs, r)
		b.vocab[r] = true
	}
	for i := b.order - 1; i < len(rs); i++ {
		for n := 1; n <= b.order; n++ {
			g := rs[i-n+1 : i+1]
			k := hash(g)
			c := b.ngrams[k]
			b.ngrams[k] = c + 1
			ck := hash(g[:n-1])
			s := b.ctx[ck]
			s[0]++
			if c == 0 {
				s[1]++
			}
			b.ctx[ck] = s
		}
	}
}

// Build prunes the n-grams seen fewer than minCount times (the contexts'
// totals keep them: their mass goes to the lower orders) and returns the
// model.
func (b *Builder) Build(minCount uint32) *Model {
	m := &Model{Order: b.order, vocab: len(b.vocab)}
	keys := make([]uint64, 0, len(b.ngrams))
	for k, c := range b.ngrams {
		if c >= minCount {
			keys = append(keys, k)
		}
	}
	slices.Sort(keys)
	m.ngramKeys = keys
	m.ngramVals = make([]uint32, len(keys))
	for i, k := range keys {
		m.ngramVals[i] = b.ngrams[k]
	}
	// A context seen fewer than minCount times keeps none of its n-grams:
	// it goes too, its characters then drawn from the shorter context.
	ck := make([]uint64, 0, len(b.ctx))
	for k, st := range b.ctx {
		if st[0] >= minCount {
			ck = append(ck, k)
		}
	}
	slices.Sort(ck)
	m.ctxKeys = ck
	m.ctxTotal = make([]uint32, len(ck))
	m.ctxTypes = make([]uint32, len(ck))
	for i, k := range ck {
		s := b.ctx[k]
		m.ctxTotal[i], m.ctxTypes[i] = s[0], s[1]
	}
	m.buildIndexes()
	return m
}

// Prob returns P(c | context), context being the characters before c (only
// the last Order-1 count). Shorter contexts are padded with spaces, as at
// the start of a text.
func (m *Model) Prob(context []rune, c rune) float64 {
	n := m.Order - 1
	var buf [17]rune
	h := buf[:0]
	for i := len(context); i < n; i++ {
		h = append(h, ' ')
	}
	if len(context) > n {
		context = context[len(context)-n:]
	}
	for _, r := range context {
		h = append(h, Fold(r))
	}
	h = append(h, Fold(c))
	return m.prob(h)
}

// prob is P(last rune of g | the others), interpolated down to the uniform
// distribution.
func (m *Model) prob(g []rune) float64 {
	p := 1 / float64(max(1, m.vocab))
	for start := len(g) - 1; start >= 0; start-- {
		gram := g[start:]
		i, ok := m.ctxIdx.find(m.ctxKeys, hash(gram[:len(gram)-1]))
		if !ok {
			continue // unseen context: keep the lower order's estimate
		}
		total, types := float64(m.ctxTotal[i]), float64(m.ctxTypes[i])
		var c float64
		if j, ok := m.ngramIdx.find(m.ngramKeys, hash(gram)); ok {
			c = float64(m.ngramVals[j])
		}
		p = (c + types*p) / (total + types)
	}
	return p
}

// LogProb is log P(c | context).
func (m *Model) LogProb(context []rune, c rune) float64 { return math.Log(m.Prob(context, c)) }

const magic = "CLM3" // CLM1 had no folding, CLM2 fewer bullets and no subscripts

// Save writes the model.
func (m *Model) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	put := func(v any) {
		if err == nil {
			err = binary.Write(w, binary.LittleEndian, v)
		}
	}
	w.WriteString(magic)
	put(uint32(m.Order))
	put(uint32(m.vocab))
	put(uint64(len(m.ngramKeys)))
	put(m.ngramKeys)
	put(m.ngramVals)
	put(uint64(len(m.ctxKeys)))
	put(m.ctxKeys)
	put(m.ctxTotal)
	put(m.ctxTypes)
	if err == nil {
		err = w.Flush()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	return err
}

// Load reads a model written by Save.
func Load(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	head := make([]byte, len(magic))
	if _, err := io.ReadFull(r, head); err != nil || string(head) != magic {
		return nil, fmt.Errorf("%s: not a character language model", path)
	}
	var order, vocab uint32
	var n, nc uint64
	get := func(v any) {
		if err == nil {
			err = binary.Read(r, binary.LittleEndian, v)
		}
	}
	get(&order)
	get(&vocab)
	get(&n)
	if err != nil || order < 1 || order > 16 || n > 1<<32 {
		return nil, fmt.Errorf("%s: damaged header", path)
	}
	m := &Model{Order: int(order), vocab: int(vocab), ngramKeys: make([]uint64, n), ngramVals: make([]uint32, n)}
	get(m.ngramKeys)
	get(m.ngramVals)
	get(&nc)
	if err != nil || nc > 1<<32 {
		return nil, fmt.Errorf("%s: damaged", path)
	}
	m.ctxKeys, m.ctxTotal, m.ctxTypes = make([]uint64, nc), make([]uint32, nc), make([]uint32, nc)
	get(m.ctxKeys)
	get(m.ctxTotal)
	get(m.ctxTypes)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	m.buildIndexes()
	return m, nil
}

// Size returns the number of n-grams and contexts kept.
func (m *Model) Size() (ngrams, contexts int) { return len(m.ngramKeys), len(m.ctxKeys) }
