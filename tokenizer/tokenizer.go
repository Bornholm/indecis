// Package tokenizer reproduit en Go pur le tokenizer BPE des modèles de la
// famille Gemma, tel que le décrit un fichier tokenizer.json de Hugging Face.
//
// Le contrat est la parité exacte avec la bibliothèque Rust `tokenizers` :
// mêmes ids pour le même texte. Un id différent, c'est un modèle qui lit un
// autre texte que celui sur lequel il a été entraîné. La chaîne est donc
// reproduite étape par étape, y compris ses particularités :
//
//  1. les tokens ajoutés (spéciaux ou non) sont repérés dans le texte brut,
//     en correspondance leftmost-longest, avant toute normalisation ;
//  2. chaque segment restant est normalisé (espace → ▁), reçoit un ▁ en tête
//     s'il n'en a pas, puis est découpé devant chaque ▁ ;
//  3. chaque morceau passe par le BPE : fusions par rang croissant, à rang
//     égal la plus à gauche, et repli sur les octets pour les caractères
//     inconnus du vocabulaire ;
//  4. le gabarit ajoute <bos> et <eos>.
//
// Seule cette configuration est acceptée : un tokenizer.json qui en diffère
// est refusé au chargement plutôt que tokenisé approximativement.
package tokenizer

import (
	"bufio"
	"bytes"
	"container/heap"
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

// Tokenizer est sûr pour un usage concurrent.
type Tokenizer struct {
	vocab   *strTable        // chaînes du vocabulaire BPE, par id
	names   map[int32]string // tokens ajoutés, s'ils diffèrent du vocabulaire
	size    int              // nombre d'ids
	merges  mergeTable
	bytes   [256]int32 // id de <0xXX>, -1 si absent
	unk     int32
	bos     int32
	eos     int32
	pad     int32
	added   map[byte][]addedToken // par premier octet, du plus long au plus court
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

// Load lit un tokenizer.json.
func Load(path string) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return read(bufio.NewReaderSize(f, 1<<16))
}

// Parse lit le contenu d'un tokenizer.json.
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
	} `json:"model"` // vocab et merges sont lus en flux (voir decode)
}

type templatePiece struct {
	SpecialToken *struct {
		ID string `json:"id"`
	} `json:"SpecialToken"`
	Sequence *struct {
		ID string `json:"id"`
	} `json:"Sequence"`
}

// matches vérifie un gabarit : chaque élément est un token spécial (nommé)
// ou une séquence (vide).
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

	t := &Tokenizer{
		names: make(map[int32]string),
		added: make(map[byte][]addedToken),
		cache: make(map[string][]int32),
	}
	for _, a := range f.AddedTokens {
		if a.ID < 0 {
			return nil, fmt.Errorf("tokenizer: id %d négatif", a.ID)
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
			return nil, fmt.Errorf("tokenizer: added token %q: normalized/single_word non pris en charge", a.Content)
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
		return nil, fmt.Errorf("tokenizer: unk token %q absent", f.Model.UnkToken)
	}
	if t.bos, ok = t.lookup("<bos>"); !ok {
		return nil, fmt.Errorf("tokenizer: <bos> absent")
	}
	if t.eos, ok = t.lookup("<eos>"); !ok {
		return nil, fmt.Errorf("tokenizer: <eos> absent")
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
			return nil, fmt.Errorf("tokenizer: merge %q %q hors vocabulaire", p[0], p[1])
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
		return fmt.Errorf("tokenizer: modèle %q non pris en charge", m.Type)
	case m.Dropout != nil && *m.Dropout != 0:
		return fmt.Errorf("tokenizer: dropout BPE non pris en charge")
	case m.ContinuingSubwordPrefix != nil && *m.ContinuingSubwordPrefix != "",
		m.EndOfWordSuffix != nil && *m.EndOfWordSuffix != "":
		return fmt.Errorf("tokenizer: préfixe/suffixe de sous-mot non pris en charge")
	case !m.ByteFallback || !m.FuseUnk || m.IgnoreMerges:
		return fmt.Errorf("tokenizer: attendu byte_fallback=true, fuse_unk=true, ignore_merges=false")
	}
	if n := f.Normalizer; n != nil && (n.Type != "Replace" || n.Pattern.String != " " || n.Content != metaspace) {
		return fmt.Errorf("tokenizer: normalizer non pris en charge")
	}
	p := f.PreTokenizer
	if p == nil || p.Type != "Metaspace" || p.Replacement != metaspace || p.PrependScheme != "always" || !p.Split {
		return fmt.Errorf("tokenizer: pré-tokenizer attendu : Metaspace(▁, always, split)")
	}
	pp := f.PostProcessor
	if pp == nil || pp.Type != "TemplateProcessing" || !matches(pp.Single, "<bos>", "", "<eos>") ||
		!matches(pp.Pair, "<bos>", "", "<eos>", "", "<eos>") {
		return fmt.Errorf("tokenizer: gabarits attendus : <bos> A <eos> et <bos> A <eos> B <eos>")
	}
	return nil
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

// VocabSize est le nombre d'ids possibles.
func (t *Tokenizer) VocabSize() int { return t.size }

// Token retourne la chaîne d'un id.
func (t *Tokenizer) Token(id int32) string {
	if id < 0 || int(id) >= t.size {
		return ""
	}
	if s, ok := t.names[id]; ok {
		return s
	}
	return t.vocab.str(id)
}

// PadID, BosID, EosID exposent les ids spéciaux.
func (t *Tokenizer) PadID() int32 { return t.pad }
func (t *Tokenizer) BosID() int32 { return t.bos }
func (t *Tokenizer) EosID() int32 { return t.eos }

// Encode tokenise text et encadre le résultat de <bos> et <eos>.
func (t *Tokenizer) Encode(text string) []int32 {
	ids := []int32{t.bos}
	ids = t.appendText(ids, text)
	return append(ids, t.eos)
}

// EncodeMax tokenise comme Encode et tronque à droite pour ne pas dépasser
// maxLen ids, <eos> final conservé. maxLen ≤ 2 donne [<bos>, <eos>].
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

// EncodePair tokenise une paire (contexte, texte) selon le gabarit
// <bos> contexte <eos> texte <eos>, celui de la bibliothèque de référence.
//
// Au-delà de maxLen ids (0 : pas de limite), la troncature sacrifie le
// contexte avant le texte : le texte est ce qu'on juge, le contexte ce qui
// l'éclaire. Le contexte garde au moins un tiers du budget s'il est assez
// long, et c'est son début qui est conservé : un prompt système pose le rôle
// de l'assistant en tête.
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
	ids = append(ids, t.bos)
	ids = append(ids, a...)
	ids = append(ids, t.eos)
	ids = append(ids, b...)
	return append(ids, t.eos)
}

// appendText découpe text autour des tokens ajoutés et tokenise les segments.
func (t *Tokenizer) appendText(ids []int32, text string) []int32 {
	start := 0 // début du segment courant
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
		ids = t.appendSegment(ids, text[start:mStart])
		ids = append(ids, a.id)
		start, i = mEnd, mEnd
	}
	return t.appendSegment(ids, text[start:])
}

// matchAdded retourne le plus long token ajouté qui commence en text[i].
func (t *Tokenizer) matchAdded(text string, i int) (addedToken, bool) {
	for _, a := range t.added[text[i]] {
		if strings.HasPrefix(text[i:], a.content) {
			return a, true
		}
	}
	return addedToken{}, false
}

// appendSegment normalise, pré-tokenise et passe chaque morceau au BPE.
func (t *Tokenizer) appendSegment(ids []int32, seg string) []int32 {
	if seg == "" {
		return ids
	}
	s := strings.ReplaceAll(seg, " ", metaspace)
	if !strings.HasPrefix(s, metaspace) {
		s = metaspace + s
	}
	// Découpage devant chaque ▁, le ▁ restant attaché au morceau suivant.
	for len(s) > 0 {
		next := strings.Index(s[len(metaspace):], metaspace)
		if next < 0 {
			ids = t.appendWord(ids, s)
			break
		}
		cut := next + len(metaspace)
		if !strings.HasPrefix(s, metaspace) {
			// Seul le premier morceau peut ne pas commencer par ▁ ; ce n'est
			// pas le cas ici puisqu'un ▁ a été ajouté en tête.
			cut = strings.Index(s, metaspace)
		}
		ids = t.appendWord(ids, s[:cut])
		s = s[cut:]
	}
	return ids
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
	merged     bool // absorbé par son voisin de gauche
}

// bpe applique les fusions à un mot, dans l'ordre de la bibliothèque de
// référence : rang croissant, puis position croissante.
func (t *Tokenizer) bpe(word string) []int32 {
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
		// Entrée périmée : la paire a changé depuis son ajout.
		if m, ok := t.merges.get(pairKey(cur.id, right.id)); !ok || m.id != top.id {
			continue
		}
		cur.id = top.id
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
