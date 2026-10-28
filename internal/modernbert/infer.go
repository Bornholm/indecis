package modernbert

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"

	"github.com/bornholm/indecis/linalg"
)

// EmbeddingTable supplies rows of the embedding table on demand. The table
// makes up 93% of the weights; in inference, a request reads only a few
// dozen rows, which is pointless to keep in float32.
type EmbeddingTable interface {
	// Row writes row id into dst (Hidden values).
	Row(id int32, dst []float32)
}

// SetEmbeddingTable replaces the float32 embedding table with an
// on-demand table. Emb.W is then nil until Materialize.
func (m *Model) SetEmbeddingTable(t EmbeddingTable) {
	m.embTable = t
	m.Emb.W = nil
}

// Materialize copies the embedding table into Emb.W in float32, as
// required by training.
func (m *Model) Materialize() {
	m.restoreWeights()
	if m.Emb.W != nil {
		return
	}
	m.Emb.W = m.EmbeddingMatrix()
	m.embTable = nil
}

// EmbeddingMatrix returns the embedding table in float32: Emb.W, or a
// temporary copy if the table is read on demand.
func (m *Model) EmbeddingMatrix() []float32 {
	if m.Emb.W != nil {
		return m.Emb.W
	}
	H := m.Cfg.Hidden
	w := make([]float32, m.Cfg.Vocab*H)
	for id := 0; id < m.Cfg.Vocab; id++ {
		m.embTable.Row(int32(id), w[id*H:(id+1)*H])
	}
	return w
}

func (m *Model) embRow(id int32, dst []float32) {
	if m.Emb.W != nil {
		H := m.Cfg.Hidden
		copy(dst, m.Emb.W[int(id)*H:(int(id)+1)*H])
		return
	}
	m.embTable.Row(id, dst)
}

// packedMat is a weight matrix ready for multiplication: packed in
// float32, or quantized to int8 (see SetInt8).
type packedMat struct {
	f *linalg.PackedB
	q *linalg.PackedB8
}

func packMat(w []float32, k, n int, int8 bool) packedMat {
	if int8 {
		return packedMat{q: linalg.PackB8(w, k, n, true)}
	}
	return packedMat{f: linalg.PackB(w, k, n, true)}
}

// mul computes c = a·Wᵀ (or c += if accumulate), a having m rows.
func (p packedMat) mul(c, a []float32, m int, accumulate bool, workers int) {
	if p.q != nil {
		linalg.MatMul8N(c, a, p.q, m, accumulate, workers)
		return
	}
	linalg.MatMulPackedN(c, a, p.f, m, accumulate, workers)
}

// packedLayer holds the matrices of a layer ready for multiplication.
type packedLayer struct {
	qkv, o, i, oMLP packedMat
}

// WeightSource re-reads a weight matrix by name, from the model file for
// example.
type WeightSource func(name string) ([]float32, error)

// Invalidate signals that the weights have changed: the matrices packed
// for inference must be redone. Training calls this after every step.
func (m *Model) Invalidate() {
	m.packMu.Lock()
	defer m.packMu.Unlock()
	m.restoreLocked()
	m.packed = nil
}

// SetCompact makes the layer matrices kept only in their packed form,
// starting at the first inference: they are no longer duplicated. The
// functions that read the weights (Params, Forward, Materialize) rebuild
// them as needed, from source if it is provided, and compact mode then
// ends. In int8, the matrices are freed only if source is provided:
// quantization is not undone.
func (m *Model) SetCompact(source WeightSource) {
	m.packMu.Lock()
	m.compact = true
	m.source = source
	m.packMu.Unlock()
}

// SetInt8 makes the layers compute their products in int8 (per-channel
// weights, per-token activations). Without a hardware kernel
// (linalg.Int8Fast), this is exact but slow. With it, the attention runs
// in int8 too (linalg.Attend8).
func (m *Model) SetInt8(on bool) {
	m.packMu.Lock()
	defer m.packMu.Unlock()
	if m.int8 == on {
		return
	}
	// The matrices will be repacked from Param.W or, if absent, from the
	// source. Without a source, freed matrices only survive in the
	// current packs: they must be restored first.
	if m.packed != nil && m.source == nil {
		compact := m.compact
		m.restoreLocked()
		m.compact = compact // changing format is not reading the weights
	}
	m.int8 = on
	m.packed = nil
}

// restoreWeights rebuilds the layer matrices if SetCompact freed them.
func (m *Model) restoreWeights() {
	m.packMu.Lock()
	defer m.packMu.Unlock()
	m.restoreLocked()
}

func (m *Model) restoreLocked() {
	m.compact = false
	for l := range m.Layers {
		L := &m.Layers[l]
		var p packedLayer
		if m.packed != nil {
			p = m.packed[l]
		}
		for _, x := range []struct {
			w *Param
			p packedMat
		}{{L.Wqkv, p.qkv}, {L.Wo, p.o}, {L.Wi, p.i}, {L.WoMLP, p.oMLP}} {
			if x.w.W != nil {
				continue
			}
			if m.source != nil {
				x.w.W = m.readSource(x.w)
				continue
			}
			x.w.W = make([]float32, x.w.Shape[0]*x.w.Shape[1])
			x.p.f.Unpack(x.w.W, true)
		}
	}
}

// readSource re-reads a matrix from the SetCompact source.
func (m *Model) readSource(w *Param) []float32 {
	if m.source == nil {
		panic(fmt.Sprintf("modernbert: %s missing and no source to reread it", w.Name))
	}
	data, err := m.source(w.Name)
	if err == nil && len(data) != w.Shape[0]*w.Shape[1] {
		err = fmt.Errorf("%d values", len(data))
	}
	if err != nil {
		panic(fmt.Sprintf("modernbert: rereading %s: %v", w.Name, err))
	}
	return data
}

func (m *Model) packs() []packedLayer {
	m.packMu.Lock()
	defer m.packMu.Unlock()
	if m.packed != nil {
		return m.packed
	}
	H, I := m.Cfg.Hidden, m.Cfg.Intermediate
	p := make([]packedLayer, len(m.Layers))
	drop := m.compact && (m.source != nil || !m.int8)
	// A missing matrix (lazy loading, see SetCompact) is read from the
	// source at the moment it is packed: there is never more than one in
	// float32 at a time.
	pack := func(w *Param, k, n int) packedMat {
		data := w.W
		if data == nil {
			data = m.readSource(w)
		}
		pm := packMat(data, k, n, m.int8)
		if drop {
			w.W = nil
		} else {
			w.W = data
		}
		return pm
	}
	for l, L := range m.Layers {
		p[l] = packedLayer{
			qkv:  pack(L.Wqkv, H, 3*H),
			o:    pack(L.Wo, H, H),
			i:    pack(L.Wi, H, 2*I),
			oMLP: pack(L.WoMLP, I, H),
		}
	}
	m.packed = p
	return p
}

// workspace groups the buffers of an inference pass, reused across
// requests.
type workspace struct {
	x, xn, qkv, ctx []float32
	pos             []int32        // position of each row in its sequence
	attn            []linalg.Attn8 // one per (sequence, head), see attention8
}

// mlpBuffers holds the MLP buffers of the workers (see run).
var mlpBuffers = sync.Pool{New: func() any { return new([]float32) }}

// attnScratches holds the buffers of the workers of attention8.
var attnScratches = sync.Pool{New: func() any { return new(linalg.AttnScratch) }}

var workspaces sync.Pool

// Encode computes the mean embedding of each sequence in the batch,
// [B, H].
//
// This is the inference path: the same computation as Forward followed by
// MeanPool, without keeping anything for backpropagation, with weights
// packed once and for all and erf/exp in float32. The discrepancy with
// Forward stays around 1e-6.
func (m *Model) Encode(b Batch) ([]float32, error) {
	if len(b.IDs) != b.B()*b.T {
		return nil, fmt.Errorf("modernbert: inconsistent batch")
	}
	seqs := make([][]int32, b.B())
	for i, n := range b.Lens {
		seqs[i] = b.IDs[i*b.T : i*b.T+n]
	}
	return m.EncodeSeqs(seqs)
}

// EncodeSeqs is Encode on sequences of arbitrary lengths, concatenated
// without padding: the per-row computations (projections, MLP,
// normalizations) run on the sum of the lengths, not on the number of
// sequences times the longest one, and attention is computed sequence by
// sequence. Combining sequences of different lengths therefore costs no
// more than computing them separately.
func (m *Model) EncodeSeqs(seqs [][]int32) ([]float32, error) {
	H := m.Cfg.Hidden
	out := make([]float32, len(seqs)*H)
	err := m.run(seqs, func(xn []float32, offs []int) {
		// Average of each sequence's states, special tokens included.
		for i := range seqs {
			n := offs[i+1] - offs[i]
			if n == 0 {
				continue
			}
			acc := make([]float64, H)
			for t := offs[i]; t < offs[i+1]; t++ {
				for k, v := range xn[t*H : (t+1)*H] {
					acc[k] += float64(v)
				}
			}
			for k := range acc {
				out[i*H+k] = float32(acc[k] / float64(n))
			}
		}
	})
	return out, err
}

// EncodeTokens returns the final state of every token of the sequences,
// laid end to end: [sum of the lengths, H].
func (m *Model) EncodeTokens(seqs [][]int32) ([]float32, error) {
	var out []float32
	err := m.run(seqs, func(xn []float32, offs []int) {
		out = append([]float32(nil), xn[:offs[len(seqs)]*m.Cfg.Hidden]...)
	})
	return out, err
}

// run encodes the sequences laid end to end and passes done the final
// states, [N, H], and the start of each sequence (offs[len(seqs)] = N).
// The states live in a pooled buffer: done copies what it keeps.
func (m *Model) run(seqs [][]int32, done func(xn []float32, offs []int)) error {
	cfg := m.Cfg
	H, I := cfg.Hidden, cfg.Intermediate
	offs := make([]int, len(seqs)+1)
	for i, sq := range seqs {
		for _, id := range sq {
			if id < 0 || int(id) >= cfg.Vocab {
				return fmt.Errorf("modernbert: id %d out of vocabulary", id)
			}
		}
		offs[i+1] = offs[i] + len(sq)
	}
	N := offs[len(seqs)]
	if N == 0 {
		done(nil, offs)
		return nil
	}
	packs := m.packs()
	// A computation takes the cores the others leave free, counting one
	// core for each other computation in progress, spread or not. The
	// count is optimistic: two spread computations together take more
	// goroutines than there are cores. Dividing the cores exactly among
	// them was measured with the same throughput and a slower 256-token
	// latency (docs/inference.md). Under load, each one spreads less, and
	// on a saturated server, not at all. The share is read again at every
	// layer, as computations come and go.
	spread := N >= parallelRows
	running.Add(1)
	defer running.Add(-1)
	share := func() int {
		if !spread {
			return 1
		}
		return max(1, linalg.Workers()-int(running.Load())+1)
	}
	w := share()
	slice := sliceRows(N)

	ws, _ := workspaces.Get().(*workspace)
	if ws == nil {
		ws = &workspace{}
	}
	defer workspaces.Put(ws)
	ws.x = grow(ws.x, N*H)
	ws.xn = grow(ws.xn, N*H)
	ws.qkv = grow(ws.qkv, N*3*H)
	ws.ctx = grow(ws.ctx, N*H)
	x, xn := ws.x, ws.xn
	ws.pos = grow(ws.pos, N)
	T := 0
	for i := range seqs {
		T = max(T, offs[i+1]-offs[i])
		for t := offs[i]; t < offs[i+1]; t++ {
			ws.pos[t] = int32(t - offs[i])
		}
	}

	r := 0
	for _, sq := range seqs {
		for _, id := range sq {
			m.embRow(id, xn[r*H:(r+1)*H])
			r++
		}
	}
	layerNormInfer(x, xn, m.EmbNorm.W, N, H, cfg.NormEps, w)

	nh, D := cfg.Heads, cfg.HeadDim()
	half := D / 2
	for l, L := range m.Layers {
		p := packs[l]
		w = share()
		rope := m.ropeFor(m.theta(l), T)
		// Outside the attention, rows are independent: each worker takes
		// slices of rows through the norm and the QKV projection, then
		// through the output projection and the MLP, its products on one
		// core. A layer waits three times for all workers, not at every
		// product: on a hybrid processor, each wait is for the slowest
		// core.
		linalg.ParallelN(w, N, slice, func(lo, hi int) {
			in := x[lo*H : hi*H]
			if L.AttnNorm != nil {
				for r := lo; r < hi; r++ {
					linalg.LayerNormRow(xn[r*H:(r+1)*H], x[r*H:(r+1)*H], L.AttnNorm.W, cfg.NormEps)
				}
				in = xn[lo*H : hi*H]
			}
			p.qkv.mul(ws.qkv[lo*3*H:hi*3*H], in, hi-lo, false, 1)
			// RoPE on the queries and keys of every head, in place.
			for r := lo; r < hi; r++ {
				t := int(ws.pos[r])
				cos, sin := rope.cos[t*half:(t+1)*half], rope.sin[t*half:(t+1)*half]
				row := ws.qkv[r*3*H : r*3*H+2*H]
				for h := 0; h < 2*nh; h++ {
					linalg.Rope(row[h*D:(h+1)*D], cos, sin)
				}
			}
		})
		m.attentionInfer(l, offs, ws, w)
		linalg.ParallelN(w, N, slice, func(lo, hi int) {
			p.o.mul(x[lo*H:hi*H], ws.ctx[lo*H:hi*H], hi-lo, true, 1) // x += attention
			for r := lo; r < hi; r++ {
				linalg.LayerNormRow(xn[r*H:(r+1)*H], x[r*H:(r+1)*H], L.MLPNorm.W, cfg.NormEps)
			}
			// The MLP works in chunks of rows: its buffers (2·I + I values
			// per row, more than everything else) do not depend on the
			// text length.
			buf := mlpBuffers.Get().(*[]float32)
			defer mlpBuffers.Put(buf)
			*buf = grow(*buf, mlpChunk*3*I)
			z, g := (*buf)[:mlpChunk*2*I], (*buf)[mlpChunk*2*I:]
			for r0 := lo; r0 < hi; r0 += mlpChunk {
				rows := min(mlpChunk, hi-r0)
				p.i.mul(z, xn[r0*H:(r0+rows)*H], rows, false, 1)
				for r := 0; r < rows; r++ {
					linalg.GeluMul(g[r*I:(r+1)*I], z[r*2*I:r*2*I+I], z[r*2*I+I:(r+1)*2*I])
				}
				if st := m.stats; st != nil {
					st.add(l, g, rows, I)
				}
				p.oMLP.mul(x[r0*H:(r0+rows)*H], g, rows, true, 1) // x += MLP
			}
		})
	}
	layerNormInfer(xn, x, m.FinalNorm.W, N, H, cfg.NormEps, share())
	done(xn, offs)
	return nil
}

// attentionInfer is attentionForward without keeping anything, on the
// rows of ws.qkv, RoPE already applied (see run): q, k, v of each
// (sequence, head) live in the task's buffers, and attention is computed
// block by block (attendBlocks), in linear memory. In int8, see
// attention8.
func (m *Model) attentionInfer(l int, offs []int, ws *workspace, workers int) {
	cfg := m.Cfg
	H, nh, D := cfg.Hidden, cfg.Heads, cfg.HeadDim()
	B := len(offs) - 1
	T := 0
	for i := 0; i < B; i++ {
		T = max(T, offs[i+1]-offs[i])
	}
	scale := float32(1 / math.Sqrt(float64(D)))
	window := -1
	if !cfg.IsGlobal(l) {
		window = cfg.Window()
	}
	if m.int8 && linalg.Int8Fast() {
		m.attention8(offs, ws, window, scale, workers)
		return
	}
	linalg.ParallelN(workers, B*nh, 1, func(lo, hi int) {
		buf := make([]float32, 4*T*D)
		q, k, v, o := buf[:T*D], buf[T*D:2*T*D], buf[2*T*D:3*T*D], buf[3*T*D:4*T*D]
		blk := newBlockBuffers()
		for task := lo; task < hi; task++ {
			bi, h := task/nh, task%nh
			off, n := offs[bi], offs[bi+1]-offs[bi]
			if n == 0 {
				continue
			}
			for t := 0; t < n; t++ {
				row := ws.qkv[(off+t)*3*H:]
				copy(q[t*D:(t+1)*D], row[h*D:(h+1)*D])
				copy(k[t*D:(t+1)*D], row[H+h*D:H+(h+1)*D])
				copy(v[t*D:(t+1)*D], row[2*H+h*D:2*H+(h+1)*D])
			}
			attendBlocks(o, q, k, v, n, D, window, scale, blk)
			for t := 0; t < n; t++ {
				copy(ws.ctx[(off+t)*H+h*D:(off+t)*H+(h+1)*D], o[t*D:(t+1)*D])
			}
		}
	})
}

// attention8 computes the attention in int8 (linalg.Attn8), in two
// passes: each (sequence, head) is quantized and packed, straight from
// the rows of ws.qkv, then the blocks of queries are scored. Spread over
// blocks rather than (sequence, head), the work divides finely even for
// one sequence and a few heads: on a hybrid processor, no slow core is
// left with a whole head while the others wait.
func (m *Model) attention8(offs []int, ws *workspace, window int, scale float32, workers int) {
	H, nh, D := m.Cfg.Hidden, m.Cfg.Heads, m.Cfg.HeadDim()
	B := len(offs) - 1
	if cap(ws.attn) < B*nh {
		ws.attn = make([]linalg.Attn8, B*nh)
	}
	prep := ws.attn[:B*nh]
	linalg.ParallelN(workers, B*nh, 1, func(lo, hi int) {
		for task := lo; task < hi; task++ {
			bi, h := task/nh, task%nh
			off, n := offs[bi], offs[bi+1]-offs[bi]
			rows := ws.qkv[off*3*H:]
			prep[task].Prepare(rows[h*D:], rows[H+h*D:], rows[2*H+h*D:], 3*H, n, D, scale)
		}
	})
	// The blocks of queries: (sequence, head, first query).
	type block struct{ task, i0 int }
	var blocks []block
	for bi := 0; bi < B; bi++ {
		for h := 0; h < nh; h++ {
			for i0 := 0; i0 < offs[bi+1]-offs[bi]; i0 += linalg.AttnBlock {
				blocks = append(blocks, block{bi*nh + h, i0})
			}
		}
	}
	linalg.ParallelN(workers, len(blocks), 1, func(lo, hi int) {
		sc := attnScratches.Get().(*linalg.AttnScratch)
		defer attnScratches.Put(sc)
		for _, b := range blocks[lo:hi] {
			bi, h := b.task/nh, b.task%nh
			off, n := offs[bi], offs[bi+1]-offs[bi]
			prep[b.task].Attend(ws.ctx[(off+b.i0)*H+h*D:], H, b.i0, min(n, b.i0+linalg.AttnBlock), window, sc)
		}
	})
}

// layerNormInfer is layerNorm without a cache for the backward pass.
func layerNormInfer(out, x, gamma []float32, n, h int, eps float64, workers int) {
	linalg.ParallelN(workers, n, rowGrain, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			linalg.LayerNormRow(out[r*h:(r+1)*h], x[r*h:(r+1)*h], gamma, eps)
		}
	})
}

// Block attention, FlashAttention-style: queries are processed in blocks
// of qBlock positions, keys in blocks of kBlock, and the softmax is
// updated as key blocks are processed (running max and sum per row). The
// full score matrix never exists: memory is linear in the length. In a
// local layer, only the key blocks that touch the window are visited: the
// computation becomes linear too. The result is that of the full softmax,
// up to rounding.
const (
	qBlock = 64
	kBlock = 128

	// mlpChunk is the number of rows the MLP processes at a time, on one
	// core.
	mlpChunk = 64

	// rowSlice is the largest grain of the slices of rows outside the
	// attention (see sliceRows).
	rowSlice = 24

	// parallelRows is the number of positions (across all sequences in
	// the batch) above which Encode spreads its computation over multiple
	// cores. Below it, waking the workers costs more than the computation.
	// Above it, the slices of rows (sliceRows) divide the work well: on a
	// Core Ultra 7 265U, all cores compute 34 positions 10% faster than
	// one, 89 twice as fast, 256 three times as fast.
	parallelRows = 32
)

// running counts the computations in progress, of every text model; a
// batch counts as one. Image models and training are not counted: a
// server that runs them alongside takes more goroutines than it has
// cores.
var running atomic.Int64

// sliceRows is the smallest slice of rows a worker takes outside the
// attention, for N rows: a multiple of the int8 kernel's 6 rows, about an
// eighth of N, at most rowSlice. Long slices make the products efficient
// (each slice reads all the weights); short texts need enough of them for
// the cores to share them. Slices of 6 rows handed out long first and
// short last were measured slower: 2.9 ms instead of 2.5 for 89 tokens,
// and no faster for 256 (4.7 ms instead of 4.9, in the noise).
func sliceRows(N int) int {
	return min(rowSlice, max(6, (N/8+3)/6*6))
}

type blockBuffers struct {
	s       []float32 // scores of a block, [qBlock, kBlock]
	mx, sum []float32 // running max and sum per row, [n]
}

func newBlockBuffers() *blockBuffers {
	return &blockBuffers{s: make([]float32, qBlock*kBlock)}
}

// attendBlocks writes into o (n×D) the attention of q over k, v (n×D
// each). window < 0: global attention; otherwise |i − j| ≤ window. q is
// modified (multiplied by scale).
//
// The key blocks are the outer loop: each block of K and V is packed once
// for the matrix product, then serves every query block that sees it. The
// softmax max and sum are kept for all rows.
func attendBlocks(o, q, k, v []float32, n, D, window int, scale float32, b *blockBuffers) {
	negInf := float32(math.Inf(-1))
	linalg.Scale(q[:n*D], scale) // scores already scaled
	clear(o[:n*D])
	if cap(b.mx) < n {
		b.mx, b.sum = make([]float32, n), make([]float32, n)
	}
	mx, sum := b.mx[:n], b.sum[:n]
	for i := range mx {
		mx[i], sum[i] = negInf, 0
	}
	for jb := 0; jb < n; jb += kBlock {
		je := min(n, jb+kBlock)
		cols := je - jb
		pk := linalg.PackB(k[jb*D:je*D], D, cols, true)  // block Kᵀ: D×cols
		pv := linalg.PackB(v[jb*D:je*D], cols, D, false) // block V: cols×D
		// Queries that see at least one key of the block.
		q0, q1 := 0, n
		if window >= 0 {
			q0, q1 = max(0, jb-window), min(n, je+window)
		}
		q0 = q0 / qBlock * qBlock
		for i0 := q0; i0 < q1; i0 += qBlock {
			i1 := min(n, i0+qBlock)
			rows := i1 - i0
			sc := b.s[:rows*cols]
			linalg.MatMulPackedN(sc, q[i0*D:i1*D], pk, rows, false, 1)
			for r := 0; r < rows; r++ {
				i := i0 + r
				row := sc[r*cols : (r+1)*cols]
				// Window columns in this block: [cs, ce).
				cs, ce := 0, cols
				if window >= 0 {
					cs, ce = max(0, i-window-jb), min(cols, i+window+1-jb)
				}
				if cs >= ce {
					clear(row) // no key of this block within the window
					continue
				}
				valid := row[cs:ce]
				m := max(mx[i], linalg.MaxOf(valid))
				if corr := exp32(mx[i] - m); corr != 1 {
					sum[i] *= corr
					linalg.Scale(o[i*D:(i+1)*D], corr)
				}
				mx[i] = m
				sum[i] += linalg.ExpShift(valid, m)
				clear(row[:cs])
				clear(row[ce:])
			}
			// o += P·V of the block.
			linalg.MatMulPackedN(o[i0*D:i1*D], sc, pv, rows, true, 1)
		}
	}
	for i := 0; i < n; i++ {
		linalg.Scale(o[i*D:(i+1)*D], 1/sum[i])
	}
}

// RemapVocabulary shrinks the embedding table to a new vocabulary:
// newID[old] is the new id of a kept token, -1 for a removed token. The
// kept ids must form [0, size).
func (m *Model) RemapVocabulary(newID []int32, size int) error {
	if len(newID) != m.Cfg.Vocab {
		return fmt.Errorf("modernbert: mapping of %d ids for a vocabulary of %d", len(newID), m.Cfg.Vocab)
	}
	if pad := newID[m.Cfg.PadID]; pad < 0 {
		return fmt.Errorf("modernbert: padding token is removed")
	}
	H := m.Cfg.Hidden
	w := make([]float32, size*H)
	for old, id := range newID {
		if id >= 0 {
			if int(id) >= size {
				return fmt.Errorf("modernbert: id %d out of the new vocabulary", id)
			}
			m.embRow(int32(old), w[int(id)*H:(int(id)+1)*H])
		}
	}
	m.Cfg.PadID = newID[m.Cfg.PadID]
	m.Cfg.Vocab = size
	m.Emb.W, m.Emb.Shape, m.embTable = w, []int{size, H}, nil
	return nil
}
