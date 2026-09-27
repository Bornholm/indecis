package indecis

import "context"

// batcher regroupe les calculs de requêtes simultanées (voir WithBatching).
type batcher struct {
	m       *Model
	maxRows int
	reqs    chan *batchReq
}

type batchReq struct {
	seqs [][]int32
	rows int
	done chan batchResult // tampon de 1 : le worker n'attend jamais
}

type batchResult struct {
	pooled []float32
	err    error
}

func newBatcher(m *Model, workers, maxRows int) *batcher {
	b := &batcher{m: m, maxRows: maxRows, reqs: make(chan *batchReq, 4*workers)}
	for range workers {
		go b.work()
	}
	return b
}

func (b *batcher) encode(ctx context.Context, seqs [][]int32) ([]float32, error) {
	r := &batchReq{seqs: seqs, done: make(chan batchResult, 1)}
	for _, s := range seqs {
		r.rows += len(s)
	}
	select {
	case b.reqs <- r:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	select {
	case res := <-r.done:
		return res.pooled, res.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// work prend la première requête en attente puis, sans attendre, celles
// déjà en file tant que le lot reste sous maxRows positions.
func (b *batcher) work() {
	H := b.m.enc.Cfg.Hidden
	for first := range b.reqs {
		batch := []*batchReq{first}
		rows := first.rows
	fill:
		for rows < b.maxRows {
			select {
			case r := <-b.reqs:
				batch = append(batch, r)
				rows += r.rows
			default:
				break fill
			}
		}
		var seqs [][]int32
		for _, r := range batch {
			seqs = append(seqs, r.seqs...)
		}
		pooled, err := b.m.enc.EncodeSeqs(seqs)
		off := 0
		for _, r := range batch {
			if err != nil {
				r.done <- batchResult{err: err}
				continue
			}
			r.done <- batchResult{pooled: pooled[off*H : (off+len(r.seqs))*H]}
			off += len(r.seqs)
		}
	}
}
