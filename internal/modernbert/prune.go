package modernbert

import (
	"fmt"
	"sync"
)

// Pruning of the MLP neurons. The MLP holds two thirds of the products of
// a layer (2·I + I values per token against 3·H + H for the attention);
// removing its least active neurons makes the model faster, and a
// distillation from the unpruned model recovers what was lost.

// mlpStats accumulates |gelu(input)·gate| per layer and neuron.
type mlpStats struct {
	mu     sync.Mutex
	sum    [][]float64
	tokens int
}

func (st *mlpStats) add(l int, g []float32, rows, I int) {
	st.mu.Lock()
	defer st.mu.Unlock()
	s := st.sum[l]
	for r := 0; r < rows; r++ {
		for i, v := range g[r*I : (r+1)*I] {
			s[i] += float64(abs32(v))
		}
	}
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// MLPActivity returns, for each layer, the mean of |gelu(input)·gate| of
// each MLP neuron over the tokens of the sequences, through the inference
// path. It must not run at the same time as another inference on the
// model.
func (m *Model) MLPActivity(seqs [][]int32) ([][]float64, error) {
	st := &mlpStats{sum: make([][]float64, len(m.Layers))}
	for l := range st.sum {
		st.sum[l] = make([]float64, m.Cfg.Intermediate)
	}
	for _, s := range seqs {
		st.tokens += len(s)
	}
	m.stats = st
	defer func() { m.stats = nil }()
	if err := m.run(seqs, func([]float32, []int) {}); err != nil {
		return nil, err
	}
	for _, s := range st.sum {
		for i := range s {
			s[i] /= float64(max(st.tokens, 1))
		}
	}
	return st.sum, nil
}

// MLPOutputNorms returns, for each layer, the norm of the output weights
// of each MLP neuron (the columns of WoMLP): what one unit of its
// activity adds to the residual stream.
func (m *Model) MLPOutputNorms() [][]float64 {
	m.restoreWeights()
	H, I := m.Cfg.Hidden, m.Cfg.Intermediate
	out := make([][]float64, len(m.Layers))
	for l, L := range m.Layers {
		out[l] = make([]float64, I)
		for h := 0; h < H; h++ {
			for i, w := range L.WoMLP.W[h*I : (h+1)*I] {
				out[l][i] += float64(w) * float64(w)
			}
		}
		for i := range out[l] {
			out[l][i] = sqrt(out[l][i])
		}
	}
	return out
}

// PruneMLP keeps, in each layer, the MLP neurons keep[l] (as many in
// every layer: config.json has one intermediate_size). The weights are
// then held in float32 only, no longer reread from their source.
func (m *Model) PruneMLP(keep [][]int) error {
	if len(keep) != len(m.Layers) {
		return fmt.Errorf("modernbert: %d layers to prune, %d layers", len(keep), len(m.Layers))
	}
	H, I := m.Cfg.Hidden, m.Cfg.Intermediate
	n := len(keep[0])
	for l, k := range keep {
		if len(k) != n || n == 0 {
			return fmt.Errorf("modernbert: layer %d keeps %d neurons, layer 0 %d", l, len(k), n)
		}
		seen := make([]bool, I)
		for _, i := range k {
			if i < 0 || i >= I || seen[i] {
				return fmt.Errorf("modernbert: layer %d: neuron %d out of range or repeated", l, i)
			}
			seen[i] = true
		}
	}
	m.restoreWeights()
	m.packMu.Lock()
	defer m.packMu.Unlock()
	for l := range m.Layers {
		L := &m.Layers[l]
		wi := make([]float32, 2*n*H)
		wo := make([]float32, H*n)
		for j, i := range keep[l] {
			copy(wi[j*H:(j+1)*H], L.Wi.W[i*H:(i+1)*H])             // input
			copy(wi[(n+j)*H:(n+j+1)*H], L.Wi.W[(I+i)*H:(I+i+1)*H]) // gate
			for h := 0; h < H; h++ {
				wo[h*n+j] = L.WoMLP.W[h*I+i]
			}
		}
		L.Wi.W, L.Wi.Shape, L.Wi.G = wi, []int{2 * n, H}, nil
		L.WoMLP.W, L.WoMLP.Shape, L.WoMLP.G = wo, []int{H, n}, nil
	}
	m.Cfg.Intermediate = n
	m.packed = nil
	m.compact = false
	m.source = nil
	return nil
}
