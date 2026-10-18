package indecis

import (
	"context"
	"fmt"
	"math"
	"sort"
)

// Distillation and pruning: a smaller model, trained to reproduce a larger
// one.
//
// PruneMLP removes the least useful neurons of the encoder's MLP: two
// thirds of the products of a layer. Fit with TrainOptions.Teacher then
// trains the pruned model on the teacher's tags as well as on the labels:
// the teacher's distributions say, token by token, how sure it is and
// what it hesitates with, which the labels do not (Hinton et al. 2015).
// The final states of the encoder can be matched too, as in TinyBERT
// (Jiao et al. 2020) and CoFi (Xia et al. 2022).

// PruneMLP keeps keep neurons in the MLP of each layer of the encoder, the
// most useful on the texts: those whose mean activity |gelu(input)·gate|,
// times the norm of their output weights, is largest. The texts are read
// as in inference, in windows. Every layer keeps as many neurons: the
// model stays a ModernBERT, readable by transformers.
//
// The model loses quality until it is trained again: see Fit and
// TrainOptions.Teacher.
func (m *Model) PruneMLP(ctx context.Context, texts []string, keep int) error {
	I := m.enc.Cfg.Intermediate
	if keep <= 0 || keep > I {
		return fmt.Errorf("indecis: keep %d of %d MLP neurons", keep, I)
	}
	seqs, err := m.readingSeqs(texts)
	if err != nil {
		return err
	}
	if len(seqs) == 0 {
		return fmt.Errorf("indecis: no text to measure the MLP on")
	}
	act := make([][]float64, len(m.enc.Layers))
	for l := range act {
		act[l] = make([]float64, I)
	}
	tokens := 0
	for start := 0; start < len(seqs); start += 32 {
		if err := ctx.Err(); err != nil {
			return err
		}
		group := seqs[start:min(start+32, len(seqs))]
		n := 0
		for _, s := range group {
			n += len(s)
		}
		a, err := m.enc.MLPActivity(group)
		if err != nil {
			return err
		}
		for l := range a {
			for i, v := range a[l] {
				act[l][i] += v * float64(n)
			}
		}
		tokens += n
	}
	norms := m.enc.MLPOutputNorms()
	idx := make([][]int, len(act))
	for l := range act {
		score := make([]float64, I)
		order := make([]int, I)
		for i := range score {
			score[i] = act[l][i] / float64(tokens) * norms[l][i]
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool { return score[order[a]] > score[order[b]] })
		idx[l] = append([]int(nil), order[:keep]...)
		sort.Ints(idx[l])
	}
	if err := m.enc.PruneMLP(idx); err != nil {
		return err
	}
	m.embedCache.clear()
	return nil
}

// readingSeqs returns the token sequences inference reads for the texts:
// the windows of a model with Spans questions, the truncated text
// otherwise.
func (m *Model) readingSeqs(texts []string) ([][]int32, error) {
	if !m.hasSpans() {
		return m.tokenizeAll(textsToInputs(texts))
	}
	prefix, suffix := m.frame()
	size := m.windowSize()
	var seqs [][]int32
	for _, text := range texts {
		ids, _ := m.tok.AppendOffsets(nil, nil, text)
		for _, w := range planWindows(len(ids), size, size/16) {
			seqs = append(seqs, append(append(append([]int32(nil), prefix...), ids[w.start:w.end]...), suffix...))
		}
	}
	return seqs, nil
}

// teaching holds, for a training batch, what the teacher computed: the
// final state of every token of the windows, laid end to end (row
// offs[b] + t for token t of window b).
type teaching struct {
	teacher *Model
	states  []float32
	offs    []int
	alpha   float64 // weight of the teacher's tags against the labels
	temp    float64
}

// teach runs the teacher on the windows of the batch.
func teach(opts TrainOptions, seqs [][]int32) (*teaching, error) {
	if opts.Teacher == nil {
		return nil, nil
	}
	states, err := opts.Teacher.enc.EncodeTokens(seqs)
	if err != nil {
		return nil, err
	}
	offs := make([]int, len(seqs)+1)
	for i, s := range seqs {
		offs[i+1] = offs[i] + len(s)
	}
	temp := opts.DistillTemp
	if temp <= 0 {
		temp = 1
	}
	return &teaching{teacher: opts.Teacher, states: states, offs: offs, alpha: opts.Distill, temp: temp}, nil
}

// checkTeacher verifies that the teacher can teach the model: same
// questions, same tokenizer, same width.
func (m *Model) checkTeacher(t *Model) error {
	if len(t.schema) != len(m.schema) {
		return fmt.Errorf("indecis: the teacher has %d questions, the model %d", len(t.schema), len(m.schema))
	}
	for i, q := range m.schema {
		tq := t.schema[i]
		if tq.Name != q.Name || tq.Kind != q.Kind || fmt.Sprint(tq.Options) != fmt.Sprint(q.Options) {
			return fmt.Errorf("indecis: question %q differs in the teacher", q.Name)
		}
	}
	if t.enc.Cfg.Hidden != m.enc.Cfg.Hidden || t.enc.Cfg.Vocab != m.enc.Cfg.Vocab {
		return fmt.Errorf("indecis: the teacher's encoder is not as wide as the model's")
	}
	if fmt.Sprint(t.tok.Encode("Jean Dupont habite à Paris.")) != fmt.Sprint(m.tok.Encode("Jean Dupont habite à Paris.")) {
		return fmt.Errorf("indecis: the teacher's tokenizer differs")
	}
	return nil
}

// softmaxT returns softmax(z/T).
func softmaxT(z []float64, T float64) []float64 {
	p := make([]float64, len(z))
	mx := math.Inf(-1)
	for _, v := range z {
		mx = max(mx, v/T)
	}
	var sum float64
	for i, v := range z {
		p[i] = math.Exp(v/T - mx)
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
	return p
}

// hiddenLossGrad adds to dHidden the gradient of weight times the mean
// squared distance, per value, between the final states of the model and
// those of the teacher, over every token of the batch, and returns that
// loss. hidden is [B·T, H], the batch padded to T tokens.
func (m *Model) hiddenLossGrad(batch []*encoded, hidden, dHidden []float32, T int, tc *teaching, weight float64) float64 {
	H := m.enc.Cfg.Hidden
	n := tc.offs[len(batch)]
	if n == 0 || weight == 0 {
		return 0
	}
	scale := weight / float64(n*H)
	var loss float64
	for bi, e := range batch {
		for t := range e.ids {
			s := hidden[(bi*T+t)*H : (bi*T+t+1)*H]
			ts := tc.states[(tc.offs[bi]+t)*H : (tc.offs[bi]+t+1)*H]
			d := dHidden[(bi*T+t)*H : (bi*T+t+1)*H]
			for k := range s {
				diff := float64(s[k] - ts[k])
				loss += diff * diff * scale
				d[k] += float32(2 * diff * scale)
			}
		}
	}
	return loss
}
