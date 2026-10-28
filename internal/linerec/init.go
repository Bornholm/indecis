package linerec

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/bornholm/indecis/safetensors"
)

// New creates a model with fresh weights, initialized as PyTorch
// initializes the reference's layers: convolution weights and biases
// uniform in ±1/√fan_in, normalizations at identity, row prior at zero.
func New(cfg Config, seed int64) (*Model, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	r := rand.New(rand.NewSource(seed))
	c := cfg
	tensors := map[string]safetensors.Tensor{}
	conv := func(name string, fanIn int, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		bound := 1 / math.Sqrt(float64(fanIn))
		w := make([]float32, n)
		for i := range w {
			w[i] = float32((2*r.Float64() - 1) * bound)
		}
		b := make([]float32, shape[0])
		for i := range b {
			b[i] = float32((2*r.Float64() - 1) * bound)
		}
		tensors[name+".weight"] = safetensors.Tensor{Shape: shape, Data: w}
		tensors[name+".bias"] = safetensors.Tensor{Shape: shape[:1], Data: b}
	}
	fill := func(name string, v float32, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		d := make([]float32, n)
		for i := range d {
			d[i] = v
		}
		tensors[name] = safetensors.Tensor{Shape: shape, Data: d}
	}

	conv("stem", c.Height*c.Stride, c.Dim, 1, c.Height, c.Stride)
	conv("context.patch", c.Stride*c.Stride, c.CtxChannels, 1, c.Stride, c.Stride)
	conv("context.dw", 9, c.CtxChannels, 1, 3, 3)
	delete(tensors, "context.dw.bias")
	conv("context.pw", c.CtxChannels, c.CtxDim, c.CtxChannels, 1, 1)
	conv("context.score", c.CtxDim, c.CtxHeads, c.CtxDim, 1, 1)
	fill("context.row_prior", 0, 1, c.CtxHeads, c.CtxRows(), 1)
	conv("context.proj", c.CtxDim, c.Dim, c.CtxDim, 1)
	norm := "stem_ln"
	if c.Norm == "batch" {
		norm = "stem_bn"
		fill(norm+".running_mean", 0, c.Dim)
		fill(norm+".running_var", 1, c.Dim)
	}
	fill(norm+".weight", 1, c.Dim)
	fill(norm+".bias", 0, c.Dim)
	for i := range c.Blocks {
		p := fmt.Sprintf("blocks.%d.", i)
		conv(p+"dw", c.Kernel, c.Dim, 1, c.Kernel)
		conv(p+"expand", c.Dim, c.Expand, c.Dim, 1)
		conv(p+"se.fc1", c.Expand, c.SE, c.Expand, 1)
		conv(p+"se.fc2", c.SE, c.Expand, c.SE, 1)
		conv(p+"project", c.Expand, c.Dim, c.Expand, 1)
	}
	conv("head", c.Dim, c.Vocab, c.Dim, 1)
	return FromTensors(cfg, tensors)
}

// Base is the configuration of Indra OCR's recognizer, with a layer norm
// (trainable without batch statistics) and a vocabulary of v symbols.
func Base(v int) Config {
	return Config{
		Height: 32, Window: 3, Stride: 4, Dim: 128, Expand: 256, SE: 64, Blocks: 10,
		Kernel: 9, CtxChannels: 32, CtxDim: 64, CtxHeads: 4, Vocab: v, Norm: "layer",
	}
}
