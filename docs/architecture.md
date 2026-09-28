# Architecture

## Packages

| Package | Role |
| --- | --- |
| `indecis` | schema, model, `Fit`, `Calibrate`, `Evaluate`, `Decide`, open mode, `Save`/`Load` |
| `dataset` | JSONL example format, splits (`HoldOut`) |
| `dataset/synth` | examples from templates, labeled by construction |
| `tokenizer` | Gemma-family BPE tokenizer, identical to Hugging Face's `tokenizers` library |
| `calibrate` | log-odds, production prior shift, evidence fusion |
| `cmd/indecis` | command line: `synth`, `train`, `eval`, `predict`, `compact`, `split` |
| `internal/modernbert` | ModernBERT encoder, forward and backward written by hand |
| `internal/linalg` | matrix products (portable SIMD, AVX2 assembly, scalar; int8 AVX-VNNI), vector operations, parallelism |
| `internal/optim` | AdamW, and sparse Adam for the embedding table |
| `internal/safetensors` | safetensors reading (memory-mapped) and writing |
| `teacher` (separate module) | labeling by LLMs, consensus, `indecis-teach` command |
| `decision` (separate module) | genai provider, `indecis-serve` server |

`teacher` and `decision` are separate modules so that the library depends on nothing but the Go standard library.

## Training

The encoder is fully fine-tuned, with forward and backward passes written by hand: the architecture is fixed, and a generic autograd would only add allocations.

The embedding table holds 93% of the parameters. Its gradient is sparse (only the rows of the tokens present in the batch), and its optimizer is a "lazy" Adam, like `torch.optim.SparseAdam`, with no weight decay. Without this, the dense gradient and the AdamW moments would take 1.6 GB.

Reductions (sums of gradients) run over fixed-size slices, then in order: a training run gives the same model bit for bit, whatever the number of cores.

Throughput on a laptop (Core Ultra 7 265U, all cores): about 2,200 tokens per second, or half an hour for 20,000 examples over two epochs.

## Parity with the reference implementation

Every stage is compared with PyTorch, transformers and tokenizers. The fixtures come from `tools/oracle` and are versioned in `testdata/`.

| Stage | Test | Result |
| --- | --- | --- |
| Tokenizer | 3,653 texts (edge cases, real corpus, hostile random strings) and 604 pairs | identical ids |
| Forward | 13 texts, up to 326 tokens | max difference 3·10⁻⁶ on the pooled embedding |
| Backward | every parameter of a small model, finite differences (Richardson) | 1,808 out of 1,808 |
| Training step | gradients, clipping and AdamW step on bekko | identical to PyTorch; a 10% error in one derivative is caught |
| Inference path | against the training forward, up to 2,100 tokens | max difference 1·10⁻⁷ |

Tests that need the bekko weights are skipped when the weights are missing. Download the model to `~/.cache/indecis/models/bekko-embedding-v1-a8m`, or point `INDECIS_BEKKO_DIR` to it.

## Build and test

```bash
make test          # all modules, SIMD enabled (GOEXPERIMENT=simd)
make test-scalar   # same suite without SIMD: the fallback must stay correct
make cli serve     # bin/indecis, bin/indecis-serve
make bench
make oracle        # Python environment of the oracle (tests only)
make fixtures      # regenerates the parity fixtures
```

`INDECIS_NOASM=1` turns the assembly kernels off, to compare or work around them. Continuous integration (`.github/workflows/ci.yml`) runs the suite in the three builds (SIMD, scalar, no assembly), with the bekko weights, plus the race detector.

Go 1.27 SIMD is experimental. indecis builds and runs without it, several times slower. A SIMD function inlined into a closure crashes the Go 1.27 compiler, so SIMD functions are marked `//go:noinline`.
