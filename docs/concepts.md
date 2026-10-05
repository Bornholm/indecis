# Concepts

An indecis model answers questions about a text with probabilities. It does not generate text. It is a small pretrained encoder, fully fine-tuned, with one head per question.

## The question types

| Type | Answer | Label in the examples |
| --- | --- | --- |
| `noul` | `P`, the probability of "yes" | boolean, or probability (soft label) |
| `choice` | `Choice` and the distribution `Probs` | option name, or distribution |
| `score` | `Score` (expected level), `Choice` (most likely level), `Probs` | level, by name or index |
| `spans` | `Spans`: the passages found, with type, byte offsets and confidence | list of passages `{"start", "end", "type"}` |

One pass of the encoder answers every question of a schema. An example may label only some of the questions; the others do not count in its loss.

```go
schema := indecis.Schema{
    indecis.NewNoul("human", "Does the message ask for a human agent?"),
    indecis.NewChoice("topic", "What is the message about?", "delivery", "billing", "account"),
    indecis.NewScore("urgency", "Urgency", "low", "medium", "high"),
}
m, _ := indecis.New(backboneDir, schema, 1)
_ = m.Fit(ctx, train, indecis.DefaultTrainOptions())
_, _ = m.Calibrate(ctx, calib)
_ = m.Save("model")
```

A `score` learns one threshold per level ("above low", "above medium"), so the order of the levels matters: mistaking low for high costs more than mistaking low for medium.

## Passages in a text

A `spans` question finds passages and types them: the people, places and organizations of a text, its personal data. Its options are the types.

```go
schema := indecis.Schema{indecis.NewSpans("entities", "Named entities", "PER", "LOC", "ORG", "MISC")}
```

```json
{"text": "Jean Dupont habite à Paris.", "labels": {"entities": [
  {"start": 0, "end": 11, "type": "PER"}, {"start": 22, "end": 27, "type": "LOC"}]}}
```

Offsets are bytes of the UTF-8 text, as Go slices them. An empty list says the text has no passage; passages may not overlap.

The head reads every token instead of the pooled vector and tags it: outside, beginning or inside of a passage of each type. Decoding keeps the most probable sequence where an inside tag follows a tag of the same type, then turns the tokens back into byte offsets (`tokenizer.EncodeOffsets`, identical to the reference library). A token sometimes carries punctuation next to a name, as in `Lamy,`: opening and closing punctuation at the edges of a passage is trimmed, dots excepted.

Texts longer than the model's length (`WithMaxLen`) are read in windows that overlap by a quarter, in training as in inference. Training tags every token of every window, so the head also learns passages cut by an edge. In inference, each token takes its tags from the window where it is furthest from an edge. The other questions of the schema read the first window, the truncation they always had. A model with a `spans` question reads no (context, text) pairs.

`WithSpanBias` favors passages in the decoding: more passages and longer ones, more recall for less precision, without training again. `Save` keeps the bias, and `SetSpanBias` or an option given to `Load` changes it.

`Evaluate` counts a passage when its type and both bounds are exact, and reports precision, recall, F1 and F2 (recall weighed twice) per type. `Calibrate` sets the temperature of the tags.

## Calibrated probabilities

After training, `Calibrate` sets one temperature per question on held-out examples. A probability of 0.8 then means that about 80% of the texts scored that way are actually positive. `Evaluate` measures this calibration (ECE, NLL, Brier score).

A model learns with some share of positives, often half. In production that share can be very different, such as 2% of attacks. `Info().TrainPrior` gives the training share, and `calibrate.PriorShift` adjusts a probability to the real one.

## (context, text) pairs

With `WithPairs()`, the model reads two texts: a context and the text to judge. The prompt-injection detector reads the system prompt with the message this way, which lets it recognize a request that takes the assistant out of its role. When a pair is longer than the maximum length, the context keeps its beginning, over at least a third of the tokens, and the text to judge gets the rest. If the text is short, the context takes back the free room.

## Fixed answers or open options

A model trained with `Fit` answers the questions of its schema and nothing else. Its options are outputs of its heads: adding one means retraining. In return, its answers are calibrated and do not depend on how the options are described.

Open mode compares the text with the description of each option, through embeddings (`ChooseNearest`, `DecideOpen`). The options are given at call time and can change every time. `FitEmbeddings` fine-tunes the encoder for this mode. See [open-categories.md](open-categories.md).

The two combine: the decision server answers learned questions with the heads, and any other question in open mode.

## Backbones

The default backbone is [bekko-embedding-v1-a8m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m): a 4-layer, 384-dimension ModernBERT, 7.7M parameters outside the embedding table, over 100 languages, MIT license. [bekko-embedding-v1-a25m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a25m) has the same architecture with 13 layers: about 10 points better on fine-grained categories, for three times the compute.

Any ModernBERT backbone in transformers format (`config.json`, `model.safetensors`, a Gemma-style `tokenizer.json`) should work; only these two have been tested.
