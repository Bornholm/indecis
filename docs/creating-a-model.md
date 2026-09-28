# Creating a model in five steps

This guide builds a model that sorts customer support messages: the topic (delivery, billing, account, product), the urgency (low, medium, high), and whether a human agent is needed. Everything goes through the `indecis` command, without writing Go. The example files are in `examples/guide`; the example is in French, like the support messages it sorts.

Allow ten minutes, less than one of which is training.

## 0. Prepare

```bash
make cli   # builds bin/indecis
```

You need a backbone, the pretrained model that indecis fine-tunes. The default is bekko-embedding-v1-a8m (multilingual, 210 MB, MIT license):

```bash
B=~/.cache/indecis/models/bekko-embedding-v1-a8m
mkdir -p $B
for f in config.json model.safetensors tokenizer.json; do
  curl -sL -o $B/$f https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m/resolve/main/$f
done
```

## 1. Write the schema

The schema lists the questions the model answers. There are three types (see [concepts.md](concepts.md)):

- `noul`: yes or no, answered with a probability;
- `choice`: one option among several;
- `score`: a level on an ordered scale.

```json
[
  {"name": "sujet", "kind": "choice", "instructions": "De quoi parle le message ?",
   "options": [
     {"name": "livraison", "description": "Commande, colis, retard ou suivi de livraison."},
     {"name": "facturation", "description": "Facture, paiement, remboursement, prélèvement."},
     {"name": "compte", "description": "Connexion, mot de passe, données personnelles, abonnement."},
     {"name": "produit", "description": "Fonctionnement, panne ou question sur un produit."}
   ]},
  {"name": "urgence", "kind": "score", "instructions": "Urgence du message", "options": ["basse", "moyenne", "haute"]},
  {"name": "humain", "kind": "noul", "instructions": "Le message demande-t-il l'intervention d'un conseiller ?"}
]
```

An option is either a plain name or an object with a description and examples. Descriptions only matter in open mode (step 3).

## 2. Produce examples

An example is one JSON line: a text and its labels.

```json
{"text": "Mon colis n'est pas arrivé.", "labels": {"sujet": "livraison", "urgence": "moyenne", "humain": false}}
```

There are three sources, and they combine:

- **Real texts that are already labeled**, such as the history of a support tool.
- **Templates**, which produce thousands of examples labeled by construction. `examples/guide/templates` holds four of them, one per topic (syntax in [data.md](data.md)).
- **LLMs labeling real texts**: `indecis-teach` has two models label the texts and keeps only what they agree on (see [data.md](data.md)).

The templates are enough for this guide:

```bash
bin/indecis synth -templates examples/guide/templates -n 900 -out data.jsonl
```

Four templates only produce about 960 distinct messages. If you ask for more, `synth` says so instead of repeating the same ones.

## 3. Train

```bash
bin/indecis train -backbone $B -schema examples/guide/schema.json \
    -train data.jsonl -test examples/guide/test.jsonl -epochs 3 -out model
```

Training takes 35 seconds on a laptop. The command sets 5% of the examples aside to calibrate the probabilities, then measures the model on `examples/guide/test.jsonl`: 24 messages written by hand, with wordings that the templates never produce.

```
sujet        n=24    acc=0.750 macroF1=0.727
urgence      n=24    acc=0.583 MAE=0.461
humain       n=24    acc=0.750 P=0.583 R=0.875 AUC=0.898
```

These numbers are modest, and that is expected: four templates do not cover the variety of real messages. A test set produced by the same templates would score 100%, which would tell you nothing. Add real texts to do better.

### Open mode

With `-open`, the model does not learn fixed answers. It learns to match a text with the description of an option, so the options can change with every call: adding a topic needs no retraining.

```bash
bin/indecis train -open -backbone $B -schema examples/guide/schema.json \
    -train data.jsonl -test examples/guide/test.jsonl -epochs 3 -out model-open
```

On the same 24 messages:

| Method | Topic | Urgency | Human agent |
| --- | --- | --- | --- |
| backbone only, no training (`eval -schema`) | 62.5% | 33.3% | 62.5% |
| fixed answers (`train`) | 75.0% | 58.3% | 75.0% |
| fine-tuned open mode (`train -open`) | 75.0% | 66.7% | 87.5% |

Pick according to your use: fixed answers are calibrated and do not depend on how you describe the options; open mode accepts new options. [open-categories.md](open-categories.md) covers open mode in detail.

## 4. Evaluate and use

```bash
bin/indecis eval -model model -data examples/guide/test.jsonl
```

`-by source` breaks the numbers down by a `meta` field of the examples, which helps when a test set mixes several origins.

`predict` reads one text per line, as plain text or JSON, and writes the answers:

```bash
printf 'Mon colis devait arriver lundi, toujours rien, je veux parler à quelqu'"'"'un.\nComment télécharger ma facture ?\n' \
  | bin/indecis predict -model model
```

| Message | Topic | Urgency (0 to 2) | Human agent |
| --- | --- | --- | --- |
| "Mon colis devait arriver lundi…" (parcel late, wants a person) | livraison (0.93) | 2.0 | 1.0 |
| "Comment télécharger ma facture ?" (where is my invoice) | facturation (0.98) | 0.15 | 0.03 |

For an open-mode model, pass the questions with `-schema`.

From Go:

```go
m, _ := indecis.Load("model", indecis.WithInt8())
d, _ := m.Decide(ctx, "Mon colis n'est pas arrivé")
d[0]["sujet"].Choice   // "livraison"
d[0]["humain"].P       // calibrated probability
```

## 5. Deploy

`indecis-serve` exposes the model with the TypeSafe and OpenRouter decision API (see [serving.md](serving.md)):

```bash
make serve
bin/indecis-serve -model support=model
```

`indecis compact -int8-embeddings` halves the model file with no measured loss (see [inference.md](inference.md)).

## Going further

- A complete, reproducible example with relabeled real data, a written labeling policy, and hand-reviewed references: the prompt-injection detector [xolo-plugin-injection-guard](../../xolo-plugin-injection-guard).
- A model that classifies emails among categories chosen at call time: [tutorial-email-classification.md](tutorial-email-classification.md).
