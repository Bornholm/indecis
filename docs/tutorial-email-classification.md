# Classifying emails with indecis-serve

This tutorial starts an HTTP server that classifies emails among categories you choose at every call. No category is fixed in the model: you can add, rename or remove categories from one request to the next.

The server speaks the TypeSafe decision API, the one OpenRouter relays under `/api/alpha/decisions`. A client written for Jev therefore works by changing only the URL.

The example emails are in French, like the data the model was fine-tuned on; the backbone is multilingual and English emails work too.

## Prerequisites

- Go 1.27 and `make`.
- `curl`, and optionally `jq` to read the answers.
- An x86-64 CPU. With AVX-VNNI (Intel since Alder Lake, AMD since Zen 4), the server computes in int8, twice as fast.

Run every command from the root of the `indecis` repository.

## 1. Build the server

```bash
make serve
```

The binary is written to `bin/indecis-serve`.

## 2. Choose a model

Two options, depending on your time.

### Option A: the backbone, no training

The quickest way to try. Download bekko-embedding-v1-a8m (MIT license, 210 MB):

```bash
MODEL=~/.cache/indecis/models/bekko-embedding-v1-a8m
mkdir -p $MODEL
for f in config.json model.safetensors tokenizer.json; do
  curl -sL -o $MODEL/$f https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m/resolve/main/$f
done
```

### Option B: a model fine-tuned for emails

Fine-tuning improves the classification, especially in French. It starts from the option A backbone, so download that first.

The next commands download the proof-of-concept data, then train the model on support tickets and on 3,000 French emails produced by the templates of `examples/email-triage/synth`. Allow about forty minutes on a laptop, plus a few minutes of evaluation.

```bash
export GOEXPERIMENT=simd   # without SIMD, training takes several hours
go run ./examples/email-triage prepare
go run ./examples/email-triage train-embed -n 3000 -epochs 1 -synth 3000 \
    -out ~/.cache/indecis/runs/courriels
MODEL=~/.cache/indecis/runs/courriels
```

The support tickets (Tobi-Bueck/customer-support-tickets) are under the CC-BY-NC license: a model trained on them is only fit for experiments.

### What to expect

Accuracy measured on category lists the model never saw:

| Test set | Backbone, name only | Fine-tuned, name only | Fine-tuned, name and 5 examples |
| --- | --- | --- | --- |
| imnim, 10 categories | 69% | 81% | 94% |
| Enron translated into French, 13 categories | 24% | 35% | 50% |

The "fine-tuned" columns come from a model also trained on 2,000 Enron emails labeled by LLMs and on translations, which option B does not redo; option B alone has not been measured. Either way, a few examples per category (step 5) help more than fine-tuning.

## 3. Start the server

```bash
bin/indecis-serve -model courriels=$MODEL -addr 127.0.0.1:8080
```

`courriels` is the name requests use to designate the model. The server logs to standard error and stays in the foreground; stop it with Ctrl+C.

In another terminal, check that it answers:

```bash
curl -s localhost:8080/healthz
curl -s localhost:8080/api/alpha/models | jq
```

Useful options:

| Option | Effect |
| --- | --- |
| `-model name=dir` | served model, repeatable; the first one is the default |
| `-addr` | listen address, `127.0.0.1:8080` by default |
| `-api-key` | requires `Authorization: Bearer <key>` (or the `INDECIS_API_KEY` variable) |
| `-threads` | maximum cores per request, all by default; a text under 1,024 tokens uses only one |
| `-max-len` | maximum tokens read per text, 256 by default (see "Limits") |
| `-max-concurrent` | decisions computed at once, the number of cores by default; the others wait |
| `-memory-limit` | soft memory limit, in MiB |
| `-int8=false` | computes in float32 instead |
| `-embed-cache` | number of option embeddings kept in memory, 4,096 by default |

## 4. Classify an email

The categories are the `criteria` of a `choice` question. Each category has a name and a one-line description:

```bash
curl -s localhost:8080/api/alpha/decisions -d '{
  "model": "courriels",
  "state": "Objet : Relance facture 2024-118\n\nBonjour, sauf erreur de notre part, la facture du 12 mars reste impayée. Merci de procéder au règlement sous huitaine.",
  "questions": {
    "categorie": {
      "type": "choice",
      "instructions": "Catégorie du courriel",
      "criteria": {
        "Support technique": "Panne, bug, problème d’accès",
        "Facturation et paiement": "Factures, paiements, relances, remboursements",
        "Ressources humaines": "Recrutement, congés, paie",
        "Réunion et agenda": "Invitation, planification de réunion"
      }
    }
  }
}' | jq
```

With the fine-tuned model, the answer gives the chosen category, the probability of each category, and the confidence:

```json
{
  "model": "courriels",
  "answers": {
    "categorie": {
      "type": "choice",
      "choice": "Facturation et paiement",
      "probabilities": {
        "Facturation et paiement": 0.997,
        "Ressources humaines": 0.002,
        "Réunion et agenda": 0.0001,
        "Support technique": 0.0007
      },
      "confidence": 0.997
    }
  },
  "usage": { "input_tokens": 50, "output_tokens": 0, "total_tokens": 50 }
}
```

To get only the category, replace `| jq` with `| jq -r .answers.categorie.choice`.

In these examples, apostrophes inside the texts are typographic (’): a straight apostrophe would close the shell's single-quoted string. Step 7 shows how to send any text without worrying about it.

The question name `categorie` is free. Only avoid `match`: it is the model's internal name, and a question named that way would not go through the classification.

## 5. Define a category by examples

A description can also be an object that carries example emails:

```bash
curl -s localhost:8080/api/alpha/decisions -d '{
  "model": "courriels",
  "state": "Le VPN ne fonctionne plus depuis ce matin, je ne peux pas travailler.",
  "questions": {
    "dossier": {
      "type": "choice",
      "instructions": "Dossier de rangement",
      "criteria": {
        "Comptabilité": {
          "description": "Factures et paiements",
          "examples": ["Votre facture n° 4521 est disponible", "Relance : paiement en retard de 30 jours"]
        },
        "Informatique": {
          "examples": ["L’imprimante du 2e étage est en panne", "Impossible de me connecter à la messagerie"]
        },
        "Agenda": {
          "examples": ["Réunion de lancement jeudi à 10 h", "Pouvez-vous décaler le point de lundi ?"]
        }
      }
    }
  }
}' | jq -r .answers.dossier.choice
```

One example per category already helps; five gave the best measured results. Take them from real emails that are already classified, as different from one another as possible.

This is still the TypeSafe API format, which accepts a description as text, object or array. The `examples` field is an indecis convention: Jev, or another compatible service, would read the whole object as a description.

## 6. Ask several questions at once

A request can classify the same email along several lists. For urgency, a `score` question goes from the lowest level to the highest:

```bash
curl -s localhost:8080/api/alpha/decisions -d '{
  "model": "courriels",
  "state": "Bonjour, le serveur de production est tombé, les clients ne peuvent plus commander. Merci d’intervenir au plus vite.",
  "questions": {
    "service": {"type": "choice", "instructions": "Service concerné", "criteria": {
      "Informatique": "Systèmes, serveurs, logiciels, accès",
      "Commercial": "Ventes, devis, clients",
      "Comptabilité": "Factures et paiements"}},
    "urgence": {"type": "score", "instructions": "Urgence du courriel", "criteria": [
      "Aucune échéance",
      "À traiter dans la semaine",
      "À traiter immédiatement"]}
  }
}' | jq '{service: .answers.service.choice, urgence: .answers.urgence.score}'
```

`score` lies between 0 (the first level) and the number of levels minus one. It is the expected level, not an integer.

On this email (the production server is down), the fine-tuned model answers "Informatique" (IT), but the backbone alone answers "Commercial". Both put the urgency at about 1.1, "within the week", although production is stopped. Embeddings capture urgency poorly. If it matters to you, try describing the levels with examples (step 5) and measure the result on your emails.

Yes/no (`noul`) questions work poorly in open mode, because the model barely tells a statement from its negation. Prefer a `choice` or `score` question with described options.

## 7. Classify an email stored in a file

A real email contains quotes and line breaks that need escaping. `jq` builds the request cleanly from the email file and a questions file:

```bash
cat > questions.json <<'EOF'
{
  "categorie": {
    "type": "choice",
    "instructions": "Catégorie du courriel",
    "criteria": {
      "Support technique": "Panne, bug, problème d'accès",
      "Facturation et paiement": "Factures, paiements, relances, remboursements",
      "Ressources humaines": "Recrutement, congés, paie",
      "Réunion et agenda": "Invitation, planification de réunion"
    }
  }
}
EOF

printf 'Objet : Congés\n\nBonjour, je souhaiterais poser mes congés du 4 au 15 août.\nMerci !\n' > courriel.txt

jq -n --rawfile mail courriel.txt --slurpfile q questions.json \
    '{model: "courriels", state: $mail, questions: $q[0]}' \
  | curl -s localhost:8080/api/alpha/decisions -d @- \
  | jq -r .answers.categorie.choice
```

The model reads only the first 256 tokens, the subject and the start of the body. Remove long quotes of previous messages before sending.

## 8. Protect the server with a key

```bash
KEY=$(openssl rand -hex 16)
echo "$KEY"   # to hand over to the clients
bin/indecis-serve -model courriels=$MODEL -api-key "$KEY"
```

Requests must then carry the `Authorization: Bearer <key>` header. In the second terminal, reuse the printed value:

```bash
KEY=<the value printed by echo>
curl -s localhost:8080/api/alpha/decisions -H "Authorization: Bearer $KEY" -d '{
  "model": "courriels",
  "state": "Pouvez-vous me renvoyer la facture de mars ?",
  "questions": {"categorie": {"type": "choice", "instructions": "Catégorie",
    "criteria": {"Facturation": null, "Informatique": null}}}
}' | jq -r .answers.categorie.choice
```

Without the header, the server answers 401.

The server listens on `127.0.0.1` by default. To expose it on the network, pass `-addr 0.0.0.0:8080`, set a key, and put it behind a TLS proxy.

## Common errors

Errors come as `{"error": {"message": "…", "code": …}}`.

| Code | Cause |
| --- | --- |
| 400 | the body is not valid JSON |
| 401 | API key missing or wrong |
| 404 | the server serves several models and the `model` field names none of them; with a single model, an unknown or missing name designates it |
| 422 | malformed question: unknown type, missing `criteria`, missing `state` |

## Limits

- **The probabilities are not calibrated.** A 0.997 does not mean a 99.7% chance of being right. To catch an email that fits no category, set a `confidence` threshold on a few dozen examples, or add an "Other" category described by examples.
- **The model reads 256 tokens by default.** Beyond that, the text is truncated. `-max-len 4096` reads more: on a8m, a 4,096-token text takes about 0.65 s on all cores. The models were fine-tuned on 256 tokens: measure the quality before reading longer texts.
- **The step 2 numbers come from a proof of concept**: a single training run, test sets without human review. Measure on your own emails before trusting the classification.

## Going further

- `examples/email-triage` contains the data preparation, training and evaluation.
- From Go, the genai `openrouter` client targets this server with the base URL `http://127.0.0.1:8080/api/v1`, and the `typesafe` client with `http://127.0.0.1:8080/v1`.
- The library offers the same functions without a server: `Model.ChooseNearest`, `Model.PrepareCandidates` and `Model.DecideOpen`.
