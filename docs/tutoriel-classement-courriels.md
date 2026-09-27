# Classer des courriels avec indecis-serve

Ce tutoriel lance un serveur HTTP qui classe des courriels parmi des catégories que vous choisissez à chaque appel. Aucune catégorie n'est figée dans le modèle : vous pouvez en ajouter, en renommer ou en retirer d'une requête à l'autre.

Le serveur parle l'API de décision de TypeSafe, celle qu'OpenRouter relaie sous `/api/alpha/decisions`. Un client écrit pour Jev fonctionne donc en changeant seulement l'URL.

## Prérequis

- Go 1.27 et `make`.
- `curl`, et `jq` pour lire les réponses (facultatif).
- Un processeur x86-64. Avec AVX-VNNI (Intel depuis Alder Lake, AMD depuis Zen 4), le serveur calcule en int8, deux fois plus vite.

Toutes les commandes se lancent depuis la racine du dépôt `indecis`.

## 1. Compiler le serveur

```bash
make serve
```

Le binaire est écrit dans `bin/indecis-serve`.

## 2. Choisir un modèle

Deux options, selon le temps dont vous disposez.

### Option A : le backbone, sans entraînement

C'est le plus rapide pour essayer. On télécharge bekko-embedding-v1-a8m (licence MIT, 210 Mo) :

```bash
MODEL=~/.cache/indecis/models/bekko-embedding-v1-a8m
mkdir -p $MODEL
for f in config.json model.safetensors tokenizer.json; do
  curl -sL -o $MODEL/$f https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m/resolve/main/$f
done
```

### Option B : un modèle affiné pour les courriels

L'affinage améliore le classement, surtout en français. Il part du backbone de l'option A : téléchargez-le d'abord.

La commande suivante télécharge les données de la preuve de concept, puis entraîne le modèle sur des tickets de support et sur 3 000 courriels français produits par les gabarits de `examples/email-triage/synth`. Comptez une quarantaine de minutes sur un portable, puis quelques minutes d'évaluation.

```bash
export GOEXPERIMENT=simd   # sans le SIMD, l'entraînement prend plusieurs heures
go run ./examples/email-triage prepare
go run ./examples/email-triage train-embed -n 3000 -epochs 1 -synth 3000 \
    -out ~/.cache/indecis/runs/courriels
MODEL=~/.cache/indecis/runs/courriels
```

Les tickets de support (Tobi-Bueck/customer-support-tickets) sont sous licence CC-BY-NC : un modèle entraîné dessus ne convient qu'à des essais.

### Ce qu'on peut en attendre

Exactitude mesurée sur des listes de catégories que le modèle n'a jamais vues :

| Jeu de test | Backbone, nom seul | Affiné, nom seul | Affiné, nom et 5 exemples |
| --- | --- | --- | --- |
| imnim, 10 catégories | 69 % | 81 % | 94 % |
| Enron traduit en français, 13 catégories | 24 % | 35 % | 50 % |

La colonne « Affiné » vient d'un modèle entraîné aussi sur 2 000 courriels Enron étiquetés par des LLM et sur des traductions, ce que l'option B ne refait pas ; je n'ai pas mesuré l'option B seule. Dans tous les cas, donner quelques exemples par catégorie (étape 5) rapporte plus que l'affinage.

## 3. Lancer le serveur

```bash
bin/indecis-serve -model courriels=$MODEL -addr 127.0.0.1:8080
```

`courriels` est le nom sous lequel les requêtes désignent le modèle. Le serveur écrit ses journaux sur la sortie d'erreur et reste au premier plan ; arrêtez-le avec Ctrl+C.

Dans un autre terminal, vérifiez qu'il répond :

```bash
curl -s localhost:8080/healthz
curl -s localhost:8080/api/alpha/models | jq
```

Options utiles :

| Option | Effet |
| --- | --- |
| `-model nom=répertoire` | modèle servi, répétable ; le premier sert par défaut |
| `-addr` | adresse d'écoute, `127.0.0.1:8080` par défaut |
| `-api-key` | exige `Authorization: Bearer <clé>` (ou la variable `INDECIS_API_KEY`) |
| `-threads` | cœurs par requête ; 1 par défaut, le plus rapide pour une requête isolée |
| `-int8=false` | revient au calcul en float32 |
| `-embed-cache` | nombre de textes dont le plongement reste en mémoire, 4 096 par défaut |

## 4. Classer un courriel

Les catégories sont les `criteria` d'une question de type `choice`. Chaque catégorie a un nom et une description d'une ligne :

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

Avec le modèle affiné, la réponse donne la catégorie retenue, la probabilité de chaque catégorie et la confiance :

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

Pour n'obtenir que la catégorie, ajoutez `| jq -r .answers.categorie.choice` à la place de `| jq`.

Dans ces exemples, les apostrophes des textes sont typographiques (’) : une apostrophe droite fermerait la chaîne entre guillemets simples du shell. L'étape 7 montre comment envoyer un texte quelconque sans s'en soucier.

Le nom `categorie` est libre. Évitez seulement `match` : c'est le nom interne du modèle, et une question qui le porte ne passerait pas par le classement.

## 5. Définir une catégorie par des exemples

Une description peut aussi être un objet qui porte des exemples de courriels :

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

Un seul exemple par catégorie aide déjà ; cinq ont donné les meilleurs résultats mesurés. Prenez-les dans de vrais courriels déjà classés, aussi différents que possible les uns des autres.

Ce format reste celui de l'API TypeSafe, qui accepte une description sous forme de texte, d'objet ou de tableau. Le champ `examples` est une convention d'indecis : Jev, ou un autre service compatible, lirait l'objet entier comme une description.

## 6. Poser plusieurs questions à la fois

Une requête peut classer le même courriel selon plusieurs listes. Pour l'urgence, une question `score` va du niveau le plus bas au plus haut :

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

`score` vaut entre 0 (premier niveau) et le nombre de niveaux moins un. C'est l'espérance du niveau, pas un entier.

Sur ce courriel, le modèle affiné répond « Informatique », mais le backbone seul répond « Commercial ». Tous deux estiment l'urgence à 1,1 environ, soit « dans la semaine », alors que la production est arrêtée. Les plongements saisissent mal l'urgence. Si elle compte pour vous, essayez de décrire les niveaux par des exemples (étape 5) et mesurez le résultat sur vos courriels.

Les questions oui/non (`noul`) marchent mal en mode ouvert, car le modèle distingue mal une affirmation de sa négation. Préférez une question `choice` ou `score` aux options décrites.

## 7. Classer un courriel stocké dans un fichier

Un vrai courriel contient des guillemets et des sauts de ligne qu'il faut échapper. `jq` construit la requête proprement à partir du fichier du courriel et d'un fichier de questions :

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

Le modèle ne lit que les 256 premiers tokens, soit l'objet et le début du corps. Retirez les longues citations des messages précédents avant l'envoi.

## 8. Protéger le serveur par une clé

```bash
CLE=$(openssl rand -hex 16)
echo "$CLE"   # à transmettre aux clients
bin/indecis-serve -model courriels=$MODEL -api-key "$CLE"
```

Les requêtes doivent alors porter l'en-tête `Authorization: Bearer <clé>`. Dans le second terminal, reprenez la valeur affichée :

```bash
CLE=<la valeur affichée par echo>
curl -s localhost:8080/api/alpha/decisions -H "Authorization: Bearer $CLE" -d '{
  "model": "courriels",
  "state": "Pouvez-vous me renvoyer la facture de mars ?",
  "questions": {"categorie": {"type": "choice", "instructions": "Catégorie",
    "criteria": {"Facturation": null, "Informatique": null}}}
}' | jq -r .answers.categorie.choice
```

Sans l'en-tête, le serveur répond 401.

Le serveur écoute sur `127.0.0.1` par défaut. Pour l'exposer au réseau, passez `-addr 0.0.0.0:8080`, mettez une clé et placez-le derrière un proxy TLS.

## Erreurs courantes

Les erreurs arrivent au format `{"error": {"message": "…", "code": …}}`.

| Code | Cause |
| --- | --- |
| 400 | le corps n'est pas du JSON valide |
| 401 | clé d'API absente ou fausse |
| 404 | le serveur sert plusieurs modèles et le champ `model` n'en désigne aucun ; avec un seul modèle, un nom inconnu ou absent désigne celui-ci |
| 422 | question mal formée : type inconnu, `criteria` manquants, `state` absent |

## Limites à connaître

- **Les probabilités ne sont pas calibrées.** Un 0,997 ne veut pas dire 99,7 % de chances d'avoir raison. Pour repérer un courriel qui ne relève d'aucune catégorie, fixez un seuil de `confidence` sur quelques dizaines d'exemples, ou ajoutez une catégorie « Autre » décrite par des exemples.
- **Le modèle lit 256 tokens au plus.** Au-delà, le texte est tronqué.
- **Les chiffres de l'étape 2 viennent d'une preuve de concept** : un seul entraînement, des tests sans relecture humaine. Mesurez sur vos propres courriels avant de vous fier au classement.

## Aller plus loin

- `examples/email-triage` contient la préparation des données, l'entraînement et l'évaluation.
- Depuis Go, le client `openrouter` de genai vise ce serveur avec l'URL de base `http://127.0.0.1:8080/api/v1`, et le client `typesafe` avec `http://127.0.0.1:8080/v1`.
- La bibliothèque expose les mêmes fonctions sans serveur : `Model.ChooseNearest`, `Model.PrepareCandidates` et `Model.DecideOpen`.
