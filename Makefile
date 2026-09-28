# indecis s'appuie sur le SIMD expérimental de Go 1.27. Sans lui, tout
# fonctionne aussi, en scalaire et plusieurs fois plus lentement.
GOEXPERIMENT ?= simd
export GOEXPERIMENT

# Gabarits variés (multilingues, obfuscations) pour les fixtures du
# tokenizer : ceux du détecteur d'injections, dans le dépôt voisin.
FIXTURE_TEMPLATES ?= ../xolo-plugin-injection-guard/model/templates
BEKKO_DIR ?= $(HOME)/.cache/indecis/models/bekko-embedding-v1-a8m
ORACLE := tools/oracle/.venv/bin/python

.PHONY: test test-scalar bench cli serve oracle fixtures clean

test:
	go test ./...
	cd teacher && go test ./...
	cd decision && go test ./...

# Le repli scalaire doit rester correct : même suite, sans SIMD.
test-scalar:
	GOEXPERIMENT= go test ./...

bench:
	go test -run xxx -bench . ./internal/linalg/ ./internal/modernbert/ ./tokenizer/

# Commande indecis (synth, train, eval, predict, compact, split).
cli:
	go build -o bin/indecis ./cmd/indecis

# Serveur HTTP compatible avec l'API de décision TypeSafe / OpenRouter.
serve:
	cd decision && go build -o ../bin/indecis-serve ./cmd/indecis-serve

# Environnement Python de l'oracle de parité (tests uniquement).
oracle:
	uv venv --python 3.12 tools/oracle/.venv
	uv pip install --python $(ORACLE) -r tools/oracle/requirements.txt --index-strategy unsafe-best-match

# Régénère les fixtures de parité à partir de transformers / tokenizers.
fixtures:
	go run ./cmd/indecis synth -templates $(FIXTURE_TEMPLATES) -n 600 -seed 42 -out /tmp/indecis-synth-sample.jsonl
	$(ORACLE) tools/oracle/tokenizer_fixtures.py --model $(BEKKO_DIR) --corpus /tmp/indecis-synth-sample.jsonl --out testdata/bekko/tokenizer_cases.jsonl
	cd tools/oracle && ../../$(ORACLE) forward_fixtures.py --model $(BEKKO_DIR) --out ../../testdata/bekko
	cd tools/oracle && ../../$(ORACLE) train_step_fixtures.py --model $(BEKKO_DIR) --out ../../testdata/bekko

clean:
	rm -rf bin
