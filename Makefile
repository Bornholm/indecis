# indecis s'appuie sur le SIMD expérimental de Go 1.27. Sans lui, tout
# fonctionne aussi, en scalaire et plusieurs fois plus lentement.
GOEXPERIMENT ?= simd
export GOEXPERIMENT

XOLO_PLUGINS_DIR ?= ../xolo/bin/plugins
BEKKO_DIR ?= $(HOME)/.cache/indecis/models/bekko-embedding-v1-a8m
ORACLE := tools/oracle/.venv/bin/python

.PHONY: test test-scalar bench plugin install-plugin serve oracle fixtures clean

test:
	go test ./...
	cd teacher && go test ./...
	cd decision && go test ./...
	cd plugins/injection-detector && go test ./...

# Le repli scalaire doit rester correct : même suite, sans SIMD.
test-scalar:
	GOEXPERIMENT= go test ./...

bench:
	go test -run xxx -bench . ./internal/linalg/ ./internal/modernbert/ ./tokenizer/

plugin:
	cd plugins/injection-detector && CGO_ENABLED=0 go build -o ../../bin/injection-detector .

install-plugin: plugin
	install -m 0755 bin/injection-detector $(XOLO_PLUGINS_DIR)/injection-detector

# Serveur HTTP compatible avec l'API de décision TypeSafe / OpenRouter.
serve:
	cd decision && go build -o ../bin/indecis-serve ./cmd/indecis-serve

# Environnement Python de l'oracle de parité (tests uniquement).
oracle:
	uv venv --python 3.12 tools/oracle/.venv
	uv pip install --python $(ORACLE) -r tools/oracle/requirements.txt --index-strategy unsafe-best-match

# Régénère les fixtures de parité à partir de transformers / tokenizers.
fixtures:
	go run ./tools/synthsample -out /tmp/indecis-synth-sample.jsonl
	$(ORACLE) tools/oracle/tokenizer_fixtures.py --model $(BEKKO_DIR) --corpus /tmp/indecis-synth-sample.jsonl --out testdata/bekko/tokenizer_cases.jsonl
	cd tools/oracle && ../../$(ORACLE) forward_fixtures.py --model $(BEKKO_DIR) --out ../../testdata/bekko
	cd tools/oracle && ../../$(ORACLE) train_step_fixtures.py --model $(BEKKO_DIR) --out ../../testdata/bekko

clean:
	rm -rf bin
