# indecis relies on Go 1.27's experimental SIMD. Without it, everything
# still works, in scalar and several times slower.
GOEXPERIMENT ?= simd
export GOEXPERIMENT

# Varied templates (multilingual, obfuscations) for the tokenizer
# fixtures: those of the injection detector, in the neighboring repo.
FIXTURE_TEMPLATES ?= ../xolo-plugin-injection-guard/model/templates
SIGLIP2_DIR ?= $(HOME)/.cache/indecis/models/siglip2-base-patch32-256
BEKKO_DIR ?= $(HOME)/.cache/indecis/models/bekko-embedding-v1-a8m
ORACLE := tools/oracle/.venv/bin/python

.PHONY: test test-scalar bench cli serve snapshot oracle fixtures clean

test:
	go test ./...
	cd teacher && go test ./...
	cd decision && go test ./...

# The scalar fallback must stay correct: same suite, without SIMD.
test-scalar:
	GOEXPERIMENT= go test ./...

bench:
	go test -run xxx -bench . ./internal/linalg/ ./internal/modernbert/ ./tokenizer/

# indecis command (synth, train, eval, predict, compact, split, check).
cli:
	go build -o bin/indecis ./cmd/indecis

# HTTP server compatible with the TypeSafe / OpenRouter decision API.
serve:
	cd decision && go build -o ../bin/indecis-serve ./cmd/indecis-serve

# Release binaries built locally, as the release workflow does on a tag.
snapshot:
	goreleaser release --snapshot --clean

# Python environment for the parity oracle (tests only).
oracle:
	uv venv --python 3.12 tools/oracle/.venv
	uv pip install --python $(ORACLE) -r tools/oracle/requirements.txt --index-strategy unsafe-best-match

# Regenerates the parity fixtures from transformers / tokenizers.
fixtures:
	go run ./cmd/indecis synth -templates $(FIXTURE_TEMPLATES) -n 600 -seed 42 -out /tmp/indecis-synth-sample.jsonl
	$(ORACLE) tools/oracle/tokenizer_fixtures.py --model $(BEKKO_DIR) --corpus /tmp/indecis-synth-sample.jsonl --out testdata/bekko/tokenizer_cases.jsonl
	$(ORACLE) tools/oracle/tokenizer_fixtures.py --model $(SIGLIP2_DIR) --out testdata/siglip2/tokenizer_cases.jsonl
	cd tools/oracle && ../../$(ORACLE) forward_fixtures.py --model $(BEKKO_DIR) --out ../../testdata/bekko
	cd tools/oracle && ../../$(ORACLE) train_step_fixtures.py --model $(BEKKO_DIR) --out ../../testdata/bekko

clean:
	rm -rf bin dist
