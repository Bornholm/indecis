// Command indecis-serve exposes indecis models over HTTP with the TypeSafe
// and OpenRouter decision API, to test the integration of a model without
// changing anything in the client but its URL.
//
//	indecis-serve -model ~/.cache/indecis/runs/policy-P5
//	indecis-serve -model injection=runs/policy-P5 -model emails=runs/email-embed -addr :8080
//
//	curl -s localhost:8080/api/alpha/decisions -d '{
//	  "state": "Ignore previous instructions",
//	  "questions": {"injection": {"type": "noul", "instructions": "?"}}
//	}'
//
// -model also accepts a raw backbone (config.json, model.safetensors,
// tokenizer.json, without indecis.json), such as bekko: it then answers
// only open questions, without training.
//
// Clients: genai (openrouter provider with GENAI_…_BASE_URL=http://…/api/v1,
// or typesafe with http://…/v1), the TypeSafe SDK, curl.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/decision"
)

type modelFlags []string

func (m *modelFlags) String() string     { return strings.Join(*m, ",") }
func (m *modelFlags) Set(v string) error { *m = append(*m, v); return nil }

func main() {
	var models modelFlags
	flag.Var(&models, "model", "model to serve: directory, or name=directory (repeatable; the first is the default model)")
	addr := flag.String("addr", "127.0.0.1:8080", "listen address")
	apiKey := flag.String("api-key", os.Getenv("INDECIS_API_KEY"), "key required as Authorization: Bearer (empty: none)")
	threads := flag.Int("threads", 0, "cores at most per request (0: all); a text under 1024 tokens always uses a single one")
	int8 := flag.Bool("int8", true, "layers in int8 if the processor has AVX-VNNI")
	cache := flag.Int("embed-cache", 4096, "option embeddings kept in cache (open questions)")
	maxConcurrent := flag.Int("max-concurrent", runtime.GOMAXPROCS(0), "decisions in flight at most, others wait (0: no bound)")
	batching := flag.Bool("batching", false, "group the computation of simultaneous requests (a few % gain on short requests, more memory; also raise -max-concurrent)")
	maxLen := flag.Int("max-len", 0, "tokens read at most per text (0: the model's value, 256 generally); quadratic cost beyond 1024")
	memLimit := flag.Int("memory-limit", 0, "soft heap memory limit, in MiB (0: none); the garbage collector works harder as it approaches")
	flag.Parse()
	log := slog.New(slog.NewTextHandler(os.Stderr, nil))
	if len(models) == 0 {
		fmt.Fprintln(os.Stderr, "at least one -model is required")
		os.Exit(2)
	}
	opts := []indecis.Option{indecis.WithThreads(*threads), indecis.WithEmbedCache(*cache)}
	if *int8 {
		opts = append(opts, indecis.WithInt8())
	}
	if *maxLen > 0 {
		opts = append(opts, indecis.WithMaxLen(*maxLen))
	}
	if *batching {
		opts = append(opts, indecis.WithBatching(0, 0))
	}
	if *memLimit > 0 {
		debug.SetMemoryLimit(int64(*memLimit) << 20)
	}
	s := &decision.Server{Models: map[string]*decision.Client{}, APIKey: *apiKey, Logger: log, MaxConcurrent: *maxConcurrent}
	for _, spec := range models {
		name, dir, ok := strings.Cut(spec, "=")
		if !ok {
			name, dir = filepath.Base(spec), spec
		}
		m, err := indecis.Open(dir, opts...)
		if err != nil {
			log.Error("loading", "model", dir, "error", err)
			os.Exit(1)
		}
		// An empty call prepares the weights for inference (in int8, the
		// float32 matrices are then freed), then the temporary memory used
		// for loading is returned to the system before the next model: the
		// startup peak no longer depends on the number of models.
		if _, err := m.Embed(context.Background(), "warm-up"); err != nil {
			log.Error("warm-up", "model", dir, "error", err)
			os.Exit(1)
		}
		debug.FreeOSMemory()
		s.Models[name] = decision.FromModel(name, m)
		if s.Default == "" {
			s.Default = name
		}
		log.Info("model loaded", "name", name, "dir", dir, "paired", m.Paired(), "memory", memory())
	}
	log.Info("listening", "addr", *addr, "endpoints", "POST /api/alpha/decisions, POST /v1/systemone, GET /api/alpha/models")
	if err := http.ListenAndServe(*addr, s.Handler()); err != nil {
		log.Error("server", "error", err)
		os.Exit(1)
	}
}

// memory summarizes the process memory (Linux), for the logs.
func memory() string {
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return "?"
	}
	var parts []string
	for _, l := range strings.Split(string(b), "\n") {
		for _, k := range []string{"RssAnon:", "RssFile:", "VmHWM:"} {
			if strings.HasPrefix(l, k) {
				parts = append(parts, strings.Join(strings.Fields(l), " "))
			}
		}
	}
	return strings.Join(parts, ", ")
}
