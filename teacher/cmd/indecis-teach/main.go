// Command indecis-teach puts LLM teachers to work on a JSONL dataset.
//
//	indecis-teach rewrite -in ex.jsonl -out variants.jsonl -instructions instr.txt -variants 2
//	indecis-teach rewrite -teachers teachers.yaml -teacher pi-minimax -variants 1 … (in batches)
//	indecis-teach label   -in ex.jsonl -out labeled.jsonl -schema schema.json [-verify]
//	indecis-teach label   -teachers teachers.yaml -guidelines policy.md … (consensus)
//
// The genai client is configured by the GENAI_* variables (see
// github.com/bornholm/genai), read from -env. Answers are cached in -cache;
// -max-calls bounds the real calls.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/bornholm/genai/llm/provider"
	_ "github.com/bornholm/genai/llm/provider/all"
	"github.com/bornholm/genai/llm/provider/env"
	"github.com/bornholm/genai/llm/ratelimit"
	"github.com/bornholm/genai/llm/retry"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
	"github.com/bornholm/indecis/teacher"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: indecis-teach rewrite|label [options]")
		os.Exit(2)
	}
	cmd := os.Args[1]
	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	envFile := fs.String("env", ".env", "GENAI_* configuration file")
	in := fs.String("in", "", "input dataset (JSONL)")
	out := fs.String("out", "", "output dataset (JSONL)")
	cachePath := fs.String("cache", "teacher-cache.jsonl", "response cache")
	maxCalls := fs.Int("max-calls", 0, "real calls at most (0: unlimited)")
	concurrency := fs.Int("concurrency", 2, "simultaneous calls")
	interval := fs.Duration("interval", 2*time.Second, "minimal interval between two calls (respects the provider's quota)")
	retries := fs.Int("retries", 5, "retries on a temporary error (429, 5xx)")
	retryDelay := fs.Duration("retry-delay", 30*time.Second, "first wait before a retry, doubled afterwards")
	limit := fs.Int("limit", 0, "only process the first n examples")
	instrFile := fs.String("instructions", "", "rewrite: one instruction per line, assigned round-robin")
	variants := fs.Int("variants", 2, "rewrite: variants per example")
	schemaFile := fs.String("schema", "", "label: JSON schema (list of questions) or a model's indecis.json")
	teachersFile := fs.String("teachers", "", "label: YAML file of command-line teachers; enables consensus. rewrite: with -teacher, rewrites in batches with that teacher")
	teacherID := fs.String("teacher", "", "rewrite -teachers: id of the teacher to use")
	disagreements := fs.String("disagreements", "", "label -teachers: JSONL file of disagreements to review")
	guidelines := fs.String("guidelines", "", "label: labeling policy (Markdown) passed to the teachers")
	verify := fs.Bool("verify", false, "label: relabel and discard examples where the teacher contradicts the existing noul label")
	keepUnverified := fs.Bool("keep-unverified", false, "label -verify: keep, marked verified=false, examples the teacher did not judge")
	cacheOnly := fs.Bool("cache-only", false, "replay the cache without any call to the LLM")
	fs.Parse(os.Args[2:])
	if *in == "" || *out == "" {
		log.Fatal("-in and -out are required")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	var t *teacher.Teacher
	switch {
	case *teachersFile != "" && cmd == "rewrite":
		// A single teacher is enough to rewrite: the one named by -teacher.
		cache, err := teacher.OpenCache(*cachePath)
		if err != nil {
			log.Fatal(err)
		}
		if t, err = harnessTeacher(*teachersFile, *teacherID, cache, *maxCalls, *cacheOnly); err != nil {
			log.Fatal(err)
		}
	case *teachersFile != "":
		if cmd != "label" {
			log.Fatal("-teachers only applies to label and rewrite")
		}
		if err := consensusLabel(ctx, *teachersFile, *in, *out, *disagreements, *schemaFile, *guidelines, *cachePath, *maxCalls, *limit, *cacheOnly); err != nil {
			log.Fatal(err)
		}
		return
	default:
		base, err := provider.Create(ctx, env.With("GENAI_", *envFile))
		if err != nil {
			log.Fatal(err)
		}
		// Bounded rate, then spaced-out retries on 429: a quota limit is
		// worked around by waiting, not by insisting.
		client := retry.NewClient(ratelimit.NewClient(base, ratelimit.WithChatLimit(*interval, 1)), *retryDelay, *retries)
		cache, err := teacher.OpenCache(*cachePath)
		if err != nil {
			log.Fatal(err)
		}
		t = &teacher.Teacher{Client: client, Model: modelName(*envFile), Cache: cache, MaxCalls: *maxCalls, Concurrency: *concurrency, CacheOnly: *cacheOnly}
	}

	examples, err := dataset.ReadFile(*in)
	if err != nil {
		log.Fatal(err)
	}
	if *limit > 0 && *limit < len(examples) {
		examples = examples[:*limit]
	}

	var result []dataset.Example
	var stats teacher.Stats
	switch cmd {
	case "rewrite":
		result, stats, err = rewrite(ctx, t, examples, *instrFile, *variants)
	case "label":
		result, stats, err = label(ctx, t, examples, *schemaFile, *verify, *keepUnverified)
	default:
		log.Fatalf("unknown command %q", cmd)
	}
	// The partial result is always written: calls already paid for must
	// not be lost to a rate limit or an interruption.
	if werr := dataset.WriteFile(*out, result); werr != nil {
		log.Fatal(werr)
	}
	log.Printf("%s: %d submitted, %d produced, %d from cache, %d refused or invalid responses, %d skipped (not in cache), %d real calls",
		cmd, stats.Requested, stats.Done, stats.Cached, stats.Refused, stats.Skipped, t.Calls())
	if err != nil {
		if errors.Is(err, teacher.ErrBudget) {
			log.Printf("budget exhausted: partial result written to %s", *out)
			return
		}
		log.Printf("interrupted: %v", err)
		log.Printf("partial result written to %s; rerunning resumes from the cache", *out)
		os.Exit(1)
	}
}

// rewrite assigns instructions round-robin: each example receives a
// single instruction, the cost stays one call per example.
func rewrite(ctx context.Context, t *teacher.Teacher, examples []dataset.Example, instrFile string, variants int) ([]dataset.Example, teacher.Stats, error) {
	b, err := os.ReadFile(instrFile)
	if err != nil {
		return nil, teacher.Stats{}, err
	}
	var instructions []string
	for _, l := range strings.Split(string(b), "\n") {
		if l = strings.TrimSpace(l); l != "" && !strings.HasPrefix(l, "#") {
			instructions = append(instructions, l)
		}
	}
	if len(instructions) == 0 {
		return nil, teacher.Stats{}, fmt.Errorf("aucune instruction dans %s", instrFile)
	}
	groups := make([][]dataset.Example, len(instructions))
	for i, e := range examples {
		groups[i%len(instructions)] = append(groups[i%len(instructions)], e)
	}
	var all []dataset.Example
	var total teacher.Stats
	for i, instr := range instructions {
		out, st, err := t.Rewrite(ctx, groups[i], instr, variants)
		all = append(all, out...)
		total.Requested += st.Requested
		total.Done += st.Done
		total.Cached += st.Cached
		total.Refused += st.Refused
		log.Printf("%q: %d -> %d variants (%d refused)", instr, st.Requested, st.Done, st.Refused)
		if err != nil {
			return all, total, err
		}
	}
	return all, total, nil
}

func label(ctx context.Context, t *teacher.Teacher, examples []dataset.Example, schemaFile string, verify, keepUnverified bool) ([]dataset.Example, teacher.Stats, error) {
	schema, err := readSchema(schemaFile)
	if err != nil {
		return nil, teacher.Stats{}, err
	}
	if !verify {
		return t.Label(ctx, schema, examples)
	}
	// Verification: the teacher labels without seeing the existing labels;
	// the example is kept, with its original labels filled in, only if it
	// agrees on every noul question.
	stripped := make([]dataset.Example, len(examples))
	for i, e := range examples {
		stripped[i] = e
		stripped[i].Labels = nil
		stripped[i].Meta = withIndex(e.Meta, i)
	}
	labeled, stats, err := t.Label(ctx, schema, stripped)
	var kept []dataset.Example
	disagree := 0
	judged := map[int]bool{}
	for _, l := range labeled {
		judged[indexOf(l.Meta)] = true
		orig := examples[indexOf(l.Meta)]
		ok := true
		for _, q := range schema {
			want, isBool := orig.Labels[q.Name].(bool)
			got, isNum := l.Labels[q.Name].(float64)
			if q.Kind == indecis.Noul && isBool && isNum && (got >= 0.5) != want {
				ok = false
			}
		}
		if !ok {
			disagree++
			continue
		}
		e := orig
		e.Labels = map[string]any{}
		for k, v := range l.Labels {
			e.Labels[k] = v
		}
		for k, v := range orig.Labels {
			e.Labels[k] = v
		}
		e.Meta = withMeta(l.Meta, "verified", "true")
		delete(e.Meta, "index")
		kept = append(kept, e)
	}
	unverified := 0
	if keepUnverified {
		for i, e := range examples {
			if !judged[i] {
				e.Meta = withMeta(e.Meta, "verified", "false")
				kept = append(kept, e)
				unverified++
			}
		}
	}
	log.Printf("verification: %d kept, of which %d unverified, %d discarded (teacher disagreement)", len(kept), unverified, disagree)
	stats.Done = len(kept)
	return kept, stats, err
}

func withMeta(meta map[string]string, k, v string) map[string]string {
	out := map[string]string{}
	for mk, mv := range meta {
		out[mk] = mv
	}
	out[k] = v
	return out
}

func withIndex(meta map[string]string, i int) map[string]string {
	out := map[string]string{}
	for k, v := range meta {
		out[k] = v
	}
	out["index"] = fmt.Sprint(i)
	return out
}

func indexOf(meta map[string]string) int {
	var i int
	fmt.Sscan(meta["index"], &i)
	return i
}

// readSchema accepts a list of questions or a model's indecis.json.
func readSchema(path string) (indecis.Schema, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var s indecis.Schema
	if err := json.Unmarshal(b, &s); err == nil && len(s) > 0 {
		return s, s.Validate()
	}
	var meta struct {
		Schema indecis.Schema `json:"schema"`
	}
	if err := json.Unmarshal(b, &meta); err != nil {
		return nil, err
	}
	return meta.Schema, meta.Schema.Validate()
}

// modelName reads the model name from the environment file, for the
// cache key; it is not secret, unlike the API key.
func modelName(envFile string) string {
	b, err := os.ReadFile(envFile)
	if err != nil {
		return "unknown"
	}
	for _, l := range strings.Split(string(b), "\n") {
		k, v, ok := strings.Cut(strings.TrimSpace(l), "=")
		if ok && strings.HasPrefix(k, "GENAI_CHAT_COMPLETION_") && strings.HasSuffix(k, "_MODEL") {
			return strings.Trim(v, `"'`)
		}
	}
	return "unknown"
}
