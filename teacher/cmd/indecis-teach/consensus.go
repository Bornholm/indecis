package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"strconv"
	"sync"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/bornholm/indecis/dataset"
	"github.com/bornholm/indecis/teacher"
)

type teachersConfig struct {
	Teachers []struct {
		ID          string        `yaml:"id"`
		Model       string        `yaml:"model"`
		Command     []string      `yaml:"command"`
		SystemFlag  string        `yaml:"system_flag"`
		Input       string        `yaml:"input"`
		Output      string        `yaml:"output"`
		Batch       int           `yaml:"batch"`
		Concurrency int           `yaml:"concurrency"`
		Interval    time.Duration `yaml:"interval"`
		Timeout     time.Duration `yaml:"timeout"`
		MaxChars    int           `yaml:"max_chars"`
	} `yaml:"teachers"`
}

// consensusLabel has each example labeled by every teacher, in parallel,
// then keeps only what they agree on.
func consensusLabel(ctx context.Context, cfgPath, in, out, disagreementsPath, schemaFile, guidelinesPath, cachePath string, maxCalls, limit int, cacheOnly bool) error {
	var guidelines string
	if guidelinesPath != "" {
		b, err := os.ReadFile(guidelinesPath)
		if err != nil {
			return err
		}
		guidelines = string(b)
	}
	b, err := os.ReadFile(cfgPath)
	if err != nil {
		return err
	}
	var cfg teachersConfig
	if err := yaml.Unmarshal(b, &cfg); err != nil {
		return fmt.Errorf("%s: %w", cfgPath, err)
	}
	if len(cfg.Teachers) < 2 {
		return fmt.Errorf("%s: at least two teachers for a consensus", cfgPath)
	}
	schema, err := readSchema(schemaFile)
	if err != nil {
		return err
	}
	examples, err := dataset.ReadFile(in)
	if err != nil {
		return err
	}
	if limit > 0 && limit < len(examples) {
		examples = examples[:limit]
	}
	cache, err := teacher.OpenCache(cachePath)
	if err != nil {
		return err
	}

	// The teachers do not see the existing labels; the index is used to
	// match their responses back up.
	stripped := make([]dataset.Example, len(examples))
	for i, e := range examples {
		stripped[i] = dataset.Example{Context: e.Context, Text: e.Text, Meta: map[string]string{"index": strconv.Itoa(i)}}
	}

	names := make([]string, len(cfg.Teachers))
	byTeacher := make([][]map[string]any, len(cfg.Teachers))
	errs := make([]error, len(cfg.Teachers))
	var wg sync.WaitGroup
	for ti, tc := range cfg.Teachers {
		names[ti] = tc.ID
		byTeacher[ti] = make([]map[string]any, len(examples))
		t := &teacher.Teacher{
			Client: &teacher.Command{
				Args: tc.Command, SystemFlag: tc.SystemFlag, Input: tc.Input, Output: tc.Output, Timeout: tc.Timeout,
			},
			Model:       tc.ID + "/" + tc.Model,
			Cache:       cache,
			MaxCalls:    maxCalls,
			Concurrency: max(tc.Concurrency, 1),
			Interval:    tc.Interval,
			BatchSize:   max(tc.Batch, 1),
			CacheOnly:   cacheOnly,
			MaxChars:    tc.MaxChars,
			Guidelines:  guidelines,
		}
		wg.Add(1)
		go func(ti int, id string, t *teacher.Teacher) {
			defer wg.Done()
			start := time.Now()
			labeled, st, err := t.Label(ctx, schema, stripped)
			for _, e := range labeled {
				i, _ := strconv.Atoi(e.Meta["index"])
				byTeacher[ti][i] = e.Labels
			}
			log.Printf("%s: %d submitted, %d labeled, %d from cache, %d refused, %d failed, %d skipped, %d real calls, %s",
				id, st.Requested, st.Done, st.Cached, st.Refused, st.Failed, st.Skipped, t.Calls(), time.Since(start).Round(time.Second))
			if err != nil && !errors.Is(err, teacher.ErrBudget) {
				errs[ti] = fmt.Errorf("%s: %w", id, err)
			}
		}(ti, tc.ID, t)
	}
	wg.Wait()

	kept, disputes := teacher.Consensus(schema, examples, names, byTeacher)
	if err := dataset.WriteFile(out, kept); err != nil {
		return err
	}
	if disagreementsPath != "" {
		f, err := os.Create(disagreementsPath)
		if err != nil {
			return err
		}
		enc := json.NewEncoder(f)
		enc.SetEscapeHTML(false)
		for _, d := range disputes {
			enc.Encode(d)
		}
		f.Close()
	}
	log.Printf("consensus: %d examples kept, %d disagreements to review%s", len(kept), len(disputes), map[bool]string{true: " -> " + disagreementsPath, false: ""}[disagreementsPath != ""])
	return errors.Join(errs...)
}

// harnessTeacher builds the teacher named id from the configuration file.
func harnessTeacher(cfgPath, id string, cache *teacher.Cache, maxCalls int, cacheOnly bool) (*teacher.Teacher, error) {
	b, err := os.ReadFile(cfgPath)
	if err != nil {
		return nil, err
	}
	var cfg teachersConfig
	if err := yaml.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("%s: %w", cfgPath, err)
	}
	for _, tc := range cfg.Teachers {
		if tc.ID != id {
			continue
		}
		return &teacher.Teacher{
			Client: &teacher.Command{
				Args: tc.Command, SystemFlag: tc.SystemFlag, Input: tc.Input, Output: tc.Output, Timeout: tc.Timeout,
			},
			Model:       tc.ID + "/" + tc.Model,
			Cache:       cache,
			MaxCalls:    maxCalls,
			Concurrency: max(tc.Concurrency, 1),
			Interval:    tc.Interval,
			BatchSize:   max(tc.Batch, 1),
			CacheOnly:   cacheOnly,
			MaxChars:    tc.MaxChars,
		}, nil
	}
	return nil, fmt.Errorf("%s: no teacher %q", cfgPath, id)
}
