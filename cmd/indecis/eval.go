package main

import (
	"context"
	"flag"
	"fmt"
	"sort"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

func runEval(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	model := fs.String("model", "", "model")
	data := fs.String("data", "", "labeled examples (JSONL, comma-separated patterns)")
	schemaPath := fs.String("schema", "", "questions asked in open mode (otherwise: the model's heads)")
	int8 := fs.Bool("int8", true, "int8 computation if the processor allows it")
	by := fs.String("by", "", "also measure by value of this meta field (source, lang, ...)")
	fs.Parse(args)
	if err := required(fs, "model", "data"); err != nil {
		return err
	}
	opts := []indecis.Option{}
	if *int8 {
		opts = append(opts, indecis.WithInt8())
	}
	m, err := indecis.Open(*model, opts...)
	if err != nil {
		return err
	}
	ex, err := readExamples(*data)
	if err != nil {
		return err
	}
	var sf *schemaFile
	if *schemaPath != "" {
		if sf, err = readSchema(*schemaPath); err != nil {
			return err
		}
	}
	ctx := context.Background()
	if *by == "" {
		return report(ctx, m, sf, ex, sf != nil)
	}
	groups := map[string][]dataset.Example{}
	var keys []string
	for _, e := range ex {
		k := e.Meta[*by]
		if _, ok := groups[k]; !ok {
			keys = append(keys, k)
		}
		groups[k] = append(groups[k], e)
	}
	sort.Strings(keys)
	fmt.Printf("== all (%d)\n", len(ex))
	if err := report(ctx, m, sf, ex, sf != nil); err != nil {
		return err
	}
	for _, k := range keys {
		fmt.Printf("== %s=%q (%d)\n", *by, k, len(groups[k]))
		if err := report(ctx, m, sf, groups[k], sf != nil); err != nil {
			return err
		}
	}
	return nil
}
