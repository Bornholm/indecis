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
	model := fs.String("model", "", "modèle")
	data := fs.String("data", "", "exemples étiquetés (JSONL, motifs séparés par des virgules)")
	schemaPath := fs.String("schema", "", "questions posées en mode ouvert (sinon : les têtes du modèle)")
	int8 := fs.Bool("int8", true, "calcul en int8 si le processeur le permet")
	by := fs.String("by", "", "mesurer aussi par valeur de ce champ meta (source, lang…)")
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
	fmt.Printf("== tous (%d)\n", len(ex))
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
