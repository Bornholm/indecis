package main

import (
	"context"
	"flag"

	"github.com/bornholm/indecis"
)

func runEval(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	model := fs.String("model", "", "modèle")
	data := fs.String("data", "", "exemples étiquetés (JSONL, motifs séparés par des virgules)")
	schemaPath := fs.String("schema", "", "questions posées en mode ouvert (sinon : les têtes du modèle)")
	int8 := fs.Bool("int8", true, "calcul en int8 si le processeur le permet")
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
	return report(context.Background(), m, sf, ex, sf != nil)
}
