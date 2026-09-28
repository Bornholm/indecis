package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"io"
	"os"
	"strings"

	"github.com/bornholm/indecis"
)

func runPredict(args []string) error {
	fs := flag.NewFlagSet("predict", flag.ExitOnError)
	model := fs.String("model", "", "modèle")
	in := fs.String("in", "-", "entrée : une ligne par texte, en texte brut ou en JSON {\"text\", \"context\"} (- : entrée standard)")
	schemaPath := fs.String("schema", "", "questions posées en mode ouvert (sinon : les têtes du modèle)")
	int8 := fs.Bool("int8", true, "calcul en int8 si le processeur le permet")
	fs.Parse(args)
	if err := required(fs, "model"); err != nil {
		return err
	}
	opts := []indecis.Option{indecis.WithEmbedCache(1024)}
	if *int8 {
		opts = append(opts, indecis.WithInt8())
	}
	m, err := indecis.Open(*model, opts...)
	if err != nil {
		return err
	}
	var questions []indecis.OpenQuestion
	if *schemaPath != "" {
		sf, err := readSchema(*schemaPath)
		if err != nil {
			return err
		}
		questions = sf.open()
	}
	var r io.Reader = os.Stdin
	if *in != "-" {
		f, err := os.Open(*in)
		if err != nil {
			return err
		}
		defer f.Close()
		r = f
	}
	ctx := context.Background()
	w := bufio.NewWriter(os.Stdout)
	defer w.Flush()
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)

	var inputs []indecis.Input
	flush := func() error {
		if len(inputs) == 0 {
			return nil
		}
		var ds []indecis.Decision
		var err error
		if questions != nil {
			texts := make([]string, len(inputs))
			for i, in := range inputs {
				texts[i] = in.Text
				if in.Context != "" {
					texts[i] = in.Context + "\n\n" + in.Text
				}
			}
			ds, err = m.DecideOpen(ctx, questions, texts...)
		} else {
			ds, err = m.DecideInputs(ctx, inputs...)
		}
		if err != nil {
			return err
		}
		for i, d := range ds {
			if err := enc.Encode(map[string]any{"text": inputs[i].Text, "context": inputs[i].Context, "answers": d}); err != nil {
				return err
			}
		}
		inputs = inputs[:0]
		return nil
	}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var in indecis.Input
		var obj struct {
			Text    string `json:"text"`
			Context string `json:"context"`
		}
		if strings.HasPrefix(line, "{") && json.Unmarshal([]byte(line), &obj) == nil && obj.Text != "" {
			in = indecis.Input{Text: obj.Text, Context: obj.Context}
		} else {
			in = indecis.Input{Text: line}
		}
		inputs = append(inputs, in)
		if len(inputs) == 64 {
			if err := flush(); err != nil {
				return err
			}
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	return flush()
}
