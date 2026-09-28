package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

func runTrain(args []string) error {
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	backbone := fs.String("backbone", "", "backbone au format transformers (config.json, model.safetensors, tokenizer.json)")
	schemaPath := fs.String("schema", "", "schéma : liste de questions JSON (voir le guide)")
	trainGlobs := fs.String("train", "", "exemples d'entraînement (JSONL, motifs séparés par des virgules)")
	calibGlobs := fs.String("calib", "", "exemples de calibration (défaut : 5 % de l'entraînement)")
	testGlobs := fs.String("test", "", "exemples de test (défaut : 10 % de l'entraînement, par famille)")
	holdout := fs.Float64("holdout", 0.1, "part tenue à l'écart pour le test sans -test")
	holdoutBy := fs.String("holdout-by", "example", "mise de côté par exemple, ou par family (familles entières : mesure la généralisation à des familles jamais vues)")
	out := fs.String("out", "", "répertoire du modèle entraîné")
	open := fs.Bool("open", false, "entraîner les plongements pour des options données à l'inférence (ChooseNearest, DecideOpen) plutôt que des têtes fixes")
	pairs := fs.Bool("pairs", false, "lire des paires (contexte, texte)")
	maxLen := fs.Int("max-len", 256, "tokens lus au plus par exemple")
	def := indecis.DefaultTrainOptions()
	epochs := fs.Int("epochs", def.Epochs, "époques")
	batch := fs.Int("batch", def.BatchSize, "exemples par lot")
	lr := fs.Float64("lr", def.LR, "taux d'apprentissage de l'encodeur (défaut en mode -open : 2e-5)")
	headLR := fs.Float64("head-lr", def.HeadLR, "taux d'apprentissage des têtes")
	seed := fs.Int64("seed", def.Seed, "graine")
	threads := fs.Int("threads", 0, "cœurs (0 : tous)")
	int8Emb := fs.Bool("int8-embeddings", false, "écrire la table d'embeddings en int8 (deux fois plus petite)")
	fs.Parse(args)
	if err := required(fs, "backbone", "schema", "train", "out"); err != nil {
		return err
	}
	ctx := context.Background()
	sf, err := readSchema(*schemaPath)
	if err != nil {
		return err
	}
	all, err := readExamples(*trainGlobs)
	if err != nil {
		return err
	}
	trainEx, test := all, []dataset.Example(nil)
	if *testGlobs != "" {
		if test, err = readExamples(*testGlobs); err != nil {
			return err
		}
	} else if *holdout > 0 {
		trainEx, test = holdOut(all, *holdout, uint64(*seed), *holdoutBy)
	}
	var calib []dataset.Example
	if *calibGlobs != "" {
		if calib, err = readExamples(*calibGlobs); err != nil {
			return err
		}
	} else if !*open {
		trainEx, calib = holdOut(trainEx, 0.05, uint64(*seed)+1, *holdoutBy)
	}
	log.Printf("%d exemples d'entraînement, %d de calibration, %d de test", len(trainEx), len(calib), len(test))

	opts := []indecis.Option{indecis.WithMaxLen(*maxLen), indecis.WithThreads(*threads)}
	if *pairs {
		opts = append(opts, indecis.WithPairs())
	}
	m, err := indecis.New(*backbone, sf.schema, *seed, opts...)
	if err != nil {
		return err
	}
	to := indecis.DefaultTrainOptions()
	to.Epochs, to.BatchSize, to.HeadLR, to.Seed = *epochs, *batch, *headLR, *seed
	to.LR = *lr
	lrSet := false
	fs.Visit(func(f *flag.Flag) { lrSet = lrSet || f.Name == "lr" })
	if *open && !lrSet {
		to.LR = 2e-5 // les plongements du backbone sont déjà bons : on les ajuste
	}
	start := time.Now()
	to.Progress = func(p indecis.Progress) {
		if p.Step%50 == 0 || p.Step == p.Steps {
			log.Printf("pas %d/%d  perte %.4f  %.0f tokens/s  %s", p.Step, p.Steps, p.Loss,
				float64(p.Tokens)/p.Elapsed.Seconds(), p.Elapsed.Round(time.Second))
		}
	}
	if *open {
		batches := indecis.ChoiceBatches(trainEx, sf.open(), *batch, *seed)
		if len(batches) == 0 {
			return fmt.Errorf("aucun exemple étiqueté pour les questions du schéma")
		}
		log.Printf("mode ouvert : %d lots", len(batches))
		if err := m.FitEmbeddings(ctx, batches, to); err != nil {
			return err
		}
	} else {
		if err := m.Fit(ctx, trainEx, to); err != nil {
			return err
		}
		if len(calib) > 0 {
			temps, err := m.Calibrate(ctx, calib)
			if err != nil {
				return err
			}
			log.Printf("températures : %v", temps)
		}
	}
	log.Printf("entraînement : %s", time.Since(start).Round(time.Second))
	if *int8Emb {
		indecis.WithInt8Embeddings()(m)
	}
	if err := m.Save(*out); err != nil {
		return err
	}
	log.Printf("modèle écrit dans %s", *out)
	if len(test) > 0 {
		return report(ctx, m, sf, test, *open)
	}
	return nil
}

// report affiche les mesures d'un modèle sur des exemples étiquetés : par
// ses têtes, ou en mode ouvert par les options du schéma.
func report(ctx context.Context, m *indecis.Model, sf *schemaFile, ex []dataset.Example, open bool) error {
	if !open {
		metrics, err := m.Evaluate(ctx, ex)
		if err != nil {
			return err
		}
		for _, mt := range metrics {
			fmt.Println(mt)
		}
		return nil
	}
	for _, q := range sf.open() {
		var texts []string
		var gold []int
		for _, e := range ex {
			if v, ok := e.Labels[q.Name]; ok {
				if j := q.OptionIndex(v); j >= 0 {
					texts = append(texts, e.Text)
					gold = append(gold, j)
				}
			}
		}
		if len(texts) == 0 {
			continue
		}
		d, err := m.DecideOpen(ctx, []indecis.OpenQuestion{q}, texts...)
		if err != nil {
			return err
		}
		correct := 0
		for i, dec := range d {
			a := dec[q.Name]
			var got string
			switch q.Kind {
			case indecis.Noul:
				got = map[bool]string{true: q.Options[0].Name, false: q.Options[1].Name}[a.P >= 0.5]
			default:
				got = a.Choice
			}
			if got == q.Options[gold[i]].Name {
				correct++
			}
		}
		fmt.Printf("%-14s n=%-5d options=%-3d exactitude=%.1f %%\n", q.Name, len(texts), len(q.Options), 100*float64(correct)/float64(len(texts)))
	}
	return nil
}

// holdOut met de côté une part des exemples, par famille ou exemple par
// exemple (Family vidée : HoldOut découpe alors sur le texte).
func holdOut(ex []dataset.Example, fraction float64, seed uint64, by string) (kept, held []dataset.Example) {
	if by != "family" {
		keyed := make([]dataset.Example, len(ex))
		for i, e := range ex {
			e.Family = ""
			keyed[i] = e
		}
		ex = keyed
	}
	return dataset.HoldOut(ex, fraction, seed)
}
