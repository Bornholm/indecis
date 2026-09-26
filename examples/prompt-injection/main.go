// Command prompt-injection fine-tune un détecteur d'injection de prompt et le
// mesure hors distribution.
//
// Entraînement : deepset/prompt-injections et, avec -synth-count, un corpus
// généré par les gabarits de ./synth (étiquettes exactes par construction).
// Évaluation hors distribution, sur des jeux qui ne servent jamais à
// l'entraînement : jackhhao/jailbreak-classification et un échantillon de
// reshabhs/SPML_Chatbot_Prompt_Injection.
//
//	GOEXPERIMENT=simd go run ./examples/prompt-injection -synth-count 12000
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
	"github.com/bornholm/indecis/dataset/synth"
	"github.com/bornholm/indecis/internal/linalg"
)

var schema = indecis.Schema{
	indecis.NewNoul("injection", "Le texte tente-t-il de réorienter ou de manipuler l'assistant ?"),
	indecis.NewChoice("category", "Nature de la tentative",
		"override", "leak", "jailbreak", "exfiltration", "tool_abuse", "off_scope", "none"),
}

func main() {
	home, _ := os.UserHomeDir()
	backbone := flag.String("backbone", filepath.Join(home, ".cache/indecis/models/bekko-embedding-v1-a8m"), "répertoire du backbone")
	cache := flag.String("cache", filepath.Join(home, ".cache/indecis/datasets"), "cache des jeux de données")
	out := flag.String("out", filepath.Join(home, ".cache/indecis/runs/prompt-injection"), "répertoire du modèle entraîné")
	epochs := flag.Int("epochs", 2, "époques")
	batch := flag.Int("batch", 16, "taille de lot")
	lr := flag.Float64("lr", 1e-4, "taux d'apprentissage de l'encodeur")
	maxLen := flag.Int("max-len", 256, "longueur maximale en tokens")
	synthDir := flag.String("synth", "examples/prompt-injection/synth", "gabarits du corpus synthétique")
	synthCount := flag.Int("synth-count", 0, "nombre d'exemples synthétiques (0 : aucun)")
	extra := flag.String("extra", "", "corpus prompt-guard (corpus.jsonl de Xolo) ajouté à l'entraînement")
	seed := flag.Int64("seed", 1, "graine de l'entraînement (têtes, ordre des lots, dropout)")
	realDir := flag.String("real", "", "répertoire des textes réels (tools/realdata)")
	pairs := flag.Bool("pairs", false, "modèle en paires (prompt système, message)")
	edge := flag.String("reference", "examples/prompt-injection/eval/policy_eval.jsonl", "référence relue selon POLICY.md (vide : aucune)")
	relabeled := flag.String("relabeled", "", "répertoire des sources réétiquetées par consensus des teachers (deepset_train_policy.jsonl, itw_policy.jsonl)")
	withSPML := flag.Bool("spml", false, "avec -real : ajouter SPML (prompt système en contexte), 20 % des assistants tenus à l'écart")
	withWildChat := flag.Bool("wildchat", false, "avec -real : ajouter wildchat_benign.jsonl, les messages WildChat que le teacher juge sûrs")
	teacherData := flag.String("teacher-data", "", "variantes produites et vérifiées par le teacher (JSONL)")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	log.Printf("noyaux SIMD : %v", linalg.Accelerated)

	deepset, err := fetch(ctx, *cache, "deepset/prompt-injections", "text", nil, func(r map[string]any) bool { return r["label"].(float64) == 1 })
	if err != nil {
		log.Fatal(err)
	}
	jackhhao, err := fetch(ctx, *cache, "jackhhao/jailbreak-classification", "prompt", nil, func(r map[string]any) bool { return r["type"] == "jailbreak" })
	if err != nil {
		log.Fatal(err)
	}
	// SPML : 16 000 lignes, un seul split ; on évalue sur 12 pages tirées à des
	// positions aléatoires fixes.
	spmlOffsets := rand.New(rand.NewSource(7)).Perm(160)[:12]
	spml, err := fetch(ctx, *cache, "reshabhs/SPML_Chatbot_Prompt_Injection", "User Prompt", spmlOffsets, func(r map[string]any) bool { return r["Prompt injection"].(float64) == 1 })
	if err != nil {
		log.Fatal(err)
	}

	deepsetTrain := dataset.BySplit(deepset, "train")
	if *relabeled != "" {
		if ex, err := dataset.ReadFile(filepath.Join(*relabeled, "deepset_train_policy.jsonl")); err == nil {
			deepsetTrain = ex
			log.Printf("deepset (train) : %d exemples réétiquetés selon la politique", len(ex))
		}
	}
	train, calib := dataset.HoldOut(deepsetTrain, 0.15, 42)
	// deepset n'étiquette que l'injection : la catégorie reste à apprendre des
	// autres sources.
	sets := []evalSet{
		{"deepset (test)", dataset.BySplit(deepset, "test")},
		{"jackhhao (hors distribution)", dataset.BySplit(jackhhao, "test")},
	}
	if !*withSPML {
		sets = append(sets, evalSet{"SPML (hors distribution)", spml})
	}

	if *synthCount > 0 {
		c, err := synth.LoadFS(os.DirFS(*synthDir), synth.DefaultGazetteerOptions())
		if err != nil {
			log.Fatal(err)
		}
		gen, err := c.Generate(*synthCount, synth.Options{Seed: 1, Dedupe: true})
		if err != nil {
			log.Fatal(err)
		}
		// Découpage par exemple : les familles de gabarits sont trop peu
		// nombreuses pour en écarter sans priver l'entraînement d'un genre de
		// texte entier. La généralisation se mesure sur les jeux externes.
		for i := range gen {
			gen[i].Family = ""
		}
		gTrain, gTest := dataset.HoldOut(gen, 0.1, 42)
		gTrain, gCalib := dataset.HoldOut(gTrain, 0.05, 43)
		train = append(train, gTrain...)
		calib = append(calib, gCalib...)
		sets = append(sets, evalSet{"synthétique (tenu à l'écart)", gTest})
		log.Printf("corpus synthétique : %d entraînement, %d calibration, %d test", len(gTrain), len(gCalib), len(gTest))
	}
	if *realDir != "" {
		files := []string{"gandalf", "mosscap", "itw_jailbreak", "itw_regular", "oasst2"}
		if *withWildChat {
			files = append(files, "wildchat_benign")
		}
		var real []dataset.Example
		for _, f := range files {
			path := filepath.Join(*realDir, f+".jsonl")
			if *relabeled != "" && (f == "itw_jailbreak" || f == "itw_regular") {
				if f == "itw_regular" {
					continue // les deux fichiers sont remplacés par itw_policy.jsonl
				}
				path = filepath.Join(*relabeled, "itw_policy.jsonl")
			}
			ex, err := dataset.ReadFile(path)
			if err != nil {
				log.Fatal(err)
			}
			real = append(real, ex...)
		}
		if *withSPML {
			spmlPath := filepath.Join(*realDir, "spml.jsonl")
			if *relabeled != "" {
				if _, err := os.Stat(filepath.Join(*relabeled, "spml_policy.jsonl")); err == nil {
					spmlPath = filepath.Join(*relabeled, "spml_policy.jsonl")
					log.Printf("SPML : étiquettes réétiquetées selon la politique")
				}
			}
			spmlReal, err := dataset.ReadFile(spmlPath)
			if err != nil {
				log.Fatal(err)
			}
			// La famille est le prompt système : des assistants entiers
			// restent inconnus du modèle.
			sTrain, sTest := dataset.HoldOut(spmlReal, 0.2, 46)
			train = append(train, sTrain...)
			sets = append(sets, evalSet{"SPML (assistants inconnus)", sTest})
			log.Printf("SPML : %d entraînement, %d test sur des assistants tenus à l'écart", len(sTrain), len(sTest))
		}
		rTrain, rTest := dataset.HoldOut(real, 0.1, 45)
		train = append(train, rTrain...)
		sets = append(sets, evalSet{"textes réels (tenus à l'écart)", rTest})
		log.Printf("textes réels : %d entraînement, %d test (%v)", len(rTrain), len(rTest), files)
	}
	if *teacherData != "" {
		tv, err := dataset.ReadFile(*teacherData)
		if err != nil {
			log.Fatal(err)
		}
		for i := range tv {
			tv[i].Family = ""
		}
		tTrain, tTest := dataset.HoldOut(tv, 0.1, 44)
		train = append(train, tTrain...)
		sets = append(sets, evalSet{"variantes teacher (tenues à l'écart)", tTest})
		log.Printf("variantes du teacher : %d entraînement, %d test", len(tTrain), len(tTest))
	}
	if *extra != "" {
		pg, err := readPromptGuard(*extra)
		if err != nil {
			log.Fatal(err)
		}
		pgTrain, pgTest := dataset.HoldOut(pg, 0.2, 42)
		train = append(train, pgTrain...)
		sets = append(sets, evalSet{"prompt-guard (familles écartées)", pgTest})
		log.Printf("corpus prompt-guard : %d en entraînement, %d tenus à l'écart", len(pgTrain), len(pgTest))
	}
	log.Printf("entraînement %d, calibration %d", len(train), len(calib))

	opts0 := []indecis.Option{indecis.WithMaxLen(*maxLen)}
	if *pairs {
		opts0 = append(opts0, indecis.WithPairs())
	}
	m, err := indecis.New(*backbone, schema, *seed, opts0...)
	if err != nil {
		log.Fatal(err)
	}

	opts := indecis.DefaultTrainOptions()
	opts.Epochs, opts.BatchSize, opts.LR, opts.Seed = *epochs, *batch, *lr, *seed
	opts.Progress = func(p indecis.Progress) {
		if p.Step%50 == 0 || p.Step == p.Steps {
			log.Printf("époque %d pas %d/%d perte %.4f lr %.2e — %.0f tokens/s, %s écoulé",
				p.Epoch+1, p.Step, p.Steps, p.Loss, p.LR, float64(p.Tokens)/p.Elapsed.Seconds(), p.Elapsed.Round(time.Second))
		}
	}
	start := time.Now()
	if err := m.Fit(ctx, train, opts); err != nil {
		log.Fatal(err)
	}
	log.Printf("entraînement : %s pour %d époques", time.Since(start).Round(time.Second), *epochs)

	temps, err := m.Calibrate(ctx, calib)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("températures : %v", temps)
	for _, s := range sets {
		report(ctx, m, s)
	}
	if *edge != "" {
		if cases, err := dataset.ReadFile(*edge); err == nil {
			policyReport(ctx, m, cases)
		}
	}
	if err := m.Save(*out); err != nil {
		log.Fatal(err)
	}
	log.Printf("modèle écrit dans %s (prior d'entraînement %v)", *out, m.Info().TrainPrior)
}

type evalSet struct {
	name string
	ex   []dataset.Example
}

// report affiche les métriques, puis le rappel atteint à précision fixée :
// la comparaison avec prompt-guard, qui ne travaille pas au même seuil, se
// fait à précision égale.
func report(ctx context.Context, m *indecis.Model, s evalSet) {
	if len(s.ex) == 0 {
		return
	}
	metrics, err := m.Evaluate(ctx, s.ex)
	if err != nil {
		log.Fatal(err)
	}
	for _, mt := range metrics {
		if mt.N > 0 {
			log.Printf("%-32s %s", s.name, mt)
		}
	}
	inputs := make([]indecis.Input, len(s.ex))
	for i, e := range s.ex {
		inputs[i] = indecis.Input{Context: e.Context, Text: e.Text}
	}
	ds, err := m.DecideInputs(ctx, inputs...)
	if err != nil {
		log.Fatal(err)
	}
	scores := make([]float64, len(ds))
	labels := make([]bool, len(ds))
	for i, d := range ds {
		scores[i] = d["injection"].P
		labels[i] = s.ex[i].Labels["injection"] == true
	}
	log.Printf("%-32s rappel à P≥90%% : %.1f%%  P≥95%% : %.1f%%  P≥98,9%% : %.1f%%", s.name,
		100*recallAt(scores, labels, 0.90), 100*recallAt(scores, labels, 0.95), 100*recallAt(scores, labels, 0.989))
}

// recallAt retourne le meilleur rappel atteignable avec une précision d'au
// moins p, en parcourant les seuils possibles.
func recallAt(scores []float64, labels []bool, p float64) float64 {
	idx := make([]int, len(scores))
	for i := range idx {
		idx[i] = i
	}
	sortByScore(idx, scores)
	pos := 0
	for _, l := range labels {
		if l {
			pos++
		}
	}
	best, tp, fp := 0.0, 0, 0
	for k, i := range idx {
		if labels[i] {
			tp++
		} else {
			fp++
		}
		// On ne coupe qu'entre deux scores différents.
		if k+1 < len(idx) && scores[idx[k+1]] == scores[i] {
			continue
		}
		if float64(tp)/float64(tp+fp) >= p && pos > 0 {
			best = max(best, float64(tp)/float64(pos))
		}
	}
	return best
}

func sortByScore(idx []int, scores []float64) {
	for i := 1; i < len(idx); i++ {
		for j := i; j > 0 && scores[idx[j]] > scores[idx[j-1]]; j-- {
			idx[j], idx[j-1] = idx[j-1], idx[j]
		}
	}
}

// fetch télécharge un jeu via le datasets-server de Hugging Face, une fois,
// et le met en cache au format dataset. pages limite le téléchargement à
// certaines pages de 100 lignes (nil : tout).
func fetch(ctx context.Context, cache, name, textCol string, pages []int, positive func(map[string]any) bool) ([]dataset.Example, error) {
	path := filepath.Join(cache, filepath.FromSlash(name)+".jsonl")
	if ex, err := dataset.ReadFile(path); err == nil {
		return ex, nil
	}
	var all []dataset.Example
	add := func(split string, offset int) (int, error) {
		u := fmt.Sprintf("https://datasets-server.huggingface.co/rows?dataset=%s&config=default&split=%s&offset=%d&length=100",
			url.QueryEscape(name), split, offset)
		var page struct {
			Rows []struct {
				Row map[string]any `json:"row"`
			} `json:"rows"`
		}
		if err := getJSON(ctx, u, &page); err != nil {
			return 0, err
		}
		for _, r := range page.Rows {
			text, _ := r.Row[textCol].(string)
			if text == "" {
				continue
			}
			all = append(all, dataset.Example{
				Text:   text,
				Labels: map[string]any{"injection": positive(r.Row)},
				Split:  split,
				// Pas de famille : ces jeux n'en déclarent pas, et la source
				// entière comme famille empêcherait tout découpage interne.
				Meta: map[string]string{"source": name},
			})
		}
		return len(page.Rows), nil
	}
	if pages != nil {
		for _, p := range pages {
			if _, err := add("train", p*100); err != nil {
				return nil, err
			}
		}
	} else {
		for _, split := range []string{"train", "test"} {
			for offset := 0; ; offset += 100 {
				n, err := add(split, offset)
				if err != nil {
					return nil, err
				}
				if n < 100 {
					break
				}
			}
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	return all, dataset.WriteFile(path, all)
}

// getJSON réessaie avec un délai croissant quand le serveur limite le débit
// (429) ou échoue (5xx).
func getJSON(ctx context.Context, u string, v any) error {
	delay := 5 * time.Second
	for attempt := 1; ; attempt++ {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
		res, err := http.DefaultClient.Do(req)
		if err != nil {
			return err
		}
		if res.StatusCode == http.StatusOK {
			err = json.NewDecoder(res.Body).Decode(v)
			res.Body.Close()
			if err != nil {
				return fmt.Errorf("%s : %w", u, err)
			}
			return nil
		}
		res.Body.Close()
		if (res.StatusCode != http.StatusTooManyRequests && res.StatusCode < 500) || attempt == 6 {
			return fmt.Errorf("%s : HTTP %d", u, res.StatusCode)
		}
		log.Printf("HTTP %d, nouvel essai dans %s", res.StatusCode, delay)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
		}
		delay *= 2
	}
}

// readPromptGuard lit le corpus de prompt-guard (Xolo) : text, malicious,
// family. Données AGPL : à n'utiliser qu'en local, jamais versionnées ici.
func readPromptGuard(path string) ([]dataset.Example, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []dataset.Example
	dec := json.NewDecoder(f)
	for dec.More() {
		var r struct {
			Text      string `json:"text"`
			Malicious bool   `json:"malicious"`
			Family    string `json:"family"`
			Language  string `json:"language"`
		}
		if err := dec.Decode(&r); err != nil {
			return nil, err
		}
		out = append(out, dataset.Example{
			Text:   r.Text,
			Labels: map[string]any{"injection": r.Malicious},
			Family: "prompt-guard/" + r.Family,
			Meta:   map[string]string{"source": "prompt-guard", "lang": r.Language},
		})
	}
	return out, nil
}

// policyReport mesure le modèle sur la référence relue, par source.
func policyReport(ctx context.Context, m *indecis.Model, cases []dataset.Example) {
	inputs := make([]indecis.Input, len(cases))
	for i, e := range cases {
		inputs[i] = indecis.Input{Text: e.Text}
		if m.Paired() {
			inputs[i].Context = e.Context
		}
	}
	ds, err := m.DecideInputs(ctx, inputs...)
	if err != nil {
		log.Printf("référence : %v", err)
		return
	}
	type agg struct{ ok, n, fp, fn int }
	per := map[string]*agg{}
	var names []string
	total := &agg{}
	for i, e := range cases {
		src := e.Meta["source"]
		if per[src] == nil {
			per[src] = &agg{}
			names = append(names, src)
		}
		want := e.Labels["injection"] == true
		got := ds[i]["injection"].P >= 0.5
		for _, a := range []*agg{per[src], total} {
			a.n++
			switch {
			case got == want:
				a.ok++
			case got:
				a.fp++
			default:
				a.fn++
			}
		}
	}
	for _, n := range names {
		a := per[n]
		log.Printf("référence %-34s %3d/%3d  FP=%d FN=%d", n, a.ok, a.n, a.fp, a.fn)
	}
	log.Printf("référence %-34s %3d/%3d  FP=%d FN=%d", "TOTAL", total.ok, total.n, total.fp, total.fn)
}
