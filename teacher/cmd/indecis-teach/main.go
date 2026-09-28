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
		fmt.Fprintln(os.Stderr, "usage : indecis-teach rewrite|label [options]")
		os.Exit(2)
	}
	cmd := os.Args[1]
	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	envFile := fs.String("env", ".env", "fichier de configuration GENAI_*")
	in := fs.String("in", "", "dataset d'entrée (JSONL)")
	out := fs.String("out", "", "dataset de sortie (JSONL)")
	cachePath := fs.String("cache", "teacher-cache.jsonl", "cache des réponses")
	maxCalls := fs.Int("max-calls", 0, "appels réels au plus (0 : illimité)")
	concurrency := fs.Int("concurrency", 2, "appels simultanés")
	interval := fs.Duration("interval", 2*time.Second, "intervalle minimal entre deux appels (ménage le quota du fournisseur)")
	retries := fs.Int("retries", 5, "nouvelles tentatives sur erreur temporaire (429, 5xx)")
	retryDelay := fs.Duration("retry-delay", 30*time.Second, "première attente avant une nouvelle tentative, doublée ensuite")
	limit := fs.Int("limit", 0, "ne traiter que les n premiers exemples")
	instrFile := fs.String("instructions", "", "rewrite : une instruction par ligne, attribuée en tourniquet")
	variants := fs.Int("variants", 2, "rewrite : variantes par exemple")
	schemaFile := fs.String("schema", "", "label : schéma JSON (liste de questions) ou indecis.json d'un modèle")
	teachersFile := fs.String("teachers", "", "label : fichier YAML de teachers en ligne de commande ; active le consensus. rewrite : avec -teacher, réécrit par lots avec ce teacher")
	teacherID := fs.String("teacher", "", "rewrite -teachers : identifiant du teacher à utiliser")
	disagreements := fs.String("disagreements", "", "label -teachers : fichier JSONL des désaccords à relire")
	guidelines := fs.String("guidelines", "", "label : politique d'étiquetage (Markdown) transmise aux teachers")
	verify := fs.Bool("verify", false, "label : réétiqueter et écarter les exemples dont le teacher contredit l'étiquette noul existante")
	keepUnverified := fs.Bool("keep-unverified", false, "label -verify : garder, marqués verified=false, les exemples que le teacher n'a pas jugés")
	cacheOnly := fs.Bool("cache-only", false, "rejouer le cache sans aucun appel au LLM")
	fs.Parse(os.Args[2:])
	if *in == "" || *out == "" {
		log.Fatal("-in et -out sont obligatoires")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	var t *teacher.Teacher
	switch {
	case *teachersFile != "" && cmd == "rewrite":
		// Un seul teacher suffit à réécrire : celui que désigne -teacher.
		cache, err := teacher.OpenCache(*cachePath)
		if err != nil {
			log.Fatal(err)
		}
		if t, err = harnessTeacher(*teachersFile, *teacherID, cache, *maxCalls, *cacheOnly); err != nil {
			log.Fatal(err)
		}
	case *teachersFile != "":
		if cmd != "label" {
			log.Fatal("-teachers ne s'applique qu'à label et rewrite")
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
		// Débit borné, puis nouvelles tentatives espacées sur 429 : une
		// limite de quota se contourne en attendant, pas en insistant.
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
		log.Fatalf("commande %q inconnue", cmd)
	}
	// Le résultat partiel est toujours écrit : les appels déjà payés ne
	// doivent pas être perdus sur une limite de débit ou une coupure.
	if werr := dataset.WriteFile(*out, result); werr != nil {
		log.Fatal(werr)
	}
	log.Printf("%s : %d soumis, %d produits, %d depuis le cache, %d refus ou réponses invalides, %d ignorés (hors cache), %d appels réels",
		cmd, stats.Requested, stats.Done, stats.Cached, stats.Refused, stats.Skipped, t.Calls())
	if err != nil {
		if errors.Is(err, teacher.ErrBudget) {
			log.Printf("budget épuisé : résultat partiel écrit dans %s", *out)
			return
		}
		log.Printf("interrompu : %v", err)
		log.Printf("résultat partiel écrit dans %s ; relancer reprend depuis le cache", *out)
		os.Exit(1)
	}
}

// rewrite attribue les instructions en tourniquet : chaque exemple reçoit
// une seule instruction, la dépense reste d'un appel par exemple.
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
		log.Printf("« %s » : %d → %d variantes (%d refus)", instr, st.Requested, st.Done, st.Refused)
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
	// Vérification : le teacher étiquette sans voir les étiquettes
	// existantes ; on garde l'exemple, avec ses étiquettes d'origine
	// complétées, seulement s'il est d'accord sur chaque question noul.
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
	log.Printf("vérification : %d gardés dont %d non vérifiés, %d écartés (désaccord du teacher)", len(kept), unverified, disagree)
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

// readSchema accepte une liste de questions ou le indecis.json d'un modèle.
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

// modelName lit le nom du modèle dans le fichier d'environnement, pour la
// clé de cache ; il n'est pas secret, contrairement à la clé d'API.
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
