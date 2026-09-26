// Command realdata collecte des textes réels pour l'exemple prompt-injection,
// étiquetés par leur provenance, et les dédoublonne contre les jeux
// d'évaluation.
//
//	go run ./tools/realdata -out ~/.cache/indecis/datasets/real \
//	    -exclude ~/.cache/indecis/datasets/deepset/prompt-injections.jsonl,...
//
// Sources (licences permissives) :
//   - Lakera/gandalf_ignore_instructions (MIT) : tentatives réelles, injection ;
//   - Lakera/mosscap_prompt_injection (MIT) : idem, échantillon ;
//   - TrustAIRLab/in-the-wild-jailbreak-prompts (MIT) : jailbreaks (injection)
//     et prompts ordinaires du même milieu (bénins) ;
//   - OpenAssistant/oasst2 (Apache-2.0) : messages d'utilisateurs, bénins ;
//   - allenai/WildChat-1M (ODC-BY) : premiers messages, non étiquetés (le
//     teacher s'en charge).
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"hash/fnv"
	"log"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"unicode"

	"github.com/bornholm/indecis/dataset"
)

var overrideRe = regexp.MustCompile(`(?i)ignore (all |the )?(previous|prior|above)|disregard (all |the )?(previous|prior|above)|forget (all |everything)`)

type source struct {
	name, config, split string
	total               int // lignes du split
	pages               int // pages de 100 à tirer (0 : tout)
	extract             func(row map[string]any) (text string, labels map[string]any, ok bool)
	// context, si défini, fournit le contexte (prompt système) d'une ligne.
	context func(row map[string]any) string
	// family, si défini, regroupe les lignes pour les découpages.
	family func(row map[string]any) string
	out    string
}

func main() {
	out := flag.String("out", "", "répertoire de sortie")
	exclude := flag.String("exclude", "", "jeux d'évaluation à exclure, séparés par des virgules")
	only := flag.String("only", "", "ne collecter que ces sorties (ex. spml,oasst2), séparées par des virgules")
	salt := flag.String("salt", "", "change les pages tirées, pour un échantillon distinct du premier")
	pages := flag.Int("pages", 0, "nombre de pages de 100 lignes à tirer par source échantillonnée (0 : celui de la source)")
	flag.Parse()
	if *out == "" {
		log.Fatal("-out est obligatoire")
	}
	ctx := context.Background()
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}

	index := newIndex()
	for _, p := range strings.Split(*exclude, ",") {
		if p == "" {
			continue
		}
		ex, err := dataset.ReadFile(p)
		if err != nil {
			log.Fatal(err)
		}
		for _, e := range ex {
			index.add(e.Text)
		}
	}
	log.Printf("%d textes d'évaluation indexés pour l'exclusion", index.n)

	attack := func(category string) map[string]any {
		return map[string]any{"injection": true, "category": category}
	}
	benign := map[string]any{"injection": false, "category": "none"}
	str := func(r map[string]any, k string) string { s, _ := r[k].(string); return strings.TrimSpace(s) }

	sources := []source{
		{name: "Lakera/gandalf_ignore_instructions", config: "default", split: "train", total: 777, out: "gandalf",
			extract: func(r map[string]any) (string, map[string]any, bool) { return str(r, "text"), attack("leak"), true }},
		{name: "Lakera/mosscap_prompt_injection", config: "default", split: "train", total: 223533, pages: 20, out: "mosscap",
			extract: func(r map[string]any) (string, map[string]any, bool) { return str(r, "prompt"), attack("leak"), true }},
		{name: "TrustAIRLab/in-the-wild-jailbreak-prompts", config: "jailbreak_2023_12_25", split: "train", total: 1405, out: "itw_jailbreak",
			extract: func(r map[string]any) (string, map[string]any, bool) {
				return str(r, "prompt"), attack("jailbreak"), true
			}},
		{name: "TrustAIRLab/in-the-wild-jailbreak-prompts", config: "regular_2023_12_25", split: "train", total: 13735, pages: 20, out: "itw_regular",
			extract: func(r map[string]any) (string, map[string]any, bool) {
				text := str(r, "prompt")
				// La source classe « ordinaires » les prompts qui commencent par
				// « Ignore all previous instructions. You are an expert… ».
				// Pour notre schéma, annuler les instructions précédentes est
				// une tentative de réorientation, quelle que soit l'intention :
				// sans ce réétiquetage, le modèle reçoit des consignes
				// contradictoires sur la formule d'attaque la plus courante.
				if overrideRe.MatchString(text) {
					return text, attack("override"), true
				}
				return text, benign, true
			}},
		{name: "OpenAssistant/oasst2", config: "default", split: "train", total: 128575, pages: 40, out: "oasst2",
			extract: func(r map[string]any) (string, map[string]any, bool) {
				if r["role"] != "prompter" || r["deleted"] == true {
					return "", nil, false
				}
				return str(r, "text"), benign, true
			}},
		{name: "reshabhs/SPML_Chatbot_Prompt_Injection", config: "default", split: "train", total: 16012, pages: 60, out: "spml",
			// SPML étiquette par rapport au prompt système : une demande hors
			// du périmètre de l'assistant y compte comme injection, ce qui
			// correspond à la politique (cas off_scope). La catégorie n'est
			// pas fournie, elle reste vide.
			extract: func(r map[string]any) (string, map[string]any, bool) {
				v, _ := r["Prompt injection"].(float64)
				return str(r, "User Prompt"), map[string]any{"injection": v == 1}, true
			},
			context: func(r map[string]any) string { return str(r, "System Prompt") },
			family: func(r map[string]any) string {
				h := fnv.New64a()
				h.Write([]byte(str(r, "System Prompt")))
				return fmt.Sprintf("spml/%x", h.Sum64())
			}},
		{name: "allenai/WildChat-1M", config: "default", split: "train", total: 837989, pages: 12, out: "wildchat",
			extract: func(r map[string]any) (string, map[string]any, bool) {
				conv, _ := r["conversation"].([]any)
				for _, turn := range conv {
					m, _ := turn.(map[string]any)
					if m["role"] == "user" {
						s, _ := m["content"].(string)
						return strings.TrimSpace(s), nil, true // à étiqueter par le teacher
					}
				}
				return "", nil, false
			}},
	}

	for _, s := range sources {
		if *only != "" && !strings.Contains(","+*only+",", ","+s.out+",") {
			continue
		}
		if *pages > 0 && s.pages > 0 {
			s.pages = *pages
		}
		ex, dropped, err := collect(ctx, s, index, *salt)
		if err != nil {
			log.Fatalf("%s/%s : %v", s.name, s.config, err)
		}
		path := filepath.Join(*out, s.out+".jsonl")
		if err := dataset.WriteFile(path, ex); err != nil {
			log.Fatal(err)
		}
		log.Printf("%-45s %-22s %5d exemples, %d écartés (doublons ou recouvrement avec l'évaluation) → %s",
			s.name, s.config, len(ex), dropped, path)
	}
}

func collect(ctx context.Context, s source, index *ngramIndex, salt string) ([]dataset.Example, int, error) {
	pages := (s.total + 99) / 100
	var offsets []int
	if s.pages == 0 || s.pages >= pages {
		for p := 0; p < pages; p++ {
			offsets = append(offsets, p*100)
		}
	} else {
		h := fnv.New64a()
		fmt.Fprint(h, s.name, s.config, salt)
		for _, p := range rand.New(rand.NewSource(int64(h.Sum64()))).Perm(pages)[:s.pages] {
			offsets = append(offsets, p*100)
		}
	}
	seen := newIndex()
	var out []dataset.Example
	dropped := 0
	for _, off := range offsets {
		u := fmt.Sprintf("https://datasets-server.huggingface.co/rows?dataset=%s&config=%s&split=%s&offset=%d&length=100",
			url.QueryEscape(s.name), url.QueryEscape(s.config), s.split, off)
		var page struct {
			Rows []struct {
				Row map[string]any `json:"row"`
			} `json:"rows"`
		}
		if err := getJSON(ctx, u, &page); err != nil {
			return nil, 0, err
		}
		for _, r := range page.Rows {
			text, labels, ok := s.extract(r.Row)
			if !ok || len(text) < 8 {
				continue
			}
			if index.overlaps(text) || seen.overlaps(text) {
				dropped++
				continue
			}
			seen.add(text)
			e := dataset.Example{
				Text: text, Labels: labels,
				Meta: map[string]string{"source": s.name + "/" + s.config},
			}
			if s.context != nil {
				e.Context = s.context(r.Row)
			}
			if s.family != nil {
				e.Family = s.family(r.Row)
			}
			out = append(out, e)
		}
		time.Sleep(time.Second) // ménage le datasets-server
	}
	return out, dropped, nil
}

// ngramIndex repère les quasi-doublons : deux textes se recouvrent s'ils
// partagent au moins la moitié des 5-grammes de mots du plus court, après
// normalisation (casse, ponctuation, espaces).
type ngramIndex struct {
	grams map[uint64][]int
	sizes []int
	n     int
}

func newIndex() *ngramIndex { return &ngramIndex{grams: map[uint64][]int{}} }

func shingles(text string) []uint64 {
	words := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) })
	if len(words) < 5 {
		h := fnv.New64a()
		h.Write([]byte(strings.Join(words, " ")))
		return []uint64{h.Sum64()}
	}
	set := map[uint64]bool{}
	for i := 0; i+5 <= len(words); i++ {
		h := fnv.New64a()
		h.Write([]byte(strings.Join(words[i:i+5], " ")))
		set[h.Sum64()] = true
	}
	out := make([]uint64, 0, len(set))
	for g := range set {
		out = append(out, g)
	}
	return out
}

func (x *ngramIndex) add(text string) {
	g := shingles(text)
	id := x.n
	x.n++
	x.sizes = append(x.sizes, len(g))
	for _, s := range g {
		x.grams[s] = append(x.grams[s], id)
	}
}

func (x *ngramIndex) overlaps(text string) bool {
	g := shingles(text)
	counts := map[int]int{}
	for _, s := range g {
		for _, id := range x.grams[s] {
			counts[id]++
		}
	}
	for id, c := range counts {
		if float64(c) >= 0.5*float64(min(len(g), x.sizes[id])) {
			return true
		}
	}
	return false
}

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
			return err
		}
		res.Body.Close()
		if (res.StatusCode != http.StatusTooManyRequests && res.StatusCode < 500) || attempt == 6 {
			return fmt.Errorf("%s : HTTP %d", u, res.StatusCode)
		}
		log.Printf("HTTP %d, nouvel essai dans %s", res.StatusCode, delay)
		time.Sleep(delay)
		delay *= 2
	}
}
