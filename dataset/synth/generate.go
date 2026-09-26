package synth

import (
	"fmt"
	"hash/fnv"
	"io/fs"
	"math/rand"
	"path"
	"sort"
	"strconv"
	"strings"

	"github.com/bornholm/indecis/dataset"
)

// Corpus regroupe des gabarits et les gazetteers qu'ils utilisent.
type Corpus struct {
	Templates  []*Template
	Gazetteers map[string]*Gazetteer
}

// LoadFS lit récursivement les gabarits (*.tmpl) et les gazetteers (*.tsv)
// d'un système de fichiers. Un gabarit est nommé par son chemin sans
// extension, un gazetteer par son nom de fichier sans extension.
func LoadFS(fsys fs.FS, opts GazetteerOptions) (*Corpus, error) {
	c := &Corpus{Gazetteers: map[string]*Gazetteer{}}
	err := fs.WalkDir(fsys, ".", func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		switch path.Ext(p) {
		case ".tmpl":
			b, err := fs.ReadFile(fsys, p)
			if err != nil {
				return err
			}
			t, err := Parse(strings.TrimSuffix(p, ".tmpl"), string(b))
			if err != nil {
				return err
			}
			c.Templates = append(c.Templates, t)
		case ".tsv":
			f, err := fsys.Open(p)
			if err != nil {
				return err
			}
			defer f.Close()
			g, err := LoadGazetteer(f, opts)
			if err != nil {
				return fmt.Errorf("%s : %w", p, err)
			}
			name := strings.TrimSuffix(path.Base(p), ".tsv")
			if _, dup := c.Gazetteers[name]; dup {
				return fmt.Errorf("gazetteer %q en double", name)
			}
			c.Gazetteers[name] = g
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(c.Templates, func(i, j int) bool { return c.Templates[i].Name < c.Templates[j].Name })
	return c, c.Validate()
}

// Validate vérifie que chaque gazetteer utilisé existe et que chaque motif
// d'inclusion désigne au moins un gabarit. Une faute de frappe doit échouer
// au chargement, pas produire un corpus silencieusement appauvri.
func (c *Corpus) Validate() error {
	if len(c.Templates) == 0 {
		return fmt.Errorf("synth : aucun gabarit")
	}
	for _, t := range c.Templates {
		var err error
		visit(t.Body, t.Blocks, func(n Node) {
			if err != nil {
				return
			}
			switch v := n.(type) {
			case Pick:
				if c.Gazetteers[v.Set] == nil {
					err = fmt.Errorf("%s : gazetteer %q absent", t.Name, v.Set)
				}
			case Include:
				if len(c.matching(v.Pattern)) == 0 {
					err = fmt.Errorf("%s : aucun gabarit pour {{include:%s}}", t.Name, v.Pattern)
				}
			}
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func visit(nodes []Node, blocks map[string][]Node, fn func(Node)) {
	for _, n := range nodes {
		fn(n)
		switch v := n.(type) {
		case Optional:
			visit(v.Body, blocks, fn)
		case Transform:
			visit(v.Body, blocks, fn)
		case OneOf:
			for _, a := range v.Alts {
				visit(a, blocks, fn)
			}
		}
	}
	for _, b := range blocks {
		for _, n := range b {
			fn(n)
		}
	}
}

func (c *Corpus) matching(pattern string) []*Template {
	var out []*Template
	for _, t := range c.Templates {
		if ok, _ := path.Match(pattern, t.Family); ok {
			out = append(out, t)
		}
	}
	return out
}

// Options règle la génération.
type Options struct {
	Seed uint64
	// OptionalRate est la probabilité d'une section optionnelle sans
	// probabilité propre. 0,5 par défaut.
	OptionalRate float64
	// MaxDepth borne les inclusions imbriquées. 3 par défaut.
	MaxDepth int
	// Dedupe écarte les textes déjà produits.
	Dedupe bool
}

// Generate produit n exemples. Les gabarits de poids nul ne sont jamais tirés
// à la racine : ce sont des fragments destinés à être inclus.
func (c *Corpus) Generate(n int, opts Options) ([]dataset.Example, error) {
	if opts.OptionalRate == 0 {
		opts.OptionalRate = 0.5
	}
	if opts.MaxDepth == 0 {
		opts.MaxDepth = 3
	}
	var roots []*Template
	var total float64
	for _, t := range c.Templates {
		if t.Weight > 0 {
			roots = append(roots, t)
			total += t.Weight
		}
	}
	if len(roots) == 0 {
		return nil, fmt.Errorf("synth : aucun gabarit de poids positif")
	}
	pick := rand.New(rand.NewSource(int64(opts.Seed ^ 0x5eed)))
	seen := map[string]bool{}
	var out []dataset.Example
	for i := 0; len(out) < n; i++ {
		if i >= 20*n+100 {
			return out, fmt.Errorf("synth : %d exemples distincts seulement après %d tirages ; les gabarits manquent de variété", len(out), i)
		}
		t := roots[len(roots)-1]
		target := pick.Float64() * total
		for _, r := range roots {
			if target < r.Weight {
				t = r
				break
			}
			target -= r.Weight
		}
		e, err := c.Render(t, exampleSeed(opts.Seed, i), opts)
		if err != nil {
			return nil, err
		}
		if opts.Dedupe {
			// Le même message sous deux contextes différents est un autre
			// exemple : hors périmètre ici, légitime là.
			key := e.Context + "\x00" + e.Text
			if seen[key] {
				continue
			}
			seen[key] = true
		}
		out = append(out, e)
	}
	return out, nil
}

func exampleSeed(global uint64, i int) uint64 {
	h := fnv.New64a()
	fmt.Fprintf(h, "%d/%d", global, i)
	return h.Sum64()
}

// Render rend un gabarit avec une graine donnée.
func (c *Corpus) Render(t *Template, seed uint64, opts Options) (dataset.Example, error) {
	if opts.OptionalRate == 0 {
		opts.OptionalRate = 0.5
	}
	if opts.MaxDepth == 0 {
		opts.MaxDepth = 3
	}
	st := &state{
		c: c, opts: opts,
		rng:      rand.New(rand.NewSource(int64(seed))),
		slots:    map[string]string{},
		includes: map[string]bool{},
		split:    -1,
	}
	text, labels, err := st.template(t, 0)
	if err != nil {
		return dataset.Example{}, err
	}
	var context string
	if st.split >= 0 {
		context, text = strings.TrimSpace(text[:st.split]), strings.TrimLeft(text[st.split:], " \n")
	}
	var inc []string
	for f := range st.includes {
		inc = append(inc, f)
	}
	sort.Strings(inc)
	meta := map[string]string{"template": t.Name, "seed": strconv.FormatUint(seed, 10)}
	if t.Lang != "" {
		meta["lang"] = t.Lang
	}
	if len(inc) > 0 {
		meta["includes"] = strings.Join(inc, ",")
	}
	return dataset.Example{Context: context, Text: text, Labels: labels, Family: t.Family, Meta: meta}, nil
}

type state struct {
	c        *Corpus
	opts     Options
	rng      *rand.Rand
	slots    map[string]string
	includes map[string]bool
	split    int // position de {{user}} dans le texte racine, -1 sinon
}

// template rend un gabarit et retourne son texte et ses étiquettes.
func (st *state) template(t *Template, depth int) (string, map[string]any, error) {
	labels := make(map[string]any, len(t.Labels))
	for k, v := range t.Labels {
		labels[k] = v
	}
	r := &rendering{st: st, t: t, depth: depth, labels: labels, decisions: map[string]bool{}}
	var b strings.Builder
	if err := r.nodes(&b, t.Body); err != nil {
		return "", nil, err
	}
	return b.String(), labels, nil
}

type rendering struct {
	st        *state
	t         *Template
	depth     int
	labels    map[string]any
	decisions map[string]bool
}

func (r *rendering) nodes(b *strings.Builder, nodes []Node) error {
	rng := r.st.rng
	for _, n := range nodes {
		switch v := n.(type) {
		case Text:
			b.WriteString(v.S)
		case Pick:
			if v.Slot != "" {
				key := v.Set + ":" + v.Slot
				val, ok := r.st.slots[key]
				if !ok {
					val = r.st.c.Gazetteers[v.Set].PickValue(rng)
					r.st.slots[key] = val
				}
				b.WriteString(val)
				continue
			}
			b.WriteString(r.st.c.Gazetteers[v.Set].PickValue(rng))
		case OneOf:
			if err := r.nodes(b, v.Alts[rng.Intn(len(v.Alts))]); err != nil {
				return err
			}
		case Int:
			b.WriteString(strconv.Itoa(v.Min + rng.Intn(v.Max-v.Min+1)))
		case Digits:
			for i := 0; i < v.N; i++ {
				b.WriteByte(byte('0' + rng.Intn(10)))
			}
		case Pad:
			b.WriteString(strings.Repeat(" ", v.Min+rng.Intn(v.Max-v.Min+1)))
		case Repeat:
			count := v.Min + rng.Intn(v.Max-v.Min+1)
			for i := 0; i < count; i++ {
				if err := r.nodes(b, r.t.Blocks[v.Block]); err != nil {
					return err
				}
				b.WriteByte('\n')
			}
		case Optional:
			keep, ok := r.decisions[v.Name]
			if !ok {
				p := v.P
				if p < 0 {
					p = r.st.opts.OptionalRate
				}
				keep = rng.Float64() < p
				r.decisions[v.Name] = keep
			}
			if keep {
				if err := r.nodes(b, v.Body); err != nil {
					return err
				}
			}
		case Transform:
			var inner strings.Builder
			if err := r.nodes(&inner, v.Body); err != nil {
				return err
			}
			s := inner.String()
			if rng.Float64() < v.P {
				s = Transforms[v.Name](s, rng)
			}
			b.WriteString(s)
		case Include:
			if rng.Float64() >= v.P {
				continue
			}
			if r.depth >= r.st.opts.MaxDepth {
				return fmt.Errorf("synth : %s : inclusions trop profondes (> %d)", r.t.Name, r.st.opts.MaxDepth)
			}
			cands := r.st.c.matching(v.Pattern)
			sub := cands[rng.Intn(len(cands))]
			text, labels, err := r.st.template(sub, r.depth+1)
			if err != nil {
				return err
			}
			b.WriteString(text)
			r.st.includes[sub.Family] = true
			for k, val := range labels {
				if cur, isBool := r.labels[k].(bool); isBool && cur {
					continue // vrai l'emporte
				}
				r.labels[k] = val
			}
		case SetLabel:
			r.labels[v.Name] = v.Value
		case UserMark:
			if r.depth > 0 {
				// Gabarit inclus : son contexte n'a pas de sens dans le
				// texte hôte, il est abandonné.
				b.Reset()
				continue
			}
			if r.st.split >= 0 {
				return fmt.Errorf("synth : %s : {{user}} rendu deux fois", r.t.Name)
			}
			r.st.split = b.Len()
		}
	}
	return nil
}
