// Package synth génère des exemples d'entraînement à partir de gabarits, avec
// des étiquettes exactes par construction.
//
// Le moteur est adapté du générateur de corpus de go-anon (même auteur,
// GPL-3.0), généralisé : les étiquettes ne sont plus des entités nommées mais
// les réponses aux questions d'un schéma indecis.
//
// Un gabarit a un en-tête et un corps :
//
//	family: benign/support-ticket
//	lang: fr
//	weight: 2
//	label.injection: false
//	label.category: none
//	---
//	Bonjour, {{one:ma commande|mon colis|ma facture}} {{pick:product}} n'est
//	[?late]toujours [/]pas arrivée.{{include:attack/*|p=0.3}}
//
// Directives du corps :
//
//	{{pick:set}}, {{pick:set:slot}}   valeur tirée d'un gazetteer ; un slot nommé
//	                                  garde la même valeur dans tout l'exemple
//	{{one:a|b|c}}                     une alternative, en texte simple (\n : saut
//	                                  de ligne)
//	{{one}}…{{|}}…{{/one}}            une alternative, pouvant contenir des directives
//	{{int:1-100}}, {{digits:6}}       nombres
//	{{pad:2-8}}                       espaces
//	[?nom]…[/], [?nom:0.3]…[/]        section optionnelle (même nom = même décision)
//	@block nom … @end + {{LINES:nom:1-5}}  bloc répété
//	{{x:transform|p=0.5}}…{{/x}}      transformation du texte rendu (voir Transforms)
//	{{include:motif|p=0.4}}           rend un autre gabarit dont la famille correspond
//	                                  au motif (path.Match) et fusionne ses étiquettes
//	{{label:nom=valeur}}              fixe une étiquette depuis la branche rendue
//	{{user}}                          ce qui précède devient le contexte de
//	                                  l'exemple (prompt système), ce qui suit
//	                                  le texte jugé ; dans un gabarit inclus,
//	                                  ce qui précède est abandonné
//
// Fusion des étiquettes : l'en-tête donne les valeurs de départ, les
// directives label les remplacent au fil du rendu, et les étiquettes d'un
// gabarit inclus s'y ajoutent à la fin de l'inclusion. Pour une étiquette
// booléenne, vrai l'emporte toujours : un document bénin qui contient une
// injection est une injection.
package synth

import (
	"fmt"
	"path"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Template est un gabarit analysé.
type Template struct {
	Name   string
	Family string
	Lang   string
	Weight float64
	// Labels sont les étiquettes de départ, déclarées dans l'en-tête.
	Labels map[string]any
	Blocks map[string][]Node
	Body   []Node
}

// Node est un élément de l'AST.
type Node interface{ node() }

type (
	Text   struct{ S string }
	Pick   struct{ Set, Slot string }
	OneOf  struct{ Alts [][]Node }
	Int    struct{ Min, Max int }
	Digits struct{ N int }
	Pad    struct{ Min, Max int }
	Repeat struct {
		Block    string
		Min, Max int
	}
	Optional struct {
		Name string
		P    float64 // < 0 : probabilité par défaut du générateur
		Body []Node
	}
	Transform struct {
		Name string
		P    float64
		Body []Node
	}
	Include struct {
		Pattern string
		P       float64
	}
	SetLabel struct {
		Name  string
		Value any
	}
	// UserMark sépare le contexte (avant) du texte jugé (après).
	UserMark struct{}
)

func (Text) node()      {}
func (Pick) node()      {}
func (OneOf) node()     {}
func (Int) node()       {}
func (Digits) node()    {}
func (Pad) node()       {}
func (Repeat) node()    {}
func (Optional) node()  {}
func (Transform) node() {}
func (Include) node()   {}
func (SetLabel) node()  {}
func (UserMark) node()  {}

// Parse analyse un gabarit complet.
func Parse(name, src string) (*Template, error) {
	t := &Template{Name: name, Weight: 1, Labels: map[string]any{}, Blocks: map[string][]Node{}}
	head, body, err := splitHeader(src)
	if err != nil {
		return nil, fmt.Errorf("%s : %w", name, err)
	}
	if err := t.parseHeader(head); err != nil {
		return nil, fmt.Errorf("%s : %w", name, err)
	}
	body, err = t.extractBlocks(body)
	if err != nil {
		return nil, fmt.Errorf("%s : %w", name, err)
	}
	body = strings.TrimSuffix(body, "\n")
	p := &parser{src: body}
	nodes, err := p.nodes("")
	if err != nil {
		return nil, fmt.Errorf("%s : %w", name, err)
	}
	t.Body = nodes
	if err := t.checkBlocks(t.Body); err != nil {
		return nil, fmt.Errorf("%s : %w", name, err)
	}
	return t, nil
}

func splitHeader(src string) (head, body string, err error) {
	lines := strings.Split(src, "\n")
	for i, l := range lines {
		if strings.TrimSpace(l) == "---" {
			return strings.Join(lines[:i], "\n"), strings.Join(lines[i+1:], "\n"), nil
		}
	}
	return "", "", fmt.Errorf("en-tête non terminé (ligne « --- » attendue)")
}

func (t *Template) parseHeader(head string) error {
	for _, l := range strings.Split(head, "\n") {
		l = strings.TrimSpace(l)
		if l == "" || strings.HasPrefix(l, "#") {
			continue
		}
		k, v, ok := strings.Cut(l, ":")
		if !ok {
			return fmt.Errorf("en-tête : ligne %q sans « : »", l)
		}
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		switch {
		case k == "family":
			t.Family = v
		case k == "lang":
			t.Lang = v
		case k == "weight":
			f, err := strconv.ParseFloat(v, 64)
			if err != nil || f < 0 {
				return fmt.Errorf("en-tête : weight invalide %q", v)
			}
			t.Weight = f
		case strings.HasPrefix(k, "label."):
			name := strings.TrimPrefix(k, "label.")
			if name == "" {
				return fmt.Errorf("en-tête : étiquette sans nom")
			}
			t.Labels[name] = parseValue(v)
		default:
			return fmt.Errorf("en-tête : clé inconnue %q", k)
		}
	}
	if t.Family == "" {
		return fmt.Errorf("en-tête : « family » est obligatoire")
	}
	return nil
}

// parseValue lit une valeur d'étiquette : booléen, nombre, sinon chaîne.
func parseValue(v string) any {
	switch v {
	case "true":
		return true
	case "false":
		return false
	}
	if f, err := strconv.ParseFloat(v, 64); err == nil {
		return f
	}
	return v
}

func (t *Template) extractBlocks(body string) (string, error) {
	var out, cur []string
	name := ""
	in := false
	for _, l := range strings.Split(body, "\n") {
		trimmed := strings.TrimSpace(l)
		switch {
		case strings.HasPrefix(trimmed, "@block "):
			if in {
				return "", fmt.Errorf("@block imbriqué dans %q", name)
			}
			in, name, cur = true, strings.TrimSpace(strings.TrimPrefix(trimmed, "@block ")), nil
			if _, dup := t.Blocks[name]; dup || name == "" {
				return "", fmt.Errorf("@block %q sans nom ou en double", name)
			}
		case trimmed == "@end":
			if !in {
				return "", fmt.Errorf("@end sans @block")
			}
			p := &parser{src: strings.Join(cur, "\n")}
			nodes, err := p.nodes("")
			if err != nil {
				return "", fmt.Errorf("bloc %q : %w", name, err)
			}
			t.Blocks[name] = nodes
			in = false
		case in:
			cur = append(cur, l)
		default:
			out = append(out, l)
		}
	}
	if in {
		return "", fmt.Errorf("@block %q non fermé", name)
	}
	return strings.Join(out, "\n"), nil
}

func (t *Template) checkBlocks(nodes []Node) error {
	for _, n := range nodes {
		var err error
		switch v := n.(type) {
		case Repeat:
			if _, ok := t.Blocks[v.Block]; !ok {
				err = fmt.Errorf("bloc %q référencé mais non déclaré", v.Block)
			}
		case Optional:
			err = t.checkBlocks(v.Body)
		case Transform:
			err = t.checkBlocks(v.Body)
		case OneOf:
			for _, alt := range v.Alts {
				if err = t.checkBlocks(alt); err != nil {
					break
				}
			}
		}
		if err != nil {
			return err
		}
	}
	return nil
}

type parser struct {
	src string
	err error
}

// nodes analyse jusqu'à l'un des terminateurs (séparés par « , »), vide au
// niveau racine. Le terminateur rencontré est consommé et mémorisé dans end.
func (p *parser) nodes(terminators string) ([]Node, error) {
	nodes, _, err := p.nodesUntil(splitTerms(terminators))
	if err == nil {
		err = p.err
	}
	return nodes, err
}

func splitTerms(s string) []string {
	if s == "" {
		return nil
	}
	return strings.Split(s, ",")
}

func (p *parser) nodesUntil(terms []string) ([]Node, string, error) {
	var nodes []Node
	var lit strings.Builder
	flush := func() {
		if lit.Len() > 0 {
			nodes = append(nodes, Text{S: lit.String()})
			lit.Reset()
		}
	}
	defer func() {
		// Un « }} » littéral trahit une directive mal formée : elle aurait
		// produit son propre texte dans le corpus.
		for _, n := range nodes {
			if t, ok := n.(Text); ok && strings.Contains(t.S, "}}") {
				p.err = fmt.Errorf("« }} » isolé dans %q", t.S)
			}
		}
	}()
	for len(p.src) > 0 {
		for _, term := range terms {
			if strings.HasPrefix(p.src, term) {
				flush()
				p.src = p.src[len(term):]
				return nodes, term, nil
			}
		}
		switch {
		case strings.HasPrefix(p.src, "{{"):
			end := matchingClose(p.src)
			if end < 0 {
				return nil, "", fmt.Errorf("« {{ » non fermé")
			}
			inner := p.src[2:end]
			p.src = p.src[end+2:]
			n, err := p.directive(inner)
			if err != nil {
				return nil, "", err
			}
			flush()
			nodes = append(nodes, n)
		case strings.HasPrefix(p.src, "[?"):
			end := strings.Index(p.src, "]")
			if end < 0 {
				return nil, "", fmt.Errorf("« [? » non fermé")
			}
			spec := p.src[2:end]
			p.src = p.src[end+1:]
			o := Optional{Name: spec, P: -1}
			if name, prob, ok := strings.Cut(spec, ":"); ok {
				f, err := strconv.ParseFloat(prob, 64)
				if err != nil || f < 0 || f > 1 {
					return nil, "", fmt.Errorf("[?%s] : probabilité invalide", spec)
				}
				o.Name, o.P = name, f
			}
			body, _, err := p.nodesUntil([]string{"[/]"})
			if err != nil {
				return nil, "", err
			}
			o.Body = body
			flush()
			nodes = append(nodes, o)
		default:
			r, size := utf8.DecodeRuneInString(p.src)
			lit.WriteRune(r)
			p.src = p.src[size:]
		}
	}
	if len(terms) > 0 {
		return nil, "", fmt.Errorf("section non fermée (%s attendu)", strings.Join(terms, " ou "))
	}
	flush()
	return nodes, "", nil
}

// matchingClose retourne l'indice du « }} » qui ferme le « {{ » initial de
// s, en tenant compte des directives imbriquées ({{one:a|{{pick:x}}}}),
// -1 s'il n'y en a pas.
func matchingClose(s string) int {
	depth := 0
	for i := 0; i+1 < len(s); i++ {
		switch {
		case s[i] == '{' && s[i+1] == '{':
			depth++
			i++
		case s[i] == '}' && s[i+1] == '}':
			depth--
			if depth == 0 {
				return i
			}
			i++
		}
	}
	return -1
}

// splitTop découpe s sur sep, hors des directives imbriquées.
func splitTop(s string, sep byte) []string {
	var out []string
	depth, start := 0, 0
	for i := 0; i < len(s); i++ {
		switch {
		case i+1 < len(s) && s[i] == '{' && s[i+1] == '{':
			depth++
			i++
		case i+1 < len(s) && s[i] == '}' && s[i+1] == '}':
			depth--
			i++
		case s[i] == sep && depth == 0:
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	return append(out, s[start:])
}

func (p *parser) directive(inner string) (Node, error) {
	if inner == "" {
		return nil, fmt.Errorf("directive vide {{}}")
	}
	if strings.HasPrefix(inner, "/") || inner == "|" {
		return nil, fmt.Errorf("{{%s}} inattendu", inner)
	}
	head, argStr, _ := strings.Cut(inner, "|")
	kind, spec, hasSpec := strings.Cut(head, ":")

	args := map[string]string{}
	parseArgs := func() error {
		for _, a := range strings.Split(argStr, "|") {
			a = strings.TrimSpace(a)
			if a == "" {
				continue
			}
			k, v, ok := strings.Cut(a, "=")
			if !ok {
				return fmt.Errorf("{{%s}} : argument %q sans « = »", inner, a)
			}
			args[strings.TrimSpace(k)] = strings.TrimSpace(v)
		}
		return nil
	}
	prob := func(def float64) (float64, error) {
		s, ok := args["p"]
		if !ok {
			return def, nil
		}
		f, err := strconv.ParseFloat(s, 64)
		if err != nil || f < 0 || f > 1 {
			return 0, fmt.Errorf("{{%s}} : p invalide", inner)
		}
		return f, nil
	}

	switch kind {
	case "one":
		if hasSpec {
			// Forme en ligne : les alternatives peuvent contenir des
			// directives, mais pas de section ouverte.
			var alts [][]Node
			for _, a := range splitTop(strings.TrimPrefix(inner, "one:"), '|') {
				// Une directive tient sur une ligne : \n y note le saut
				// de ligne.
				sub := &parser{src: strings.ReplaceAll(a, `\n`, "\n")}
				nodes, err := sub.nodes("")
				if err != nil {
					return nil, fmt.Errorf("{{%s}} : %w", inner, err)
				}
				alts = append(alts, nodes)
			}
			return OneOf{Alts: alts}, nil
		}
		var alts [][]Node
		for {
			body, term, err := p.nodesUntil([]string{"{{|}}", "{{/one}}"})
			if err != nil {
				return nil, err
			}
			alts = append(alts, body)
			if term == "{{/one}}" {
				return OneOf{Alts: alts}, nil
			}
		}
	case "pick":
		if !hasSpec {
			return nil, fmt.Errorf("{{pick}} sans gazetteer")
		}
		set, slot, _ := strings.Cut(spec, ":")
		return Pick{Set: set, Slot: slot}, nil
	case "int", "pad":
		if !hasSpec {
			return nil, fmt.Errorf("{{%s}} sans bornes", kind)
		}
		lo, hi, err := parseRange(spec)
		if err != nil {
			return nil, fmt.Errorf("{{%s}} : %w", inner, err)
		}
		if kind == "pad" {
			return Pad{Min: lo, Max: hi}, nil
		}
		return Int{Min: lo, Max: hi}, nil
	case "digits":
		n, err := strconv.Atoi(spec)
		if err != nil || n <= 0 || n > 64 {
			return nil, fmt.Errorf("{{%s}} : longueur invalide", inner)
		}
		return Digits{N: n}, nil
	case "LINES":
		block, rng, ok := strings.Cut(spec, ":")
		if !ok {
			return nil, fmt.Errorf("{{%s}} : forme attendue {{LINES:bloc:n-m}}", inner)
		}
		lo, hi, err := parseRange(rng)
		if err != nil {
			return nil, fmt.Errorf("{{%s}} : %w", inner, err)
		}
		return Repeat{Block: block, Min: lo, Max: hi}, nil
	case "x":
		if err := parseArgs(); err != nil {
			return nil, err
		}
		if _, ok := Transforms[spec]; !ok {
			return nil, fmt.Errorf("{{%s}} : transformation %q inconnue", inner, spec)
		}
		pr, err := prob(1)
		if err != nil {
			return nil, err
		}
		body, _, err := p.nodesUntil([]string{"{{/x}}"})
		if err != nil {
			return nil, err
		}
		return Transform{Name: spec, P: pr, Body: body}, nil
	case "include":
		if err := parseArgs(); err != nil {
			return nil, err
		}
		if _, err := path.Match(spec, ""); err != nil || spec == "" {
			return nil, fmt.Errorf("{{%s}} : motif invalide", inner)
		}
		pr, err := prob(1)
		if err != nil {
			return nil, err
		}
		return Include{Pattern: spec, P: pr}, nil
	case "user":
		if hasSpec || argStr != "" {
			return nil, fmt.Errorf("{{%s}} : {{user}} ne prend pas d'argument", inner)
		}
		return UserMark{}, nil
	case "label":
		name, value, ok := strings.Cut(spec, "=")
		if !ok || name == "" {
			return nil, fmt.Errorf("{{%s}} : forme attendue {{label:nom=valeur}}", inner)
		}
		return SetLabel{Name: name, Value: parseValue(value)}, nil
	}
	return nil, fmt.Errorf("directive inconnue {{%s}}", inner)
}

func parseRange(s string) (int, int, error) {
	a, b, ok := strings.Cut(s, "-")
	if !ok {
		n, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil {
			return 0, 0, fmt.Errorf("borne invalide %q", s)
		}
		return n, n, nil
	}
	lo, err1 := strconv.Atoi(strings.TrimSpace(a))
	hi, err2 := strconv.Atoi(strings.TrimSpace(b))
	if err1 != nil || err2 != nil || lo > hi {
		return 0, 0, fmt.Errorf("bornes invalides %q", s)
	}
	return lo, hi, nil
}
