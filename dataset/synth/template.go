// Package synth generates training examples from templates, with labels
// that are exact by construction.
//
// The engine is adapted from the corpus generator of go-anon (same author,
// GPL-3.0) and generalized: the labels are no longer named entities but the
// answers to the questions of an indecis schema.
//
// A template has a header and a body:
//
//	family: benign/support-ticket
//	lang: en
//	weight: 2
//	label.injection: false
//	label.category: none
//	---
//	Hello, {{one:my order|my parcel|my invoice}} {{pick:product}} has
//	[?late]still [/]not arrived.{{include:attack/*|p=0.3}}
//
// Body directives:
//
//	{{pick:set}}, {{pick:set:slot}}   value drawn from a gazetteer; a named slot
//	                                  keeps the same value in the whole example
//	{{one:a|b|c}}                     one alternative, in plain text (\n: line
//	                                  break)
//	{{one}}…{{|}}…{{/one}}            one alternative, which may hold directives
//	{{int:1-100}}, {{digits:6}}       numbers
//	{{pad:2-8}}                       spaces
//	[?name]…[/], [?name:0.3]…[/]      optional section (same name, same decision)
//	@block name … @end + {{LINES:name:1-5}}  repeated block
//	{{x:transform|p=0.5}}…{{/x}}      transformation of the rendered text (see Transforms)
//	{{include:pattern|p=0.4}}         renders another template whose family matches
//	                                  the pattern (path.Match) and merges its labels
//	{{label:name=value}}              sets a label from the rendered branch
//	{{user}}                          what comes before becomes the example's
//	                                  context (system prompt), what follows the
//	                                  judged text; in an included template,
//	                                  what comes before is dropped
//
// Label merging: the header gives the starting values, label directives
// replace them during rendering, and the labels of an included template are
// merged at the end of the inclusion. For a boolean label, true always wins:
// a benign document that holds an injection is an injection.
package synth

import (
	"fmt"
	"path"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Template is a parsed template.
type Template struct {
	Name   string
	Family string
	Lang   string
	Weight float64
	// Labels are the starting labels, declared in the header.
	Labels map[string]any
	Blocks map[string][]Node
	Body   []Node
}

// Node is an element of the AST.
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
		P    float64 // < 0: generator's default probability
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
	// UserMark separates the context (before) from the judged text (after).
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

// Parse parses a complete template.
func Parse(name, src string) (*Template, error) {
	t := &Template{Name: name, Weight: 1, Labels: map[string]any{}, Blocks: map[string][]Node{}}
	head, body, err := splitHeader(src)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	if err := t.parseHeader(head); err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	body, err = t.extractBlocks(body)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	body = strings.TrimSuffix(body, "\n")
	p := &parser{src: body}
	nodes, err := p.nodes("")
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	t.Body = nodes
	if err := t.checkBlocks(t.Body); err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
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
	return "", "", fmt.Errorf("unterminated header (expected \"---\" line)")
}

func (t *Template) parseHeader(head string) error {
	for _, l := range strings.Split(head, "\n") {
		l = strings.TrimSpace(l)
		if l == "" || strings.HasPrefix(l, "#") {
			continue
		}
		k, v, ok := strings.Cut(l, ":")
		if !ok {
			return fmt.Errorf("header: line %q without \":\"", l)
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
				return fmt.Errorf("header: invalid weight %q", v)
			}
			t.Weight = f
		case strings.HasPrefix(k, "label."):
			name := strings.TrimPrefix(k, "label.")
			if name == "" {
				return fmt.Errorf("header: label without name")
			}
			t.Labels[name] = parseValue(v)
		default:
			return fmt.Errorf("header: unknown key %q", k)
		}
	}
	if t.Family == "" {
		return fmt.Errorf("header: \"family\" is required")
	}
	return nil
}

// parseValue reads a label value: boolean, number, otherwise string.
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
				return "", fmt.Errorf("@block nested in %q", name)
			}
			in, name, cur = true, strings.TrimSpace(strings.TrimPrefix(trimmed, "@block ")), nil
			if _, dup := t.Blocks[name]; dup || name == "" {
				return "", fmt.Errorf("@block %q without name or duplicated", name)
			}
		case trimmed == "@end":
			if !in {
				return "", fmt.Errorf("@end without @block")
			}
			p := &parser{src: strings.Join(cur, "\n")}
			nodes, err := p.nodes("")
			if err != nil {
				return "", fmt.Errorf("block %q: %w", name, err)
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
		return "", fmt.Errorf("@block %q not closed", name)
	}
	return strings.Join(out, "\n"), nil
}

func (t *Template) checkBlocks(nodes []Node) error {
	for _, n := range nodes {
		var err error
		switch v := n.(type) {
		case Repeat:
			if _, ok := t.Blocks[v.Block]; !ok {
				err = fmt.Errorf("block %q referenced but not declared", v.Block)
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

// nodes parses up to one of the terminators (separated by ","), empty
// at the root level. The terminator encountered is consumed and recorded in end.
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
		// A literal "}}" betrays a malformed directive: it would have
		// produced its own text in the corpus.
		for _, n := range nodes {
			if t, ok := n.(Text); ok && strings.Contains(t.S, "}}") {
				p.err = fmt.Errorf("isolated \"}}\" in %q", t.S)
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
				return nil, "", fmt.Errorf("unclosed \"{{\"")
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
				return nil, "", fmt.Errorf("unclosed \"[?\"")
			}
			spec := p.src[2:end]
			p.src = p.src[end+1:]
			o := Optional{Name: spec, P: -1}
			if name, prob, ok := strings.Cut(spec, ":"); ok {
				f, err := strconv.ParseFloat(prob, 64)
				if err != nil || f < 0 || f > 1 {
					return nil, "", fmt.Errorf("[?%s]: invalid probability", spec)
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
		return nil, "", fmt.Errorf("unclosed section (%s expected)", strings.Join(terms, " or "))
	}
	flush()
	return nodes, "", nil
}

// matchingClose returns the index of the "}}" that closes the initial
// "{{" of s, accounting for nested directives ({{one:a|{{pick:x}}}}),
// -1 if there is none.
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

// splitTop splits s on sep, outside of nested directives.
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
		return nil, fmt.Errorf("empty directive {{}}")
	}
	if strings.HasPrefix(inner, "/") || inner == "|" {
		return nil, fmt.Errorf("unexpected {{%s}}", inner)
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
				return fmt.Errorf("{{%s}}: argument %q without \"=\"", inner, a)
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
			return 0, fmt.Errorf("{{%s}}: invalid p", inner)
		}
		return f, nil
	}

	switch kind {
	case "one":
		if hasSpec {
			// Inline form: alternatives can contain directives, but no
			// open section.
			var alts [][]Node
			for _, a := range splitTop(strings.TrimPrefix(inner, "one:"), '|') {
				// A directive fits on one line: \n marks the line
				// break in it.
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
			return nil, fmt.Errorf("{{pick}} without gazetteer")
		}
		set, slot, _ := strings.Cut(spec, ":")
		return Pick{Set: set, Slot: slot}, nil
	case "int", "pad":
		if !hasSpec {
			return nil, fmt.Errorf("{{%s}} without bounds", kind)
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
			return nil, fmt.Errorf("{{%s}}: invalid length", inner)
		}
		return Digits{N: n}, nil
	case "LINES":
		block, rng, ok := strings.Cut(spec, ":")
		if !ok {
			return nil, fmt.Errorf("{{%s}}: expected form {{LINES:block:n-m}}", inner)
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
			return nil, fmt.Errorf("{{%s}}: unknown transformation %q", inner, spec)
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
			return nil, fmt.Errorf("{{%s}}: invalid pattern", inner)
		}
		pr, err := prob(1)
		if err != nil {
			return nil, err
		}
		return Include{Pattern: spec, P: pr}, nil
	case "user":
		if hasSpec || argStr != "" {
			return nil, fmt.Errorf("{{%s}}: {{user}} takes no argument", inner)
		}
		return UserMark{}, nil
	case "label":
		name, value, ok := strings.Cut(spec, "=")
		if !ok || name == "" {
			return nil, fmt.Errorf("{{%s}}: expected form {{label:name=value}}", inner)
		}
		return SetLabel{Name: name, Value: parseValue(value)}, nil
	}
	return nil, fmt.Errorf("unknown directive {{%s}}", inner)
}

func parseRange(s string) (int, int, error) {
	a, b, ok := strings.Cut(s, "-")
	if !ok {
		n, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil {
			return 0, 0, fmt.Errorf("invalid bound %q", s)
		}
		return n, n, nil
	}
	lo, err1 := strconv.Atoi(strings.TrimSpace(a))
	hi, err2 := strconv.Atoi(strings.TrimSpace(b))
	if err1 != nil || err2 != nil || lo > hi {
		return 0, 0, fmt.Errorf("invalid bounds %q", s)
	}
	return lo, hi, nil
}
