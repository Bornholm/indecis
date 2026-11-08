package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/bornholm/indecis"
)

// runEmbed writes the embedding of each input text: one JSON line
// {"id", "embedding"} per text, or, with -format f32, the vectors laid end
// to end in float32 little-endian (n × dimension), for bulk indexing.
func runEmbed(args []string) error {
	fs := flag.NewFlagSet("embed", flag.ExitOnError)
	model := fs.String("model", "", "model or backbone (sentence-transformers directory)")
	in := fs.String("in", "-", "input: one line per text, as plain text or JSON {\"id\", \"text\"} (- : standard input)")
	out := fs.String("out", "-", "output (- : standard output)")
	query := fs.Bool("query", false, "the texts are search queries: the model's query prompt comes first")
	dim := fs.Int("dim", 0, "Matryoshka truncation to this many dimensions (0: all)")
	maxLen := fs.Int("max-len", 0, "tokens read at most per text (0: the model's value, 256 generally)")
	format := fs.String("format", "jsonl", "jsonl or f32")
	batch := fs.Int("batch", 64, "texts embedded together")
	int8 := fs.Bool("int8", true, "int8 computation if the processor allows it")
	fs.Parse(args)
	if err := required(fs, "model"); err != nil {
		return err
	}
	if *format != "jsonl" && *format != "f32" {
		return fmt.Errorf("embed: -format jsonl or f32")
	}
	var opts []indecis.Option
	if *int8 {
		opts = append(opts, indecis.WithInt8())
	}
	if *maxLen > 0 {
		opts = append(opts, indecis.WithMaxLen(*maxLen))
	}
	m, err := indecis.Open(*model, opts...)
	if err != nil {
		return err
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
	var w io.Writer = os.Stdout
	if *out != "-" {
		f, err := os.Create(*out)
		if err != nil {
			return err
		}
		defer f.Close()
		w = f
	}
	bw := bufio.NewWriterSize(w, 1<<20)
	enc := json.NewEncoder(bw)

	ctx := context.Background()
	var ids []any
	var texts []string
	flush := func() error {
		if len(texts) == 0 {
			return nil
		}
		embed := m.EmbedDocuments
		if *query {
			embed = m.EmbedQueries
		}
		vecs, err := embed(ctx, texts...)
		if err != nil {
			return err
		}
		for i, v := range vecs {
			v = indecis.Truncate(v, *dim)
			if *format == "f32" {
				var b [4]byte
				for _, x := range v {
					binary.LittleEndian.PutUint32(b[:], math.Float32bits(x))
					if _, err := bw.Write(b[:]); err != nil {
						return err
					}
				}
				continue
			}
			if err := enc.Encode(map[string]any{"id": ids[i], "embedding": v}); err != nil {
				return err
			}
		}
		ids, texts = ids[:0], texts[:0]
		return nil
	}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	n := 0
	for sc.Scan() {
		line := sc.Text()
		var id any = n
		text := line
		if strings.HasPrefix(strings.TrimSpace(line), "{") {
			var row struct {
				ID   any    `json:"id"`
				Text string `json:"text"`
			}
			if err := json.Unmarshal([]byte(line), &row); err != nil {
				return fmt.Errorf("embed: line %d: %w", n+1, err)
			}
			text = row.Text
			if row.ID != nil {
				id = row.ID
			}
		}
		ids, texts = append(ids, id), append(texts, text)
		n++
		if len(texts) >= *batch {
			if err := flush(); err != nil {
				return err
			}
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	if err := flush(); err != nil {
		return err
	}
	// A write error held by the buffer (full disk, closed pipe) must fail
	// the command: an f32 file has no length to tell it was cut short.
	return bw.Flush()
}
