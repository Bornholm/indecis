package ocr

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"os"
)

// Shards hold lines ready for training: each line's text and its ink map
// as cut by CropLine, PNG-encoded (255 = ink). A shard is a sequence of
// records: a JSON header (text, width, meta) and the PNG, each preceded by
// its length (uint32, little endian), after the magic "ILN1".

const shardMagic = "ILN1"

// maxRecord bounds a record's declared length: a line 4,096 pixels wide and
// 96 high is 400 kB raw, far less as PNG. A damaged length is then an error,
// not a 4 GB allocation.
const maxRecord = 16 << 20

// ShardLine is a line read from a shard.
type ShardLine struct {
	Text string            `json:"text"`
	W    int               `json:"w"`
	Rows int               `json:"rows"`
	Meta map[string]string `json:"meta,omitempty"`
	Ink  []float32         `json:"-"` // [Rows, W]
}

// ShardWriter writes lines to a shard.
type ShardWriter struct {
	w   *bufio.Writer
	f   io.Closer
	buf bytes.Buffer
	enc png.Encoder
}

// CreateShard creates a shard file.
func CreateShard(path string) (*ShardWriter, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	w := &ShardWriter{w: bufio.NewWriterSize(f, 1<<20), f: f, enc: png.Encoder{CompressionLevel: png.BestSpeed}}
	if _, err := w.w.WriteString(shardMagic); err != nil {
		f.Close()
		return nil, err
	}
	return w, nil
}

// Write appends a line: its text and its crop.
func (w *ShardWriter) Write(text string, c Crop, meta map[string]string) error {
	rows := len(c.Ink) / c.W
	img := image.NewGray(image.Rect(0, 0, c.W, rows))
	for i, v := range c.Ink {
		img.Pix[i] = uint8(min(max(v*255+0.5, 0), 255))
	}
	w.buf.Reset()
	if err := w.enc.Encode(&w.buf, img); err != nil {
		return err
	}
	h, err := json.Marshal(ShardLine{Text: text, W: c.W, Rows: rows, Meta: meta})
	if err != nil {
		return err
	}
	for _, b := range [][]byte{h, w.buf.Bytes()} {
		if err := binary.Write(w.w, binary.LittleEndian, uint32(len(b))); err != nil {
			return err
		}
		if _, err := w.w.Write(b); err != nil {
			return err
		}
	}
	return nil
}

// Close flushes and closes the shard.
func (w *ShardWriter) Close() error {
	if err := w.w.Flush(); err != nil {
		w.f.Close()
		return err
	}
	return w.f.Close()
}

// ShardReader reads the lines of a shard in order.
type ShardReader struct {
	r    *bufio.Reader
	f    io.Closer
	path string
}

// OpenShard opens a shard file.
func OpenShard(path string) (*ShardReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	r := &ShardReader{r: bufio.NewReaderSize(f, 1<<20), f: f, path: path}
	magic := make([]byte, len(shardMagic))
	if _, err := io.ReadFull(r.r, magic); err != nil || string(magic) != shardMagic {
		f.Close()
		return nil, fmt.Errorf("%s: not a line shard", path)
	}
	return r, nil
}

// Next returns the next line, or io.EOF.
func (r *ShardReader) Next() (ShardLine, error) {
	var l ShardLine
	h, err := r.record()
	if err != nil {
		return l, err
	}
	if err := json.Unmarshal(h, &l); err != nil {
		return l, fmt.Errorf("%s: %w", r.path, err)
	}
	p, err := r.record()
	if err != nil {
		return l, unexpected(err)
	}
	img, err := png.Decode(bytes.NewReader(p))
	if err != nil {
		return l, fmt.Errorf("%s: %w", r.path, err)
	}
	g, ok := img.(*image.Gray)
	if !ok || g.Rect.Dx() != l.W || g.Rect.Dy() != l.Rows {
		return l, fmt.Errorf("%s: line image does not match its header", r.path)
	}
	l.Ink = make([]float32, len(g.Pix))
	for i, v := range g.Pix {
		l.Ink[i] = float32(v) / 255
	}
	return l, nil
}

func (r *ShardReader) record() ([]byte, error) {
	var n uint32
	if err := binary.Read(r.r, binary.LittleEndian, &n); err != nil {
		return nil, err // io.EOF between records
	}
	if n > maxRecord {
		return nil, fmt.Errorf("%s: record of %d bytes: not a line shard, or a damaged one", r.path, n)
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(r.r, b); err != nil {
		return nil, unexpected(err)
	}
	return b, nil
}

func unexpected(err error) error {
	if errors.Is(err, io.EOF) {
		return io.ErrUnexpectedEOF
	}
	return err
}

// Close closes the shard.
func (r *ShardReader) Close() error { return r.f.Close() }
