package decision

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"hash/crc32"
	"image"
	"image/color"
	"image/png"
	"strings"
	"testing"
)

func pngBytes(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	img.Set(0, 0, color.RGBA{255, 0, 0, 255})
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

// hugePNG is a PNG header declaring w×h pixels, with no pixel data: what an
// attacker sends to make a server allocate before checking.
func hugePNG(w, h uint32) []byte {
	var b bytes.Buffer
	b.WriteString("\x89PNG\r\n\x1a\n")
	ihdr := make([]byte, 13)
	binary.BigEndian.PutUint32(ihdr[0:], w)
	binary.BigEndian.PutUint32(ihdr[4:], h)
	ihdr[8], ihdr[9] = 8, 2 // 8-bit RGB
	binary.Write(&b, binary.BigEndian, uint32(len(ihdr)))
	b.WriteString("IHDR")
	b.Write(ihdr)
	binary.Write(&b, binary.BigEndian, crc32.ChecksumIEEE(append([]byte("IHDR"), ihdr...)))
	return b.Bytes()
}

// stateImage needs no model: every branch runs in CI.
func TestStateImage(t *testing.T) {
	small := base64.StdEncoding.EncodeToString(pngBytes(t, 4, 3))
	for _, c := range []struct {
		name    string
		state   any
		max     int
		wantErr string
	}{
		{"data URL", "data:image/png;base64," + small, 0, ""},
		{"object with data URL", map[string]any{"image": "data:image/png;base64," + small}, 0, ""},
		{"object with bare base64", map[string]any{"image": small}, 0, ""},
		{"base64 wrapped in lines", "data:image/png;base64," + wrap(small, 8), 0, ""},
		{"within a set bound", "data:image/png;base64," + small, 12, ""},
		{"text", "just text", 0, "base64"},
		{"empty object", map[string]any{"text": "x"}, 0, "reads an image"},
		{"number", 42, 0, "reads an image"},
		{"data URL without base64", "data:image/png," + small, 0, "base64 data URL"},
		{"corrupt base64", "data:image/png;base64,@@@", 0, "base64"},
		{"not an image", "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("hello")), 0, "unreadable"},
		{"beyond the default bound", "data:image/png;base64," + base64.StdEncoding.EncodeToString(hugePNG(8192, 8192)), 0, "at most"},
		{"beyond a set bound", "data:image/png;base64," + small, 10, "at most"},
	} {
		img, err := stateImage(c.state, c.max)
		switch {
		case c.wantErr == "" && err != nil:
			t.Errorf("%s: %v", c.name, err)
		case c.wantErr == "" && img.Bounds().Dx() != 4:
			t.Errorf("%s: %v", c.name, img.Bounds())
		case c.wantErr != "" && (err == nil || !strings.Contains(err.Error(), c.wantErr)):
			t.Errorf("%s: error %v, want %q", c.name, err, c.wantErr)
		}
	}
}

// wrap cuts s into lines of n characters, as base64(1) does.
func wrap(s string, n int) string {
	var b strings.Builder
	for len(s) > n {
		b.WriteString(s[:n] + "\r\n")
		s = s[n:]
	}
	return b.String() + s
}
