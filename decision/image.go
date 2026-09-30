package decision

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	_ "image/gif" // formats accepted in the state
	_ "image/jpeg"
	_ "image/png"
	"io"
	"strings"

	"github.com/bornholm/genai/llm"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/vision"
)

// DefaultMaxImagePixels bounds the size of a decoded image when
// Client.MaxImagePixels is 0: a small compressed file can declare a huge
// image. The encoder reads 256×256; 16 megapixels leaves room for camera
// photos while a decoded image stays under about 64 MB (RGBA).
const DefaultMaxImagePixels = 16 << 20

// FromVision wraps an image model (SigLIP), designated by name. Its state
// is an image and all its questions are open: their criteria are compared
// to the image.
func FromVision(name string, v *vision.Model) *Client {
	return &Client{defaultDir: name, models: map[string]*indecis.Model{}, visions: map[string]*vision.Model{name: v}}
}

func decideImage(ctx context.Context, name string, v *vision.Model, state any, questions llm.Questions, maxPixels int) (llm.DecisionResponse, error) {
	img, err := stateImage(state, maxPixels)
	if err != nil {
		return nil, err
	}
	// A question named after one of the head's is learned; any other is
	// open. Both are answered with one pass of the encoder.
	schema := map[string]indecis.Question{}
	for _, q := range v.Schema() {
		schema[q.Name] = q
	}
	var learned []string
	open := make([]indecis.OpenQuestion, 0, len(questions))
	for id, q := range questions {
		if _, ok := schema[id]; ok {
			if err := compatible(id, q, schema); err != nil {
				return nil, err
			}
			learned = append(learned, id)
			continue
		}
		open = append(open, toOpen(id, q))
	}
	d, err := v.Decide(ctx, img, learned, open)
	if err != nil {
		return nil, err
	}
	answers := make(map[string]llm.Answer, len(questions))
	for id, q := range questions {
		if sq, ok := schema[id]; ok {
			answers[id] = toAnswer(q, sq, d[id])
		} else {
			answers[id] = toOpenAnswer(q, d[id])
		}
	}
	return llm.NewDecisionResponse(name, answers, llm.NewDecisionUsage(0, 0, 0)), nil
}

// stateImage decodes the image of a request's state: a data URL
// ("data:image/png;base64,..."), or an object {"image": ...} holding a data
// URL or bare base64. PNG, JPEG and GIF are accepted; URLs are not fetched.
// Images beyond maxPixels (0: DefaultMaxImagePixels) are refused before
// being decoded.
func stateImage(state any, maxPixels int) (image.Image, error) {
	if maxPixels <= 0 {
		maxPixels = DefaultMaxImagePixels
	}
	var s string
	switch v := state.(type) {
	case string:
		s = v
	case map[string]any:
		s, _ = v["image"].(string)
	}
	if s == "" {
		return nil, llm.NewValidationError("state", `this model reads an image: pass "data:image/png;base64,..." or {"image": "..."}`)
	}
	if strings.HasPrefix(s, "data:") {
		_, data, ok := strings.Cut(s, ",")
		if !ok || !strings.Contains(s[:len(s)-len(data)], ";base64") {
			return nil, llm.NewValidationError("state", "expected a base64 data URL")
		}
		s = data
	}
	raw, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, llm.NewValidationError("state", "invalid base64: "+err.Error())
	}
	r := bytes.NewReader(raw)
	cfg, _, err := image.DecodeConfig(r)
	if err != nil {
		return nil, llm.NewValidationError("state", "unreadable image: "+err.Error())
	}
	if cfg.Width <= 0 || cfg.Height <= 0 {
		return nil, llm.NewValidationError("state", fmt.Sprintf("image of %d×%d pixels", cfg.Width, cfg.Height))
	}
	if int64(cfg.Width)*int64(cfg.Height) > int64(maxPixels) {
		return nil, llm.NewValidationError("state", fmt.Sprintf("image of %d×%d pixels, at most %d", cfg.Width, cfg.Height, maxPixels))
	}
	r.Seek(0, io.SeekStart)
	img, _, err := image.Decode(r)
	if err != nil {
		return nil, llm.NewValidationError("state", "unreadable image: "+err.Error())
	}
	return img, nil
}
