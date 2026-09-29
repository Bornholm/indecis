package decision

import (
	"context"
	"encoding/base64"
	"fmt"
	"image"
	_ "image/gif" // formats accepted in the state
	_ "image/jpeg"
	_ "image/png"
	"strings"

	"github.com/bornholm/genai/llm"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/vision"
)

// maxPixels bounds the size of a decoded image (64 megapixels): a small
// compressed file can declare a huge image.
const maxPixels = 64 << 20

// FromVision wraps an image model (SigLIP), designated by name. Its state
// is an image and all its questions are open: their criteria are compared
// to the image.
func FromVision(name string, v *vision.Model) *Client {
	return &Client{defaultDir: name, models: map[string]*indecis.Model{}, visions: map[string]*vision.Model{name: v}}
}

// visionModel returns the image model of dir, loading it if dir holds a
// SigLIP checkpoint, or nil for a text model.
func (c *Client) visionModel(dir string) (*vision.Model, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if v, ok := c.visions[dir]; ok {
		return v, nil
	}
	if _, ok := c.models[dir]; ok || !vision.IsModel(dir) {
		return nil, nil
	}
	v, err := vision.Load(dir, vision.WithInt8())
	if err != nil {
		return nil, fmt.Errorf("indecis: loading %s: %w", dir, err)
	}
	if c.visions == nil {
		c.visions = map[string]*vision.Model{}
	}
	c.visions[dir] = v
	return v, nil
}

func decideImage(ctx context.Context, name string, v *vision.Model, state any, questions llm.Questions) (llm.DecisionResponse, error) {
	img, err := stateImage(state)
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
func stateImage(state any) (image.Image, error) {
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
	cfg, _, err := image.DecodeConfig(strings.NewReader(string(raw)))
	if err != nil {
		return nil, llm.NewValidationError("state", "unreadable image: "+err.Error())
	}
	if cfg.Width*cfg.Height > maxPixels {
		return nil, llm.NewValidationError("state", fmt.Sprintf("image of %d×%d pixels, at most %d", cfg.Width, cfg.Height, maxPixels))
	}
	img, _, err := image.Decode(strings.NewReader(string(raw)))
	if err != nil {
		return nil, llm.NewValidationError("state", "unreadable image: "+err.Error())
	}
	return img, nil
}
