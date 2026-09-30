package vision

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/internal/safetensors"
)

// A trained image model is a directory holding vision.json (the encoder it
// was trained on and the head's shape) and head.safetensors (the head's
// weights). The encoder is not copied: the head only reads its outputs.
type manifest struct {
	// Backbone is the SigLIP directory, relative to this one or absolute.
	Backbone string         `json:"backbone"`
	Schema   indecis.Schema `json:"schema"`
	Patches  int            `json:"patches"`
	Hidden   int            `json:"hidden"`
	K        int            `json:"k"`
	Layer    int            `json:"layer,omitempty"`
}

const manifestFile = "vision.json"

// SaveHead writes a trained head to dir, pointing at the encoder in
// backbone.
func SaveHead(dir, backbone string, h *Head) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	absBackbone, err := filepath.Abs(backbone)
	if err != nil {
		return err
	}
	// Relative when the encoder is close (the model moves with it),
	// absolute otherwise.
	rel, err := filepath.Rel(absDir, absBackbone)
	if err != nil || strings.Count(rel, "..") > 2 {
		rel = absBackbone
	}
	b, err := json.MarshalIndent(manifest{Backbone: rel, Schema: h.Schema, Patches: h.T, Hidden: h.H, K: h.K, Layer: h.Layer}, "", "  ")
	if err != nil {
		return err
	}
	// A model directory is one whose manifest exists: the old manifest goes
	// first, each file is written aside then renamed, the new manifest
	// last. An interrupted save leaves no model rather than a manifest
	// over other weights.
	manifestPath := filepath.Join(dir, manifestFile)
	if err := os.Remove(manifestPath); err != nil && !os.IsNotExist(err) {
		return err
	}
	weights := filepath.Join(dir, "head.safetensors")
	f, err := os.Create(weights + ".tmp")
	if err != nil {
		return err
	}
	_, o := h.outputs()
	err = safetensors.Write(f, map[string]safetensors.Tensor{
		"w1": {Shape: []int{h.K, h.H}, Data: h.W1},
		"b1": {Shape: []int{h.K}, Data: h.B1},
		"w2": {Shape: []int{o, h.features()}, Data: h.W2},
		"b2": {Shape: []int{o}, Data: h.B2},
	}, nil)
	if err == nil {
		err = f.Sync() // the bytes are durable before the rename is
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = os.Rename(weights+".tmp", weights)
	}
	if err != nil {
		os.Remove(weights + ".tmp")
		return err
	}
	if err := writeSynced(manifestPath+".tmp", append(b, '\n')); err != nil {
		os.Remove(manifestPath + ".tmp")
		return err
	}
	return os.Rename(manifestPath+".tmp", manifestPath)
}

func writeSynced(path string, b []byte) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	_, err = f.Write(b)
	if err == nil {
		err = f.Sync()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	return err
}

// readManifest returns the manifest of a trained model directory, or nil
// if dir is a bare encoder.
func readManifest(dir string) (*manifest, error) {
	b, err := os.ReadFile(filepath.Join(dir, manifestFile))
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var m manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("vision: %s: %w", manifestFile, err)
	}
	if !filepath.IsAbs(m.Backbone) {
		m.Backbone = filepath.Join(dir, m.Backbone)
	}
	return &m, nil
}

func loadHead(dir string, m *manifest) (*Head, error) {
	if m.Patches < 1 || m.Hidden < 1 || m.K < 1 || m.Layer < 0 {
		return nil, fmt.Errorf("vision: manifest with patches %d, hidden %d, k %d, layer %d", m.Patches, m.Hidden, m.K, m.Layer)
	}
	h := &Head{Schema: m.Schema, T: m.Patches, H: m.Hidden, K: m.K, Layer: m.Layer}
	if err := h.Schema.Validate(); err != nil {
		return nil, err
	}
	f, err := safetensors.Open(filepath.Join(dir, "head.safetensors"))
	if err != nil {
		return nil, err
	}
	_, o := h.outputs()
	for _, p := range []struct {
		name string
		dst  *[]float32
		n    int
	}{{"w1", &h.W1, h.K * h.H}, {"b1", &h.B1, h.K}, {"w2", &h.W2, o * h.features()}, {"b2", &h.B2, o}} {
		t, ok, err := f.Tensor(p.name)
		if err != nil {
			return nil, err
		}
		if !ok || len(t.Data) != p.n {
			return nil, fmt.Errorf("vision: head.safetensors: %s missing or of the wrong size", p.name)
		}
		*p.dst = append([]float32(nil), t.Data...)
	}
	return h, nil
}
