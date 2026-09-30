package siglip

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bornholm/indecis/internal/safetensors"
)

// A checkpoint missing a tensor gives an error, not a panic: the text
// tower loads during a request, and a panic there would leave it broken.
func TestLoadEncoderMissingTensor(t *testing.T) {
	cfg := Config{Hidden: 4, Layers: 2, Heads: 1, MLP: 8, Eps: 1e-6}
	vec := func(n int) safetensors.Tensor { return safetensors.Tensor{Shape: []int{n}, Data: make([]float32, n)} }
	path := filepath.Join(t.TempDir(), "model.safetensors")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	// Only the first LayerNorm of the first layer.
	err = safetensors.Write(f, map[string]safetensors.Tensor{
		"enc.layers.0.layer_norm1.weight": vec(4),
		"enc.layers.0.layer_norm1.bias":   vec(4),
	}, nil)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	sf, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, quantized := range []bool{false, true} {
		_, err := loadEncoder(cfg, weights{f: sf}, "enc", quantized, true)
		if err == nil || !strings.Contains(err.Error(), "missing") {
			t.Errorf("quantized=%v: error %v, want a missing tensor", quantized, err)
		}
	}
}

// A text tower shaped differently from the vision tower is refused, head
// count included: a wrong head count would split qkv wrongly, silently.
func TestReadConfigTowerShapes(t *testing.T) {
	for _, c := range []struct {
		text string
		ok   bool
	}{
		{`{}`, true},
		{`{"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12}`, true},
		{`{"num_attention_heads": 16}`, false},
		{`{"hidden_size": 1024}`, false},
	} {
		dir := t.TempDir()
		cfg := `{"model_type": "siglip", "vision_config": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12}, "text_config": ` + c.text + `}`
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := ReadConfig(dir); (err == nil) != c.ok {
			t.Errorf("text_config %s: error %v, accepted expected %v", c.text, err, c.ok)
		}
	}
}
