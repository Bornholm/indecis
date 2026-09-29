package siglip

import "fmt"

// Pixels converts an RGB image already resized to S×S, as interleaved
// bytes [S, S, 3], into the model's input: channel-major floats, each
// value mapped from [0, 255] to [-1, 1] (mean and std 0.5, as SigLIP's
// image processor does).
func Pixels(rgb []uint8, S int) ([]float32, error) {
	if len(rgb) != S*S*3 {
		return nil, fmt.Errorf("siglip: %d bytes, expected %d×%d×3", len(rgb), S, S)
	}
	out := make([]float32, 3*S*S)
	for i := 0; i < S*S; i++ {
		for c := 0; c < 3; c++ {
			// Same operations as the processor: rescale, then normalize.
			v := float32(rgb[i*3+c]) * float32(1.0/255)
			out[c*S*S+i] = (v - 0.5) / 0.5
		}
	}
	return out, nil
}
