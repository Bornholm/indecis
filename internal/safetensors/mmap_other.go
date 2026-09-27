//go:build !unix

package safetensors

import "os"

// mapFile lit le fichier entier là où la projection n'est pas prise en
// charge.
func mapFile(path string) ([]byte, bool, error) {
	b, err := os.ReadFile(path)
	return b, false, err
}

func evict([]byte) {}
