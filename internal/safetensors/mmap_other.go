//go:build !unix

package safetensors

import "os"

// mapFile reads the whole file where mmap is not supported.
func mapFile(path string) ([]byte, bool, error) {
	b, err := os.ReadFile(path)
	return b, false, err
}

func evict([]byte) {}
