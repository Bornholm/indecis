//go:build unix

package safetensors

import (
	"os"
	"syscall"
)

// mapFile projette le fichier en lecture seule. La projection vit autant
// que le processus : les tenseurs bruts qui en sont tirés la référencent.
func mapFile(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if st.Size() == 0 {
		return nil, nil
	}
	b, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return os.ReadFile(path) // système de fichiers sans projection
	}
	return b, nil
}
