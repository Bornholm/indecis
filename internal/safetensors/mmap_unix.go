//go:build unix

package safetensors

import (
	"os"
	"syscall"
	"unsafe"
)

// mapFile projette le fichier en lecture seule. La projection vit autant
// que le processus : les tenseurs bruts qui en sont tirés la référencent.
func mapFile(path string) ([]byte, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, false, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, false, err
	}
	if st.Size() == 0 {
		return nil, false, nil
	}
	b, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		b, err := os.ReadFile(path) // système de fichiers sans projection
		return b, false, err
	}
	return b, true, nil
}

// evict signale au noyau que les pages de b ne seront plus lues : elles
// quittent la mémoire du processus, et seront relues dans le fichier si on
// y revient. Seules les pages entièrement couvertes par b sont concernées.
func evict(b []byte) {
	page := uintptr(os.Getpagesize())
	if len(b) == 0 {
		return
	}
	start := uintptr(unsafe.Pointer(&b[0]))
	first := (start + page - 1) &^ (page - 1)
	last := (start + uintptr(len(b))) &^ (page - 1)
	if last <= first {
		return
	}
	syscall.Madvise(b[first-start:last-start], syscall.MADV_DONTNEED)
}
