//go:build unix

package safetensors

import (
	"os"
	"syscall"
	"unsafe"
)

// mapFile maps the file read-only. The mapping lives as long as the
// process: the raw tensors drawn from it reference it.
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
		b, err := os.ReadFile(path) // filesystem without mmap support
		return b, false, err
	}
	return b, true, nil
}

// evict tells the kernel that the pages of b will no longer be read: they
// leave the process's memory, and will be reread from the file if it is
// accessed again. Only pages fully covered by b are affected.
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
