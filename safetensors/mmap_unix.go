//go:build unix

package safetensors

import (
	"os"
	"syscall"
	"unsafe"
)

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
