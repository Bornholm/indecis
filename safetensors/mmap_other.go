//go:build !linux

package safetensors

// evict does nothing: the syscall package offers madvise on Linux only, and
// the pages are left to the kernel elsewhere.
func evict([]byte) {}
