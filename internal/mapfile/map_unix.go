//go:build unix

// Package mapfile maps files into memory, read-only.
package mapfile

import (
	"os"
	"syscall"
)

// Map maps the file read-only and reports whether it is mapped: where
// mmap fails, the file is read into memory. The mapping lives as long as
// the process: the slices drawn from it reference it. An empty file gives
// nil.
func Map(path string) ([]byte, bool, error) {
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

// Unmap releases a mapping returned by Map, if it was mapped.
func Unmap(b []byte, mapped bool) {
	if mapped {
		syscall.Munmap(b)
	}
}
