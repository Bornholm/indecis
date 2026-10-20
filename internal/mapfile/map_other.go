//go:build !unix

// Package mapfile maps files into memory, read-only.
package mapfile

import "os"

// Map reads the whole file where mmap is not supported.
func Map(path string) ([]byte, bool, error) {
	b, err := os.ReadFile(path)
	return b, false, err
}

// Unmap releases a mapping returned by Map, if it was mapped.
func Unmap(b []byte, mapped bool) {}
