//go:build !unix

package safetensors

func evict([]byte) {}
