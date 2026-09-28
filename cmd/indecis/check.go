package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/bornholm/indecis"
)

// runCheck reports the acceleration available on this machine. It exits
// with status 1 when the portable SIMD is missing or emulated: inference
// then runs several times slower.
func runCheck(args []string) error {
	fs := flag.NewFlagSet("check", flag.ExitOnError)
	fs.Parse(args)
	h := indecis.DetectHardware()
	yesNo := func(b bool) string {
		if b {
			return "yes"
		}
		return "no"
	}
	fmt.Printf("version       %s\n", version)
	fmt.Printf("architecture  %s\n", h.Arch)
	switch {
	case !h.SIMD:
		fmt.Println("simd          not compiled in (build with GOEXPERIMENT=simd)")
	case h.SIMDEmulated:
		fmt.Printf("simd          emulated in pure Go: the processor lacks the instructions\n")
	default:
		fmt.Printf("simd          yes, %d-bit vectors\n", h.VectorBits)
	}
	if h.Arch == "amd64" {
		fmt.Printf("avx2 + fma    %s (float32 matrix kernel)\n", yesNo(h.AVX2FMA))
		fmt.Printf("avx-vnni      %s (int8 inference)\n", yesNo(h.AVXVNNI))
		if h.AVX2FMA && !h.Assembly {
			fmt.Println("assembly      turned off by INDECIS_NOASM=1")
		}
		if !h.AVXVNNI {
			fmt.Println("note          without AVX-VNNI, -int8 has no effect: models compute in float32")
		}
	}
	if !h.SIMD || h.SIMDEmulated {
		fmt.Println("result        slow: inference runs several times slower on this machine")
		os.Exit(1)
	}
	fmt.Println("result        ok")
	return nil
}
