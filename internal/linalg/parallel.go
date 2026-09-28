package linalg

import (
	"runtime"
	"sync"
	"sync/atomic"
)

var maxWorkers atomic.Int64

// SetMaxWorkers caps the number of compute goroutines (0: GOMAXPROCS).
// For an isolated request on a laptop processor, a single fast core often
// beats all cores combined: frequency drops when several cores compute,
// and efficiency cores are slower.
func SetMaxWorkers(n int) { maxWorkers.Store(int64(max(n, 0))) }

// Workers returns the number of compute goroutines allowed.
func Workers() int {
	if n := int(maxWorkers.Load()); n > 0 {
		return min(n, runtime.GOMAXPROCS(0))
	}
	return runtime.GOMAXPROCS(0)
}

// Parallel splits [0, n) into slices of at least grain elements and calls
// fn on each, in parallel. The slices are disjoint.
func Parallel(n, grain int, fn func(lo, hi int)) {
	ParallelN(Workers(), n, grain, fn)
}

// ParallelN is Parallel with at most limit goroutines (also bounded by
// Workers).
func ParallelN(limit, n, grain int, fn func(lo, hi int)) {
	if n <= 0 {
		return
	}
	workers := max(1, min(limit, Workers()))
	if grain < 1 {
		grain = 1
	}
	chunks := min(workers, (n+grain-1)/grain)
	if chunks <= 1 {
		fn(0, n)
		return
	}
	step := (n + chunks - 1) / chunks
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += step {
		wg.Add(1)
		go parallelChunk(&wg, fn, lo, min(lo+step, n))
	}
	wg.Wait()
}

func parallelChunk(wg *sync.WaitGroup, fn func(lo, hi int), lo, hi int) {
	defer wg.Done()
	fn(lo, hi)
}

// Partials reduces [0, n) in fixed slices of grain elements: fn accumulates
// each slice into its own vector of width float64, then the vectors are
// summed in slice order.
//
// The splitting does not depend on the number of cores: the result is the
// same, bit for bit, on every machine. This is what makes training
// reproducible.
func Partials(n, grain, width int, fn func(lo, hi int, acc []float64)) []float64 {
	out := make([]float64, width)
	if n <= 0 {
		return out
	}
	if grain < 1 {
		grain = 1
	}
	chunks := (n + grain - 1) / grain
	accs := make([]float64, chunks*width)
	Parallel(chunks, 1, func(lo, hi int) {
		for c := lo; c < hi; c++ {
			fn(c*grain, min((c+1)*grain, n), accs[c*width:(c+1)*width])
		}
	})
	for c := 0; c < chunks; c++ {
		for i, v := range accs[c*width : (c+1)*width] {
			out[i] += v
		}
	}
	return out
}
