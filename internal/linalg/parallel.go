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
// fn on each, in parallel. The slices are disjoint; fn may be called
// several times by the same goroutine.
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
	if workers == 1 || n <= grain {
		fn(0, n)
		return
	}
	// Guided slices, taken as they come: each is a share of what remains
	// (half of it divided among the workers), a multiple of grain, so of
	// the int8 kernel's rows when grain is, except the last. From 4 ×
	// workers × grain elements on (training batches, attention heads), the
	// first slices are long and the last ones short, so that on a hybrid
	// processor a slow core (E, LP-E) holding the last slice keeps the
	// others waiting a short time. Below, every slice is grain elements:
	// this is the case of inference, where shorter last slices were
	// measured no faster (see sliceRows). The calling goroutine takes
	// slices too. The slices are disjoint, so the result does not depend on
	// who computes which. If fn panics on the calling goroutine, the panic waits for
	// the workers: the caller's buffers are not reused while they write.
	// fn must therefore not wait on the caller. A panic on a worker
	// stops the process, as in any goroutine.
	workers = min(workers, (n+grain-1)/grain)
	var next atomic.Int64
	var wg sync.WaitGroup
	defer wg.Wait()
	wg.Add(workers - 1)
	for w := 1; w < workers; w++ {
		go func() {
			defer wg.Done()
			guided(&next, fn, n, grain, workers)
		}()
	}
	guided(&next, fn, n, grain, workers)
}

// guided takes slices of [0, n) from next until none remain.
func guided(next *atomic.Int64, fn func(lo, hi int), n, grain, workers int) {
	for {
		lo := int(next.Load())
		if lo >= n {
			return
		}
		step := max(1, (n-lo)/(2*workers)/grain) * grain
		hi := min(lo+step, n)
		if next.CompareAndSwap(int64(lo), int64(hi)) {
			fn(lo, hi)
		}
	}
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
