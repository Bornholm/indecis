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
// several times by the same goroutine, the calling one included, so fn
// must not wait on the caller. If fn panics on the calling goroutine, the
// panic waits for the other goroutines to leave fn: the caller's buffers
// are not reused while they write. A panic on another goroutine stops the
// process, as in any goroutine.
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
	// About four slices per worker, taken as they come: on a hybrid
	// processor, a slow core (E, LP-E) takes fewer slices instead of holding
	// everyone up with an equal share. The calling goroutine takes slices
	// too. The slices are disjoint, so the result does not depend on who
	// computes which. Each slice is a multiple of grain, so of the int8
	// kernel's rows when grain is, except the last. Slices shrinking
	// towards the end were measured no faster at inference (see
	// modernbert's sliceRows), and they multiply the calls of fn for a
	// grain of 1.
	step := ((n+4*workers-1)/(4*workers) + grain - 1) / grain * grain
	workers = min(workers, (n+step-1)/step)
	var next atomic.Int64
	var wg sync.WaitGroup
	defer wg.Wait()
	wg.Add(workers - 1)
	for w := 1; w < workers; w++ {
		go func() {
			defer wg.Done()
			slices(&next, fn, n, step)
		}()
	}
	slices(&next, fn, n, step)
}

// slices takes slices of step elements of [0, n) from next until none
// remain.
func slices(next *atomic.Int64, fn func(lo, hi int), n, step int) {
	for {
		lo := int(next.Add(1)-1) * step
		if lo >= n {
			return
		}
		fn(lo, min(lo+step, n))
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
