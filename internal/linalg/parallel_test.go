package linalg

import (
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// The slices of ParallelN cover [0, n) exactly once, each at least grain
// long and a multiple of it, except the last.
func TestParallelSlices(t *testing.T) {
	for _, limit := range []int{1, 2, 3, 8, 64} {
		for _, grain := range []int{0, 1, 6, 24} {
			for _, n := range []int{0, 1, 5, 24, 25, 97, 256, 1000} {
				seen := make([]atomic.Int32, n)
				var bad atomic.Bool
				ParallelN(limit, n, grain, func(lo, hi int) {
					g := max(grain, 1)
					if lo >= hi || ((hi-lo)%g != 0 && hi != n) {
						bad.Store(true)
					}
					for i := lo; i < hi; i++ {
						seen[i].Add(1)
					}
				})
				if bad.Load() {
					t.Fatalf("limit %d grain %d n %d: a slice is empty or not a multiple of the grain", limit, grain, n)
				}
				for i := range seen {
					if c := seen[i].Load(); c != 1 {
						t.Fatalf("limit %d grain %d n %d: element %d computed %d times", limit, grain, n, i, c)
					}
				}
			}
		}
	}
}

// A panic in fn on the calling goroutine waits for the other workers.
func TestParallelPanic(t *testing.T) {
	if Workers() < 2 {
		t.Skip("one core")
	}
	var busy atomic.Int32
	release := make(chan struct{})
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("no panic")
			}
			if n := busy.Load(); n != 0 {
				t.Fatalf("%d workers still running after the panic", n)
			}
		}()
		caller := goid()
		ParallelN(4, 1000, 1, func(lo, hi int) {
			if goid() == caller {
				// Panic while a worker is inside fn, and let it go on only
				// later: without the wait, the panic would get through
				// before it is done.
				for busy.Load() == 0 {
					runtime.Gosched()
				}
				time.AfterFunc(20*time.Millisecond, func() { close(release) })
				panic("fn")
			}
			busy.Add(1)
			<-release
			busy.Add(-1)
		})
	}()
}

// goid is the number of the current goroutine, read from its stack trace.
func goid() string {
	b := make([]byte, 64)
	b = b[:runtime.Stack(b, false)]
	return strings.Fields(string(b))[1]
}
