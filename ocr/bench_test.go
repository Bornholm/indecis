package ocr

import (
	"os"
	"testing"
)

// BenchmarkReadLines reads the lines of a shard with a recognizer, one at
// a time: INDECIS_OCR_BENCH_MODEL and INDECIS_OCR_BENCH_DATA name them,
// INDECIS_INT8=1 reads in int8. Run it pinned to one core:
//
//	taskset -c 2 go test ./ocr -run X -bench ReadLines -cpu 1
func BenchmarkReadLines(b *testing.B) {
	dir, data := os.Getenv("INDECIS_OCR_BENCH_MODEL"), os.Getenv("INDECIS_OCR_BENCH_DATA")
	if dir == "" || data == "" {
		b.Skip("INDECIS_OCR_BENCH_MODEL and INDECIS_OCR_BENCH_DATA not set")
	}
	var opts []Option
	if os.Getenv("INDECIS_INT8") == "1" {
		opts = append(opts, WithInt8())
	}
	r, err := Load(dir, opts...)
	if err != nil {
		b.Fatal(err)
	}
	lines, err := LoadLines(data, r.CropOptions())
	if err != nil {
		b.Fatal(err)
	}
	lines = lines[:min(len(lines), 300)]
	px := 0
	for _, l := range lines {
		px += l.W
	}
	b.ResetTimer()
	for range b.N {
		for _, l := range lines {
			r.ReadCrop(Crop{Ink: l.Ink, W: l.W})
		}
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(px)*800/1e6, "ms/800px")
}
