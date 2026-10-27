package ocr

import "testing"

func TestReadingOrder(t *testing.T) {
	line := func(x0, y float64, w float64) Box { return Box{X0: x0, Y0: y, X1: x0 + w, Y1: y + 10} }
	// A title across the page, two columns of three lines, a footer.
	boxes := []Box{
		line(320, 220, 260), // 0: right column, line 1
		line(10, 10, 570),   // 1: title
		line(10, 220, 280),  // 2: left column, line 1
		line(10, 236, 280),  // 3: left column, line 2
		line(320, 236, 260), // 4: right column, line 2
		line(10, 252, 200),  // 5: left column, line 3
		line(320, 252, 150), // 6: right column, line 3
		line(250, 500, 80),  // 7: footer
	}
	got := ReadingOrder(boxes)
	want := []int{1, 2, 3, 5, 0, 4, 6, 7}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("order %v, want %v", got, want)
		}
	}
	if len(ReadingOrder(nil)) != 0 {
		t.Fatal("empty page")
	}
}
