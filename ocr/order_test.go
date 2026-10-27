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

func TestReadingOrderNarrowGutter(t *testing.T) {
	line := func(x0, y float64, w float64) Box { return Box{X0: x0, Y0: y, X1: x0 + w, Y1: y + 10} }
	// Two columns of text 4 pixels apart, as old prints set them: read one
	// after the other.
	var boxes []Box
	for i := range 5 {
		boxes = append(boxes, line(0, float64(12*i), 200), line(204, float64(12*i), 200))
	}
	got := ReadingOrder(boxes)
	for i, j := range got {
		if want := 2*(i%5) + i/5; j != want {
			t.Fatalf("columns: order %v", got)
		}
	}
	// Short cells as far apart stay rows: a label, then its amount.
	boxes = boxes[:0]
	for i := range 5 {
		boxes = append(boxes, line(0, float64(12*i), 40), line(44, float64(12*i), 30))
	}
	got = ReadingOrder(boxes)
	for i, j := range got {
		if j != i {
			t.Fatalf("rows: order %v", got)
		}
	}
}

func TestReadingOrderTable(t *testing.T) {
	line := func(x0, y float64, w float64) Box { return Box{X0: x0, Y0: y, X1: x0 + w, Y1: y + 10} }
	// A receipt: labels left, amounts far right, read row by row.
	boxes := []Box{line(0, 0, 70), line(300, 0, 40), line(0, 20, 30), line(300, 20, 40), line(0, 40, 40), line(300, 40, 40)}
	got := ReadingOrder(boxes)
	for i, j := range got {
		if j != i {
			t.Fatalf("order %v", got)
		}
	}
}
