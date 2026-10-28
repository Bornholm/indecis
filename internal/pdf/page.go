package pdf

// A Page is one page of a file, with the attributes it inherits.
type Page struct {
	f         *File
	dict      Dict
	resources Dict
	box       [4]float64 // the crop box (llx, lly, urx, ury), within the media box
	rotate    int        // clockwise quarter turns of the page when shown
}

// Pages returns the pages of the file in order.
func (f *File) Pages() []*Page {
	var pages []*Page
	seen := map[Ref]bool{}
	var walk func(node any, res Dict, media, crop []float64, rot int64, depth int)
	walk = func(node any, res Dict, media, crop []float64, rot int64, depth int) {
		if r, ok := node.(Ref); ok {
			if seen[r] {
				return
			}
			seen[r] = true
		}
		d := f.dict(node)
		if d == nil || depth > 64 {
			return
		}
		if r := f.dict(d["Resources"]); r != nil {
			res = r
		}
		if b := f.rect(d["MediaBox"]); b != nil {
			media = b
		}
		if b := f.rect(d["CropBox"]); b != nil {
			crop = b
		}
		if v, ok := f.Resolve(d["Rotate"]).(int64); ok {
			rot = v
		}
		kids := f.array(d["Kids"])
		if kids == nil || f.name(d["Type"]) == "Page" {
			p := &Page{f: f, dict: d, resources: res, rotate: int(((rot/90)%4 + 4) % 4)}
			if media == nil {
				media = []float64{0, 0, 612, 792}
			}
			box := media
			if crop != nil {
				// The crop box, clipped to the media box.
				box = []float64{max(crop[0], media[0]), max(crop[1], media[1]), min(crop[2], media[2]), min(crop[3], media[3])}
				if box[2] <= box[0] || box[3] <= box[1] {
					box = media
				}
			}
			copy(p.box[:], box)
			pages = append(pages, p)
			return
		}
		for _, k := range kids {
			walk(k, res, media, crop, rot, depth+1)
		}
	}
	root := f.dict(f.trailer["Root"])
	walk(root["Pages"], nil, nil, nil, 0, 0)
	return pages
}

// rect reads a rectangle, its corners in order.
func (f *File) rect(o any) []float64 {
	a := f.array(o)
	if len(a) != 4 {
		return nil
	}
	r := make([]float64, 4)
	for i, v := range a {
		x, ok := f.num(v)
		if !ok {
			return nil
		}
		r[i] = x
	}
	if r[0] > r[2] {
		r[0], r[2] = r[2], r[0]
	}
	if r[1] > r[3] {
		r[1], r[3] = r[3], r[1]
	}
	if r[2]-r[0] < 1 || r[3]-r[1] < 1 {
		return nil
	}
	return r
}

// Size is the width and height of the page as shown, in points: its crop
// box, turned by its rotation.
func (p *Page) Size() (w, h float64) {
	w, h = p.box[2]-p.box[0], p.box[3]-p.box[1]
	if p.rotate%2 == 1 {
		w, h = h, w
	}
	return w, h
}

// Rotation is the page's clockwise quarter turns when shown (its /Rotate).
func (p *Page) Rotation() int { return p.rotate }

// toView maps a point of the page's default user space to the page as
// shown: points, origin top left, y down.
func (p *Page) toView(x, y float64) (float64, float64) {
	x -= p.box[0]
	y = p.box[3] - y
	w, h := p.box[2]-p.box[0], p.box[3]-p.box[1]
	switch p.rotate {
	case 1:
		return h - y, x
	case 2:
		return w - x, h - y
	case 3:
		return y, w - x
	}
	return x, y
}

func (p *Page) contents() []byte {
	var out []byte
	add := func(o any) {
		if s, ok := p.f.Resolve(o).(*Stream); ok {
			if data, err := p.f.decode(s); err == nil || len(data) > 0 {
				out = append(out, data...)
				out = append(out, '\n')
			}
		}
	}
	switch v := p.f.Resolve(p.dict["Contents"]).(type) {
	case Array:
		for _, o := range v {
			add(o)
		}
	case *Stream:
		add(v)
	}
	return out
}
