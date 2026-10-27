package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"unicode"

	"github.com/bornholm/indecis/ocr"
)

// corpus holds paragraphs of running text, one per line of its files.
type corpus struct {
	paras []string
}

func loadCorpus(paths []string, maxParas int) (*corpus, error) {
	c := &corpus{}
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			return nil, err
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<24)
		n := 0
		for sc.Scan() && (maxParas <= 0 || n < maxParas) {
			t := strings.TrimSpace(sc.Text())
			if len(t) >= 20 {
				c.paras = append(c.paras, t)
				n++
			}
		}
		f.Close()
		if err := sc.Err(); err != nil {
			return nil, err
		}
	}
	if len(c.paras) == 0 {
		return nil, fmt.Errorf("no text in %v", paths)
	}
	return c, nil
}

// targetLen draws a line length in characters: short labels, ordinary
// lines and long lines of small print.
func targetLen(r *rand.Rand) int {
	switch x := r.Float64(); {
	case x < 0.25:
		return 2 + r.Intn(14)
	case x < 0.8:
		return 16 + r.Intn(50)
	default:
		return 66 + r.Intn(50)
	}
}

// excerpt cuts a run of whole words of about n characters from a random
// paragraph.
func (c *corpus) excerpt(r *rand.Rand, n int) string {
	words := strings.Fields(c.paras[r.Intn(len(c.paras))])
	start := r.Intn(len(words))
	var b strings.Builder
	for i := start; i < len(words) && b.Len() < n; i++ {
		if b.Len() > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(words[i])
	}
	return b.String()
}

var (
	firstNames = []string{"Marie", "Jean", "Sophie", "Pierre", "Camille", "Nicolas", "Léa", "Thomas", "Chloé", "Émile", "Anaïs", "François", "John", "Sarah", "Michael", "Emma"}
	lastNames  = []string{"Martin", "Bernard", "Dubois", "Lefèvre", "Moreau", "Girard", "Rousseau", "Fontaine", "Smith", "Johnson", "Brown", "Lambert", "Bonnet", "Mercier"}
	streets    = []string{"rue de la République", "avenue Victor-Hugo", "boulevard Saint-Michel", "place de l'Église", "chemin des Écoliers", "allée des Tilleuls", "Main Street", "impasse du Moulin"}
	cities     = []string{"Paris", "Lyon", "Marseille", "Toulouse", "Nantes", "Strasbourg", "Lille", "Rennes", "Dijon", "Besançon", "Saint-Étienne", "London", "Bruxelles", "Genève"}
	months     = []string{"janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"}
	items      = []string{"NASI LEMAK", "TEH TARIK", "MINERAL WATER 500ML", "BREAD WHITE", "CHICKEN RICE", "PAPER A4 80G", "BATTERY AA 4PCS", "PAIN AU CHOCOLAT", "BAGUETTE TRADITION", "CAFE CREME", "EAU MINERALE 1,5L", "SANDWICH JAMBON", "COCA COLA 33CL", "MILK 1L", "RICE 5KG", "SUGAR 1KG", "TISSUE BOX"}
	labels     = []string{"Total TTC", "Montant HT", "TVA 20 %", "Référence", "Date", "Objet", "N° client", "Échéance", "Invoice", "Amount due", "Page", "Tél.", "Fax", "Siret", "IBAN", "Réf. dossier"}
	domains    = []string{"example.fr", "mairie-dijon.fr", "gouv.fr", "entreprise.com", "laposte.net", "univ-lyon.fr"}
)

func pick[T any](r *rand.Rand, xs []T) T { return xs[r.Intn(len(xs))] }

func digits(r *rand.Rand, n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte('0' + r.Intn(10))
	}
	return string(b)
}

// generated returns a line of the kinds running text lacks: amounts,
// dates, references, addresses, contacts, table rows.
func generated(r *rand.Rand, charset []rune) string {
	amount := func() string {
		n := r.Intn(100000)
		s := fmt.Sprintf("%d", n)
		if n >= 1000 && r.Intn(2) == 0 {
			s = fmt.Sprintf("%d %03d", n/1000, n%1000)
		}
		return s + "," + digits(r, 2)
	}
	if r.Float64() < 0.2 {
		return receipt(r, amount)
	}
	switch r.Intn(14) {
	case 0:
		return pick(r, labels) + " : " + amount() + " " + pick(r, []string{"€", "EUR", "$", "£", ""})
	case 1:
		return fmt.Sprintf("%02d/%02d/%d", 1+r.Intn(28), 1+r.Intn(12), 1950+r.Intn(80))
	case 2:
		return fmt.Sprintf("le %d %s %d", 1+r.Intn(28), pick(r, months), 1900+r.Intn(130))
	case 3:
		return fmt.Sprintf("%d %s, %s %s", 1+r.Intn(150), pick(r, streets), digits(r, 5), strings.ToUpper(pick(r, cities)))
	case 4:
		return fmt.Sprintf("Tél. : 0%d %s %s %s %s", 1+r.Intn(9), digits(r, 2), digits(r, 2), digits(r, 2), digits(r, 2))
	case 5:
		f, l := pick(r, firstNames), pick(r, lastNames)
		return strings.ToLower(f+"."+l) + "@" + pick(r, domains)
	case 6:
		return "https://www." + pick(r, domains) + "/" + strings.ToLower(pick(r, lastNames)) + "/" + digits(r, 4)
	case 7:
		return fmt.Sprintf("Réf. %s-%s/%s", strings.ToUpper(string(rune('A'+r.Intn(26)))+string(rune('A'+r.Intn(26)))), digits(r, 4), digits(r, 2))
	case 8:
		return pick(r, []string{"M.", "Mme", "Dr", "Me"}) + " " + pick(r, firstNames) + " " + strings.ToUpper(pick(r, lastNames))
	case 9:
		return fmt.Sprintf("%s %s  %d  %s  %s", pick(r, []string{"•", "-", "–", "*", fmt.Sprintf("%d.", 1+r.Intn(20))}), pick(r, labels), 1+r.Intn(99), amount(), amount())
	case 10:
		return "FR76 " + digits(r, 4) + " " + digits(r, 4) + " " + digits(r, 4) + " " + digits(r, 4) + " " + digits(r, 3)
	case 11:
		return fmt.Sprintf("Article %d — %s", 1+r.Intn(300), pick(r, []string{"Dispositions générales", "Durée", "Résiliation", "Prix et paiement", "Objet du contrat"}))
	case 12:
		return fmt.Sprintf("%d,%s %% ; %d° ; %s × %s = %s", r.Intn(100), digits(r, 1), r.Intn(360), digits(r, 2), digits(r, 2), digits(r, 4))
	default:
		// Random characters of the charset: rare symbols get seen too.
		n := 3 + r.Intn(20)
		b := make([]rune, 0, n)
		for len(b) < n {
			c := charset[r.Intn(len(charset))]
			if c == ' ' && (len(b) == 0 || b[len(b)-1] == ' ') {
				continue
			}
			b = append(b, c)
		}
		return strings.TrimSpace(string(b))
	}
}

// receipt returns a line of a till receipt.
func receipt(r *rand.Rand, amount func() string) string {
	price := func() string { return fmt.Sprintf("%d.%s", r.Intn(200), digits(r, 2)) }
	cur := pick(r, []string{"RM", "$", "€", "EUR", ""})
	switch r.Intn(10) {
	case 0, 1, 2:
		return fmt.Sprintf("%s %d %s %s", pick(r, items), 1+r.Intn(5), price(), price())
	case 3:
		return fmt.Sprintf("%s %s %s", pick(r, []string{"TOTAL", "SUBTOTAL", "TOTAL (RM)", "NETT TOTAL", "ROUNDING", "CASH", "CHANGE", "TOTAL TTC", "TOTAL HT", "GST 6%", "TVA 20%", "DISCOUNT"}), cur, price())
	case 4:
		return fmt.Sprintf("%s: %s%s", pick(r, []string{"INVOICE NO", "RECEIPT #", "BILL NO", "TICKET N°", "CASHIER", "TERMINAL"}), pick(r, []string{"CS", "INV", "T", ""}), digits(r, 4+r.Intn(6)))
	case 5:
		return fmt.Sprintf("%02d/%02d/%d %02d:%02d:%02d", 1+r.Intn(28), 1+r.Intn(12), 2010+r.Intn(16), r.Intn(24), r.Intn(60), r.Intn(60))
	case 6:
		return pick(r, []string{"THANK YOU", "THANK YOU. PLEASE COME AGAIN.", "GOODS SOLD ARE NOT RETURNABLE", "*** COPY ***", "TAX INVOICE", "MERCI DE VOTRE VISITE", "E.&O.E.", "GST REG NO: "}) + strings.Repeat(digits(r, 1), r.Intn(2))
	case 7:
		return fmt.Sprintf("%s %s @ %s", digits(r, 13), pick(r, items), price())
	case 8:
		return fmt.Sprintf("%s SDN BHD (%s-%s)", strings.ToUpper(pick(r, lastNames)), digits(r, 6), string(rune('A'+r.Intn(26))))
	default:
		return fmt.Sprintf("QTY: %d  %s %s", 1+r.Intn(20), cur, amount())
	}
}

// typography gives the straight quotes of a text their typographic form,
// or the reverse: the corpora hardly hold “ ” and ‘, documents do.
func typography(r *rand.Rand, s string) string {
	switch x := r.Float64(); {
	case x < 0.35:
		rs := []rune(s)
		open := true
		for i, c := range rs {
			letterBefore := i > 0 && unicode.IsLetter(rs[i-1])
			switch c {
			case '\'':
				if letterBefore {
					rs[i] = '’'
				} else {
					rs[i] = '‘'
				}
			case '"':
				if open {
					rs[i] = '“'
				} else {
					rs[i] = '”'
				}
				open = !open
			}
		}
		return string(rs)
	case x < 0.45:
		return strings.NewReplacer("’", "'", "‘", "'", "“", "\"", "”", "\"").Replace(s)
	}
	return s
}

// texts draws the lines of a patch: the line to read, and the neighbours
// above and below from the same source.
type texter struct {
	corpora []*corpus
	charset []rune
	enc     *ocr.Encoder
}

func (t *texter) line(r *rand.Rand) string {
	for {
		var s string
		if r.Float64() < 0.3 {
			s = generated(r, t.charset)
		} else {
			s = pick(r, t.corpora).excerpt(r, targetLen(r))
		}
		s = typography(r, s)
		switch x := r.Float64(); {
		case x < 0.08:
			s = strings.ToUpper(s)
		case x < 0.12:
			s = titleCase(s)
		}
		s = ocr.NormalizeText(s)
		if s == "" {
			continue
		}
		if _, _, ok := t.enc.Encode(s); ok {
			return s
		}
	}
}

func titleCase(s string) string {
	rs := []rune(s)
	for i := range rs {
		if i == 0 || rs[i-1] == ' ' {
			rs[i] = unicode.ToUpper(rs[i])
		}
	}
	return string(rs)
}
