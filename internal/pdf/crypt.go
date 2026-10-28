package pdf

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/md5"
	"crypto/rc4"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/binary"
	"hash"
)

// crypt decrypts a file of the standard security handler opened with the
// empty user password.
type crypt struct {
	key       []byte
	stm, str  string // method of streams and strings: "", "RC4", "AESV2", "AESV3"
	encDict   Ref
	keyLength int
}

var padding = []byte{
	0x28, 0xBF, 0x4E, 0x5E, 0x4E, 0x75, 0x8A, 0x41, 0x64, 0x00, 0x4E, 0x56, 0xFF, 0xFA, 0x01, 0x08,
	0x2E, 0x2E, 0x00, 0xB6, 0xD0, 0x68, 0x3E, 0x80, 0x2F, 0x0C, 0xA9, 0xFE, 0x64, 0x53, 0x69, 0x7A,
}

func newCrypt(f *File, enc any) (*crypt, error) {
	ref, _ := enc.(Ref)
	d := f.dict(enc)
	if d == nil || f.name(d["Filter"]) != "Standard" {
		return nil, errEncrypted
	}
	v, _ := f.Resolve(d["V"]).(int64)
	r, _ := f.Resolve(d["R"]).(int64)
	o, _ := f.Resolve(d["O"]).(String)
	u, _ := f.Resolve(d["U"]).(String)
	p, _ := f.Resolve(d["P"]).(int64)
	c := &crypt{encDict: ref, stm: "RC4", str: "RC4"}
	length := int64(40)
	if l, ok := f.Resolve(d["Length"]).(int64); ok && l >= 40 {
		length = l
	}
	if v >= 4 {
		cfs := f.dict(d["CF"])
		method := func(n Name) string {
			if n == "Identity" || n == "" {
				return ""
			}
			cf := f.dict(cfs[n])
			switch f.name(cf["CFM"]) {
			case "AESV2":
				return "AESV2"
			case "AESV3":
				return "AESV3"
			case "V2":
				if l, ok := f.Resolve(cf["Length"]).(int64); ok && l > 0 {
					if l <= 32 { // bytes, some producers write
						l *= 8
					}
					length = l
				}
				return "RC4"
			}
			return ""
		}
		c.stm = method(f.name(d["StmF"]))
		c.str = method(f.name(d["StrF"]))
	}
	if r >= 5 {
		ue, _ := f.Resolve(d["UE"]).(String)
		if len(u) < 48 || len(ue) < 32 {
			return nil, errEncrypted
		}
		var k []byte
		if r == 5 {
			h := sha256.Sum256([]byte(u[40:48]))
			k = h[:]
		} else {
			k = hash2B(nil, []byte(u[40:48]), nil)
		}
		// The check of the password: the hash with the validation salt.
		var check []byte
		if r == 5 {
			h := sha256.Sum256([]byte(u[32:40]))
			check = h[:]
		} else {
			check = hash2B(nil, []byte(u[32:40]), nil)
		}
		if !bytes.Equal(check, []byte(u[:32])) {
			return nil, errEncrypted
		}
		b, _ := aes.NewCipher(k)
		key := make([]byte, 32)
		cipher.NewCBCDecrypter(b, make([]byte, 16)).CryptBlocks(key, []byte(ue[:32]))
		c.key = key
		return c, nil
	}
	n := int(length / 8)
	if n < 5 || n > 16 {
		n = 5
	}
	c.keyLength = n
	var id []byte
	if ids := f.array(f.trailer["ID"]); len(ids) > 0 {
		s, _ := f.Resolve(ids[0]).(String)
		id = []byte(s)
	}
	h := md5.New()
	h.Write(padding)
	h.Write([]byte(o))
	var pb [4]byte
	binary.LittleEndian.PutUint32(pb[:], uint32(int32(p)))
	h.Write(pb[:])
	h.Write(id)
	if r >= 4 {
		if em, ok := f.Resolve(d["EncryptMetadata"]).(bool); ok && !em {
			h.Write([]byte{0xff, 0xff, 0xff, 0xff})
		}
	}
	key := h.Sum(nil)
	if r >= 3 {
		for range 50 {
			s := md5.Sum(key[:n])
			key = s[:]
		}
	}
	c.key = key[:n]
	// The user password check.
	var want []byte
	if r == 2 {
		want = make([]byte, 32)
		ci, _ := rc4.NewCipher(c.key)
		ci.XORKeyStream(want, padding)
		if len(u) >= 32 && !bytes.Equal(want, []byte(u[:32])) {
			return nil, errEncrypted
		}
	} else {
		h := md5.New()
		h.Write(padding)
		h.Write(id)
		want = h.Sum(nil)
		for i := range 20 {
			k := make([]byte, len(c.key))
			for j := range k {
				k[j] = c.key[j] ^ byte(i)
			}
			ci, _ := rc4.NewCipher(k)
			ci.XORKeyStream(want, want)
		}
		if len(u) >= 16 && !bytes.Equal(want, []byte(u[:16])) {
			return nil, errEncrypted
		}
	}
	return c, nil
}

// hash2B is the hash of ISO 32000-2 (7.6.4.3.4) for revision 6.
func hash2B(pw, salt, udata []byte) []byte {
	h := sha256.New()
	h.Write(pw)
	h.Write(salt)
	h.Write(udata)
	k := h.Sum(nil)
	for i := 0; ; i++ {
		var k1 []byte
		for range 64 {
			k1 = append(k1, pw...)
			k1 = append(k1, k...)
			k1 = append(k1, udata...)
		}
		b, _ := aes.NewCipher(k[:16])
		e := make([]byte, len(k1))
		cipher.NewCBCEncrypter(b, k[16:32]).CryptBlocks(e, k1)
		sum := 0
		for _, v := range e[:16] {
			sum += int(v)
		}
		var hh hash.Hash
		switch sum % 3 {
		case 0:
			hh = sha256.New()
		case 1:
			hh = sha512.New384()
		default:
			hh = sha512.New()
		}
		hh.Write(e)
		k = hh.Sum(nil)
		if i >= 63 && int(e[len(e)-1]) <= i-31 {
			break
		}
	}
	return k[:32]
}

func (c *crypt) objectKey(ref Ref, aes bool) []byte {
	if len(c.key) == 32 { // AESV3: one key for all
		return c.key
	}
	h := md5.New()
	h.Write(c.key)
	h.Write([]byte{byte(ref.Num), byte(ref.Num >> 8), byte(ref.Num >> 16), byte(ref.Gen), byte(ref.Gen >> 8)})
	if aes {
		h.Write([]byte("sAlT"))
	}
	k := h.Sum(nil)
	return k[:min(len(c.key)+5, 16)]
}

func (c *crypt) decrypt(method string, ref Ref, data []byte) []byte {
	switch method {
	case "RC4":
		out := make([]byte, len(data))
		ci, _ := rc4.NewCipher(c.objectKey(ref, false))
		ci.XORKeyStream(out, data)
		return out
	case "AESV2", "AESV3":
		if len(data) < 32 || len(data)%16 != 0 {
			return data
		}
		b, err := aes.NewCipher(c.objectKey(ref, true))
		if err != nil {
			return data
		}
		out := make([]byte, len(data)-16)
		cipher.NewCBCDecrypter(b, data[:16]).CryptBlocks(out, data[16:])
		if n := int(out[len(out)-1]); n >= 1 && n <= 16 {
			out = out[:len(out)-n]
		}
		return out
	}
	return data
}

func (c *crypt) decryptStream(s *Stream, data []byte) []byte {
	if s.ref == c.encDict {
		return data
	}
	return c.decrypt(c.stm, s.ref, data)
}

func (c *crypt) decryptObject(o any, ref Ref) any {
	if ref == c.encDict {
		return o
	}
	switch v := o.(type) {
	case String:
		return String(c.decrypt(c.str, ref, []byte(v)))
	case Array:
		for i := range v {
			v[i] = c.decryptObject(v[i], ref)
		}
	case Dict:
		c.decryptStrings(v, ref)
	}
	return o
}

func (c *crypt) decryptStrings(d Dict, ref Ref) {
	for k, v := range d {
		d[k] = c.decryptObject(v, ref)
	}
}
