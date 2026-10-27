"""The characters of an extended recognizer: a model's vocabulary, in its
order (its symbols keep their ids, see indecis ocr train -extend), then the
letters of the Latin, Cyrillic and Greek scripts seen at least -min times
in the given texts, then a fixed list of letters and signs the texts
rarely hold (Vietnamese, Romanian and West African letters, circled
numbers, logic). Writes them on one line, for -charset.

    python3 tools/ocrsynth/extend_charset.py vocab.json charset.txt [min] texts...
"""

import json
import sys
import unicodedata
from collections import Counter

FIXED = (
    # Vietnamese: every precomposed vowel with its tone marks.
    "".join(chr(c) for c in range(0x1EA0, 0x1EFA))
    + "ĂăÂâĐđÊêÔôƠơƯư"
    # Romanian (comma below), Icelandic and Faroese, Baltic, Maltese.
    + "ȘșȚțŞşŢţÐðÞþÝýĀāĒēĪīŌōŪūĖėĮįŲųĢģĶķĻļŅņĊċĠġĦħŻż"
    # West African letters (Hausa, Yoruba, Igbo, Ewe, Wolof).
    + "ƁɓƊɗƘƙƳƴŊŋƐɛƆɔƏəƎǝƗɨƲʋƝɲẸẹỌọṢṣỊịỤụṄṅ"
    # The basic Cyrillic and Greek alphabets, whole.
    + "".join(chr(c) for c in range(0x0410, 0x0450)) + "Ёё"
    + "".join(chr(c) for c in range(0x0391, 0x03AA) if c != 0x03A2)
    + "".join(chr(c) for c in range(0x03B1, 0x03CA))
    # Signs met in documents.
    + "".join(chr(c) for c in range(0x2460, 0x2474))  # ① to ⑳
    + "➢➤❖✦★☆✧◦‣⁃∙"
    + "¬∧∨∀∃∈∉⊂⊆∪∩≡≅∓∫∏∇"
    + "©"
)

SCRIPTS = ("LATIN", "CYRILLIC", "GREEK")


def main():
    vocab, out = sys.argv[1], sys.argv[2]
    minimum = int(sys.argv[3])
    v = json.load(open(vocab, encoding="utf-8"))
    chars = list(v["chars"] if isinstance(v, dict) else v)
    have = set(chars)
    seen = Counter()
    for path in sys.argv[4:]:
        with open(path, encoding="utf-8") as f:
            for line in f:
                seen.update(unicodedata.normalize("NFC", line))
    added = []
    for c, n in seen.most_common():
        if n < minimum or c in have or len(c) != 1:
            continue
        name = unicodedata.name(c, "")
        if name.startswith(SCRIPTS) and unicodedata.category(c).startswith("L"):
            added.append(c)
            have.add(c)
    for c in FIXED:
        if c not in have:
            added.append(c)
            have.add(c)
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(chars + added))
    print(f"{len(chars)} + {len(added)} = {len(chars) + len(added)} characters", file=sys.stderr)
    print("".join(added), file=sys.stderr)


if __name__ == "__main__":
    main()
