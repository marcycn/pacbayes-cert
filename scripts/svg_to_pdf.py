#!/usr/bin/env python3
"""Convert the hand-drawn data-partition figure images/m1.svg to fig_partition.pdf.

The SVG (authored in an office tool) uses Unicode Mathematical Alphanumeric letters
(e.g. U+1D446 MATHEMATICAL ITALIC S) in a "Cambria Math" font that cairosvg cannot
resolve, which otherwise renders every symbol as a missing-glyph box. This script
preprocesses the SVG so cairosvg produces a faithful PDF:

  1. NFKC-normalise text so math-italic letters become ASCII (S, P, Q, R, n, ...);
  2. route the now-ASCII letter runs to Georgia (renders an italic serif);
  3. route set-symbol runs (cap/cup/in/setminus) to DejaVu Sans, which has them;
  4. replace the empty-set glyph with the universally available stroked-O (Ø).

Run:  python scripts/svg_to_pdf.py
Out:  paper/muthesis/images/fig_partition.pdf
"""
from __future__ import annotations

import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG = os.path.join(ROOT, "paper", "muthesis", "images")
SRC = os.path.join(IMG, "m1.svg")
FIXED = os.path.join(IMG, "m1_fixed.svg")
OUT = os.path.join(IMG, "fig_partition.pdf")

_SET_SYMS = set("∪∅∈∩∖")


def main():
    import cairosvg

    s = open(SRC, encoding="utf-8").read()
    s = unicodedata.normalize("NFKC", s)                       # math-italic -> ASCII
    for k in ("Cambria Math,Cambria Math_MSFontService,sans-serif",
              "Cambria Math_MSFontService", "Cambria Math"):
        s = s.replace(k, "Georgia, serif")

    def route(m):                                              # set symbols -> DejaVu Sans
        tag = m.group(0)
        if any(ch in tag for ch in _SET_SYMS):
            tag = re.sub(r'font-family="[^"]*"', 'font-family="DejaVu Sans"', tag, count=1)
        return tag

    s = re.sub(r"<text\b[^>]*>.*?</text>", route, s, flags=re.S)
    s = s.replace("∅", "Ø")                                    # emptyset glyph

    open(FIXED, "w", encoding="utf-8").write(s)
    cairosvg.svg2pdf(url=FIXED, write_to=OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
