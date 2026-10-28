# Font licences

Every font file in this folder is self-hosted and loaded by relative URL. One row per file or file set.
`lib/theme.css` declares `@font-face` for the four woff2 files by relative URL `../fonts/`; the page
links `katex/katex.min.css`, and the KaTeX script loads from cdn.jsdelivr.net/npm/katex@0.16.22.

| File | Family | Designer | Licence | Source | Build |
|---|---|---|---|---|---|
| SourceSerif4-Roman-opsz-wght.woff2 | Source Serif 4 (roman, variable opsz 8–60) | Frank Grießhammer / Adobe | SIL OFL 1.1 (OFL-sourceserif4.txt) | github.com/google/fonts/tree/main/ofl/sourceserif4 `SourceSerif4[opsz,wght].ttf` | `varLib.instancer wght=300:650`; `pyftsubset --layout-features='*'` Latin-1, General Punctuation, super/subscripts, arrows, U+2212, math operators, Greek, Misc Technical, Geometric Shapes |
| SourceSerif4-Italic-opsz-wght.woff2 | Source Serif 4 Italic | Frank Grießhammer / Adobe | SIL OFL 1.1 (OFL-sourceserif4.txt) | same dir, `SourceSerif4-Italic[opsz,wght].ttf` | `wght=400:600`, same subset |
| Inter-opsz-wght.woff2 | Inter (variable opsz 14–32) | Rasmus Andersson | SIL OFL 1.1 (OFL-inter.txt) | github.com/google/fonts/tree/main/ofl/inter `Inter[opsz,wght].ttf` | `wght=250:650`, same subset |
| JetBrainsMono-wght.woff2 | JetBrains Mono | JetBrains (Philipp Nurullin, Konstantin Bulenkov) | SIL OFL 1.1 (OFL-jetbrainsmono.txt) | github.com/google/fonts/tree/main/ofl/jetbrainsmono `JetBrainsMono[wght].ttf` | `wght=300:600`, same subset |
| katex/katex.min.css + katex/fonts/KaTeX_{AMS-Regular, Main-Regular, Math-Italic, Size1-Regular, Size2-Regular}.woff2 | KaTeX 0.16.22 stylesheet and the five fonts the page's TeX uses | Khan Academy and contributors | MIT (katex/LICENSE) | npm `katex@0.16.22` `dist/` | unchanged files; the other 15 KaTeX woff2 fonts are omitted, because no formula on the page requests them. The script loads from cdn.jsdelivr.net/npm/katex@0.16.22. The stylesheet also names .woff/.ttf fallbacks and the omitted fonts; none of these is shipped, and every current browser takes the .woff2 of a font it uses. |
