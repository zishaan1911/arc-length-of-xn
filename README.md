# How long is the graph of xⁿ?

**Logarithmic corner defects in arc length.** Paper: [paper.pdf](paper.pdf)

The graph of y = xⁿ on [0,1] has arc length L(n) → 2 as n → ∞. This paper determines how fast:

- 2 − L(n) = (ln n + 1 − ln 2)/(n − 1) + O(ln²n / n²)
- L(n) = 1 + (2/(en))^{1/(n−1)} + π²/(12n²) + O(ln n / n³)
- an exact, rapidly convergent Gamma-function formula for L(n), obtained from a single Beta integral
- corners of any turning angle θ: the coefficient of (ln n)/n is 2 sin²(θ/2)
- by contrast, the squircle |x|ᵖ + |y|ᵖ = 1 has perimeter 8 − 4K/p + O(1/p²) with
  K = 2√2 ln(1+√2) − 2 ln 2, with no logarithm

## Files

| File | Contents |
| --- | --- |
| `paper.tex` | LaTeX source |
| `paper.pdf` | Compiled paper |
| `verify.py` | Recomputes every numerical claim and table in the paper |

## Reproducing

```bash
pip install mpmath
python verify.py
```

To rebuild the PDF with [Tectonic](https://tectonic-typesetting.github.io/):

```bash
tectonic -X compile paper.tex
```
