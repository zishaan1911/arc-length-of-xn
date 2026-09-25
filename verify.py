"""Numerical verification for "How Long Is the Graph of x^n? Logarithmic Corner Defects in Arc Length".

Recomputes every numerical claim in the paper with mpmath (pip install mpmath) and checks it.
Labels (Lemma 2, Table III, ...) follow the numbering in paper.pdf.
Run:  python verify.py
"""
from mpmath import (mp, mpf, sqrt, log, exp, pi, e, gamma, zeta, binomial, nsum, quad, inf, cos,
                    taylor, nstr, pslq)

mp.dps = 40


def R(eps):
    """R(eps) = Gamma(1+eps/2) Gamma((1-eps)/2) / (sqrt(pi) (1+eps)), eq. (3)."""
    return gamma(1 + eps / 2) * gamma((1 - eps) / 2) / (sqrt(pi) * (1 + eps))


def L_exact(n, c=1):
    """Theorem 1 (c = 1) and Remark 2 (y = c x^n); needs n > 2 and c n > 1."""
    n = mpf(n)
    eps = 1 / (n - 1)
    m = c * n
    tail = nsum(lambda k: binomial(mpf(1) / 2, k) / (m ** (2 * k - 1) * ((2 * k - 1) * (n - 1) - 1)), [1, inf])
    return c + R(eps) / m**eps - tail


def L_quad(n, c=1):
    """Direct quadrature of int_0^1 sqrt(1 + (c n x^(n-1))^2) dx."""
    n = mpf(n)
    h = 1 / n
    pts = sorted({mpf(0), mpf(1)} | {1 - k * h for k in (50, 20, 10, 5, 3, 2, 1, 0.5, 0.2, 0.1, 0.05) if 1 - k * h > 0})
    return quad(lambda x: sqrt(1 + (c * n * x ** (n - 1)) ** 2), pts)


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    assert ok


# Lemma 2: eps * M(eps) = R(eps)
for eps in (mpf("0.37"), mpf("0.8")):
    # k(s) = sqrt(1+s^2) - s, written stably as 1/(s + sqrt(1+s^2)); substitute s = e^u
    M = quad(lambda u: exp(eps * u) / (exp(u) + sqrt(1 + exp(2 * u))), [-inf, -20, 0, 20, 100, 300, inf])
    check(f"Lemma 2 at eps={eps}", abs(eps * M - R(eps)) < mpf(10) ** -15)

# Theorem 1, Remark 2, Example 1, Remark 1
for n in (2.5, 3, 5, 7.5, 10, 100, 1000):
    check(f"Theorem 1 at n={n}", abs(L_exact(n) - L_quad(n)) < mpf(10) ** -25)
for c, n in ((3, 5), (0.5, 7), (0.25, 9.5)):
    check(f"Remark 2 (scaled curves) at c={c}, n={n}", abs(L_exact(n, c) - L_quad(n, c)) < mpf(10) ** -25)
L3 = 1 + gamma(mpf(1) / 4) ** 2 / (6 * sqrt(3 * pi)) - nsum(
    lambda k: binomial(mpf(1) / 2, k) / (3 ** (2 * k - 1) * (4 * k - 3)), [1, inf])
check("Example 1: closed form of L(3)", abs(L3 - quad(lambda x: sqrt(1 + 9 * x**4), [0, 0.5, 1])) < mpf(10) ** -35)
parabola = sqrt(5) / 2 + log(2 + sqrt(5)) / 4
check("Remark 1: parabola length", abs(L_quad(2) - parabola) < mpf(10) ** -25)
# the poles at n = 2 cancel; at n = 2 + 1e-10 the formula is within ~1e-11 of L(2)
check("Remark 1: Theorem 1 tends to L(2) as n -> 2+", abs(L_exact(2 + mpf(10) ** -10) - parabola) < mpf(10) ** -10)

# Lemma 3: Taylor coefficients of ln R
a1, a2, a3 = log(2) - 1, pi**2 / 12 + mpf(1) / 2, zeta(3) / 4 - mpf(1) / 3
co = taylor(lambda x: log(R(x)), 0, 3)
check("Lemma 3: a1", abs(co[1] - a1) < mpf(10) ** -20)
check("Lemma 3: a2", abs(co[2] - a2) < mpf(10) ** -20)
check("Lemma 3: a3", abs(co[3] - a3) < mpf(10) ** -20)

# Proposition 1, Corollary 1, Proposition 2, Table III
d = a2 * (1 + log(2)) + zeta(3) / 4 - mpf(4) / 3
print("\nTable III:  n | L(n) | E_a(n) | E_b(n)")
for n in (3, 5, 10, 20, 50, 100, 1000, 10000, 100000):
    n = mpf(n)
    Ln = L_exact(n)
    D = 2 - Ln
    lo = (log(n) - 1) / (n - 1) - log(n) ** 2 / (2 * (n - 1) ** 2)
    hi = (log(n) + 1) / (n - 1)
    assert lo <= D <= hi, "Proposition 1"
    Ea = Ln - (2 - (log(n) + 1 - log(2)) / (n - 1))
    Eb = Ln - (1 + (2 / (e * n)) ** (1 / (n - 1)) + pi**2 / (12 * n**2))
    E3 = Eb - (d - a2 * log(n)) / n**3
    if n >= 10:
        assert abs(Ea) * n**2 / log(n) ** 2 < 1, "Corollary 1(a) error rate"
        assert abs(Eb) * n**3 / log(n) < 1.5, "Corollary 1(b) error rate"
    if n >= 100:
        assert abs(E3) * n**4 / log(n) ** 2 < 1, "Proposition 2 error rate"
    print(f"  {int(n):>6} | {nstr(Ln, 13)} | {nstr(Ea, 2)} | {nstr(Eb, 2)}")
print("PASS Proposition 1 bounds, and the error rates of Corollary 1 and Proposition 2")
print(f"d = {nstr(d, 8)}")

# Corollary 2: 1 + c - l_c(n) = (ln(cn) + 1 - ln 2)/(n-1) + O(ln^2 n / n^2)
for c in (mpf("0.5"), mpf(3), mpf(10)):
    for n in (10**2, 10**3, 10**4, 10**5):
        n = mpf(n)
        err = (1 + c - L_exact(n, c)) - (log(c * n) + 1 - log(2)) / (n - 1)
        assert abs(err) * n**2 / log(n) ** 2 < 2, "Corollary 2"
print("PASS Corollary 2 for c in {0.5, 3, 10} and n up to 10^5")

# Theorem 2: the constant C(theta)
for th in (pi / 2, pi / 3, 1, 2 * pi / 3, mpf(3)):
    c = cos(th)
    # cancellation-free forms: h = 2(1-c)s/(1+s+r) and h - (1-c) = (1-c)(s-1-r)/(1+s+r)
    r = lambda s: sqrt(1 + 2 * c * s + s * s)
    h = lambda s: 2 * (1 - c) * s / (1 + s + r(s))
    h_minus = lambda s: (1 - c) * (-1 - (2 * c * s + 1) / (s + r(s))) / (1 + s + r(s))
    Cnum = quad(lambda s: h(s) / s, [0, 0.5, 1]) + quad(lambda s: h_minus(s) / s, [1, 10, 100, 1e4, 1e8, inf])
    Cform = (1 - c) + (1 + c) * log((1 + c) / 2)
    check(f"Theorem 2 constant C(theta) at theta={nstr(th, 5)}", abs(Cnum - Cform) < mpf(10) ** -20)

# Table IV: C_hat_n(theta) = (n-1)(2 - L_theta(n)) - 2 sin^2(theta/2) ln n at n = 10^6
n = mpf(10) ** 6
print("\nTable IV:  theta | 2 sin^2(theta/2) | C(theta) | C_hat_n(theta), n = 10^6")
for num, den, lab in ((1, 6, "pi/6"), (1, 4, "pi/4"), (1, 3, "pi/3"), (1, 2, "pi/2"),
                      (2, 3, "2pi/3"), (3, 4, "3pi/4"), (5, 6, "5pi/6")):
    c = cos(pi * num / den)
    Lt = quad(lambda x: sqrt(1 + 2 * c * n * x ** (n - 1) + (n * x ** (n - 1)) ** 2),
              sorted({mpf(0), mpf(1)} | {1 - k / n for k in (1e5, 3e4, 1e4, 3e3, 1e3, 300, 100, 50, 20, 10, 5, 2, 1, 0.5, 0.1)}))
    Cform = (1 - c) + (1 + c) * log((1 + c) / 2)
    emp = (n - 1) * (2 - Lt) - (1 - c) * log(n)
    assert abs(emp - Cform) < 2 * log(n) ** 2 / n, "Table IV"
    print(f"  {lab:>6} | {nstr(1 - c, 6)} | {nstr(Cform, 8)} | {nstr(emp, 8)}")
print("PASS Table IV values agree with C(theta) to within 2 ln^2 n / n")

# Theorem 3: superellipse
K = 2 * sqrt(2) * log(1 + sqrt(2)) - 2 * log(2)
check("Theorem 3: F(0) = K", abs(quad(lambda t: (1 - sqrt(t * t + (1 - t) ** 2)) / (t * (1 - t)), [0, 0.5, 1]) - K) < mpf(10) ** -30)


def F(eps):
    f = lambda t: t ** (eps - 1) + (1 - t) ** (eps - 1) - sqrt(t ** (2 * eps - 2) + (1 - t) ** (2 * eps - 2))
    return 2 * quad(f, [0, mpf(10) ** -20, mpf(10) ** -10, mpf(10) ** -5, 0.01, 0.1, 0.5])


def P_direct(p):
    """Perimeter via y = (1-x^p)^(1/p) on [0, 2^(-1/p)], using symmetry."""
    p = mpf(p)
    xm = 2 ** (-1 / p)
    f = lambda x: sqrt(1 + (x ** (p - 1) * (1 - x**p) ** (1 / p - 1)) ** 2)
    return 8 * quad(f, [xm * k / 60 for k in range(61)])


for p in (2, 4, 10):
    check(f"Theorem 3 parametrization at p={p}", abs(4 * (2 - F(1 / mpf(p)) / p) - P_direct(p)) < mpf(10) ** -25)
check("P(2) = 2 pi", abs(4 * (2 - F(mpf(1) / 2) / 2) - 2 * pi) < mpf(10) ** -25)

print(f"\nK = {nstr(K, 12)}\nTable V:  p | P(p) | 8 - 4K/p | p^2 Delta(p)")
for p in (2, 4, 10, 20, 50, 100, 1000):
    P = 4 * (2 - F(1 / mpf(p)) / p)
    main = 8 - 4 * K / p
    assert abs(P - main) <= 12 / mpf(p) ** 2, "Theorem 3 bound"
    print(f"  {p:>5} | {nstr(P, 13)} | {nstr(main, 10)} | {nstr((P - main) * p**2, 4)}")
print("PASS Theorem 3 bound holds for every p in Table V")


def dPhi(t):
    a, b = 1 / t, 1 / (1 - t)
    r = sqrt(a * a + b * b)
    return a * log(t) + b * log(1 - t) - (a * a * log(t) + b * b * log(1 - t)) / r


F1 = 2 * quad(dPhi, [0, mpf(10) ** -30, mpf(10) ** -10, 0.01, 0.5])
kappa = -4 * F1
print(f"F'(0) = {nstr(F1, 12)},  kappa = -4 F'(0) = {nstr(kappa, 12)}")

# Problem 1: integer-relation (PSLQ) search for kappa; finding no relation is the expected outcome
s2, l2, l12 = sqrt(2), log(2), log(1 + sqrt(2))
basis = [kappa, 1, pi**2, l2**2, l12**2, s2 * l2 * l12, l2, s2 * l12]
rel = pslq(basis, tol=mpf(10) ** -30, maxcoeff=1000, maxsteps=10**6)
print("PSLQ relation for kappa:", rel if rel else "none found (coefficients up to 1000)")
