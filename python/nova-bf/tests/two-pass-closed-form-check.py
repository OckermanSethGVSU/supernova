"""Numerical checks for two-pass..

CPU-only tests using float64 and exact rational arithmetic.

1. table(): Compute bound constants and inverse dimension bounds.
2. check_accumulation_model(): Test the tensor-core accumulation model with
   adversarial inputs and exact arithmetic, including subnormals.
3. envelope_check(): Verify coefficient bounds across dimensions and models.
4. sharpness(): Show that the conversion error bounds are nearly attained.
5. composition_check(): Verify error-weight rules and the corrected
   division bounds using exact arithmetic.
6. overflow_check(): Bound scaled intermediates under worst-case inputs,
   including scaling factors greater than one and both scaling orders.
7. evaluation_error_check(): Compare binary64 error bounds against
   high-precision arithmetic as a supporting numerical check.

Certified expressions use only correctly rounded arithmetic and sqrt,
avoiding non-exact power operations.

rho_q and sigma_max represent the actual pass-one scaling factors,
not their ideal reciprocals. Use conservative upper bounds when needed,
since rounding can make stored scaling factors larger than the exact
reciprocals and Psi increases with both factors.
"""
from __future__ import annotations

import math
import random
from fractions import Fraction

import numpy as np

U = 2.0 ** -24          # binary32 unit roundoff
U16 = 2.0 ** -11        # binary16 unit roundoff
UBF = 2.0 ** -8         # bfloat16 unit roundoff
ETA16 = 2.0 ** -25      # binary16 absolute conversion floor, mu_t/2 (round-to-nearest conversion);
                        # a TRUNCATING conversion would use a full quantum mu_t
LAMBDA16 = 2.0 ** -14   # smallest binary16 normal (operand exponent floor)
LAMBDA32 = 2.0 ** -126  # smallest binary32 normal (bf16 / tf32 operand exponent floor)
FLOOR_COS = 2.0 ** -44  # absolute floor constant carried in C(d) (cosine bound)
D_MIN, D_MAX = 64, 2 ** 20


def gamma(n: int, u: float = U) -> float:
    return n * u / (1.0 - n * u)


def kappa_norm(d: int) -> float:
    """kappa(d): ||v|| <= kappa(d) * N whenever the computed binary32 norm N >= 2^-40.
    From the binary32 norm computation: ||v||^2 <= k0^2 N^2 + d 2^-150/(1-u), k0 the first
    two factors. At N >= 2^-40 the additive term is a relative (1 + d 2^-70).
    Evaluated with divisions, multiplications and sqrt only -- no pow: IEEE 754 requires
    correct rounding for + - * / sqrt but NOT for a generic library pow."""
    k0 = 1.0 / ((1.0 - U) * math.sqrt(1.0 - U)) * (1.0 / math.sqrt(1.0 - gamma(d - 1)))
    return k0 * (1.0 + d * 2.0 ** -70)


# --------------------------------------------------------------------------- hardware model
# A row is a property of the INSTRUCTION, not of the die: Turing's m16n8k8 (HMMA.1688) is
# (8, 1) -- measured on a T4 -- and shares the Ampere/Ada
# m16n8k16 row; the published N_FMA = 4 describes Volta's m8n8k4 (HMMA.884) alone.
HW_MODELS = {
    # name: (b = N_FMA, nu = extra alignment bits, operand exponent floor lambda_t, e_min or None)
    # [Khattak & Mikaitis 2026, Table 3; Valpey et al. 2025; Fasi et al. 2021]
    "Volta m8n8k4":     (4, 0, LAMBDA16, None),
    "Turing/Ampere/Ada": (8, 1, LAMBDA16, -132),
    "Hopper/Blackwell": (16, 2, LAMBDA16, -133),
    "Ampere/Ada bf16":  (8, 1, LAMBDA32, -132),
    "Hopper bf16":      (16, 2, LAMBDA32, -133),
    "Ampere/Ada tf32":  (4, 1, LAMBDA32, -132),
    "Hopper tf32":      (8, 2, LAMBDA32, -133),
}


def beta_per_block(b: int, nu: int) -> float:
    """Per-block relative growth: (b+1) aligned addends each losing < 2^(1-nu) u W,
    plus one final truncation to 24 bits losing < 2u |t|."""
    return (b + 1) * 2.0 ** (1 - nu) + 2.0


def kappa_acc_model(d: int, b: int, nu: int, extra_blocks: int = 0) -> float:
    m = -(-d // b) + extra_blocks
    return (1.0 + beta_per_block(b, nu) * U) ** m - 1.0


def kappa_acc(d: int, c_hw: float = 4.0) -> float:
    """Envelope kappa_acc(d) = c_hw d u (1 + c_hw d u). With c_hw = 4 it dominates the
    relative accumulation error of every listed model with m <= ceil(d/b) + s_x
    non-trivial blocks and s_x <= d/16 extra binary32 RN additions, for all
    64 <= d <= 2^20 (envelope_check sweeps every d)."""
    x = c_hw * d * U
    return x * (1.0 + x)


def theta(d: int, c_hw: float = 4.0) -> float:
    """(1 + kappa_acc)(1+u)^2 - 1, written without cancellation."""
    k = kappa_acc(d, c_hw)
    return k + (2 * U + U * U) * (1 + k)


# --------------------------------------------------------------------------- the bound
def bracket(d: int, u_q: float, u_c: float, c_hw: float = 4.0, half_out: bool = False) -> float:
    """The dimensionless bracket shared by the cosine bound (times Lambda) and the dot bound (times Qn P)."""
    th = theta(d, c_hw)
    inp = u_q + u_c + u_q * u_c
    # exact pass: gamma_d (1+u)^4/(1-u)^2 + 6u + 22u^2 covers the accumulation, the two
    # scalings and a scale-factor mismatch of up to ((1+u)/(1-u))^2 between the passes (the admissible scale-factor band)
    mism = ((1 + U) * (1 + U)) * ((1 + U) * (1 + U)) / ((1 - U) * (1 - U))
    core = mism * gamma(d) + 6 * U + 22 * U * U + th + inp * (1 + th)
    if half_out:  # unfused path: fp16 Gram written before sigma is applied
        core += U16 * (1 + th) * (1 + inp)
    return core


def Lambda(d: int) -> float:
    k = kappa_norm(d)
    return k * k * ((1 + U) * (1 + U))           # (1+u)^2: stored reciprocals fl(1/N)


def C_const(d: int, u_q: float, u_c: float, c_hw: float = 4.0, half_out: bool = False) -> float:
    """The run constant C(d), cosine units, before the final-add wrapper."""
    return Lambda(d) * bracket(d, u_q, u_c, c_hw, half_out) + FLOOR_COS


def Psi(d: int, u_q: float, u_c: float, eta_q: float, eta_c: float, rho_q: float, sigma_max: float,
        c_hw: float = 4.0, half_out: bool = False, lam_t: float = LAMBDA16) -> float:
    """The cosine bound's data-dependent absolute term Psi(q, S): subnormal conversions and
    the accumulation model's denormalised-anchor term (lam_t = smallest normal of the first-pass format).
    rho_q and sigma_max are the factors PASS ONE APPLIES, or upper bounds for them
    (module docstring). Psi is increasing in both."""
    th = theta(d, c_hw)
    k = kappa_norm(d)
    sd = math.sqrt(d)
    conv = (1 + th) * (k * (1 + U) * sd * (eta_q * (1 + u_c) * rho_q + eta_c * (1 + u_q) * sigma_max)
                       + d * eta_q * eta_c * rho_q * sigma_max)
    # (kappa_acc + d u): the accumulation model's anchor term plus its
    # zero-anchored trivial-block episodes
    anchor = (kappa_acc(d, c_hw) + d * U) * ((1 + U) * (1 + U)) * (
        lam_t * (k * (1 + U) * ((1 + u_q) * sigma_max + (1 + u_c) * rho_q) + sd * (eta_q + eta_c) * rho_q * sigma_max)
        + lam_t * lam_t * rho_q * sigma_max)
    out = (ETA16 * rho_q * sigma_max * ((1 + U) * (1 + U))) if half_out else 0.0
    return (conv + anchor) * ((1 + U16) if half_out else 1.0) + out


def eps_final(E: float, B: float) -> float:
    """The universal wrapper: eps >= (E (1+u) + u B)/(1-u) so that fl(A + eps) >= A + E
    for |A| <= B; evaluated in binary64, inflated by (1 + 2^-30), rounded toward +inf
    to binary32."""
    e = (E * (1 + U) + U * B) / (1 - U) * (1 + 2.0 ** -30)
    f = np.float32(e)
    if float(f) < e:
        f = np.nextafter(f, np.float32(np.inf))
    return float(f)


def eps_cos(d: int, u_q: float, u_c: float, c_hw: float = 4.0, half_out: bool = False,
            eta_q: float = 0.0, eta_c: float = 0.0, rho_q: float = 1.0, sigma_max: float = 1.0,
            lam_t: float = LAMBDA16) -> float:
    E = C_const(d, u_q, u_c, c_hw, half_out) + Psi(d, u_q, u_c, eta_q, eta_c, rho_q, sigma_max, c_hw, half_out, lam_t)
    B = Lambda(d) * (1 + theta(d, c_hw)) * (1 + u_q) * (1 + u_c) * ((1 + U16) if half_out else 1.0) + E
    return eps_final(E, B)


def eps_dot(d: int, u_q: float, u_c: float, N_q: float, Nc_max: float, c_hw: float = 4.0,
            half_out: bool = False, eta_q: float = 0.0, eta_c: float = 0.0,
            lam_t: float = LAMBDA16) -> float:
    k = kappa_norm(d)
    th = theta(d, c_hw)
    sd = math.sqrt(d)
    Qn, Cn = k * N_q, k * Nc_max
    E = Qn * Cn * bracket(d, u_q, u_c, c_hw, half_out)
    absolute = (1 + th) * (sd * (eta_q * (1 + u_c) * Cn + eta_c * (1 + u_q) * Qn)
                           + d * eta_q * eta_c)
    absolute += (kappa_acc(d, c_hw) + d * U) * (1 + U) ** 2 * (
        lam_t * ((1 + u_q) * Qn + (1 + u_c) * Cn + sd * (eta_q + eta_c)) + lam_t ** 2)
    if half_out:
        absolute = absolute * (1 + U16) + ETA16 * (1 + U) ** 2
    zeta = (2 * d - 1) * 2.0 ** -150 / (1 - d * U)
    E += absolute + 3.01 * (d + 8) * 2.0 ** -149 + (1 + U) ** 4 / (1 - U) ** 2 * zeta
    B = Qn * Cn * (1 + th) * (1 + u_q) * (1 + u_c) * ((1 + U16) if half_out else 1.0) + E
    return eps_final(E, B)


def table():
    # (label, u_q, u_c, eta_q, eta_c, lambda_t) -- the Psi terms use the full per-format constants
    cases = [("fp32 q, fp32 c -> fp16 both", U16, U16, ETA16, ETA16, LAMBDA16),
             ("fp32 q, fp16-stored c", U16, 0.0, ETA16, 0.0, LAMBDA16),
             ("fp16-exact q and c", 0.0, 0.0, 0.0, 0.0, LAMBDA16),
             ("bf16 pass one (both rounded)", UBF, UBF, 2.0 ** -134, 2.0 ** -134, LAMBDA32)]
    dims = [128, 384, 768, 1024, 1536, 3072, 4096, 8192]
    print("eps in cosine units, FUSED first pass, Psi at rho_q = sigma_max = 1 (unit norms), full eta/lambda_t per format")
    for c_hw, label in [(4.0, "envelope c_hw = 4 (proposed default)"), (1.375, "Ampere/Ada model, beta/b = 1.375"),
                        (1.0, "IEEE binary32 RN first pass, c_hw = 1")]:
        print(f"\n  {label}")
        print("  " + "d".rjust(6) + "".join(f"{c[0][:28]:>30}" for c in cases))
        for d in dims:
            print("  " + f"{d:6d}" + "".join(f"{eps_cos(d, uq, uc, c_hw, eta_q=eq, eta_c=ec, lam_t=lt):30.4e}"
                                            for _, uq, uc, eq, ec, lt in cases))
    # TF32 first pass: the two conversion modes, round to nearest and truncating.  The truncating row
    # doubles BOTH u and eta; using the round-to-nearest eta = 2^-137 with it is unsafe.
    print("\nTF32 first pass, envelope c_hw = 4, unit norms (both axes converted from binary32):")
    for lab, ut, et in [("round to nearest (cvt.rna)", 2.0 ** -11, 2.0 ** -137),
                        ("truncating              ", 2.0 ** -10, 2.0 ** -136)]:
        print(f"  {lab}  u_t = 2^{np.log2(ut):.0f}, eta_t = 2^{np.log2(et):.0f}: "
              + "  ".join(f"d={d}: {eps_cos(d, ut, ut, 4.0, eta_q=et, eta_c=et, lam_t=LAMBDA32):.4e}"
                          for d in (768, 4096)))
    mu_t = 2.0 ** -136                       # smallest positive TF32 subnormal
    t_bad = mu_t - 2.0 ** -149               # binary32-representable, truncates to zero in TF32
    assert 2.0 ** -10 * t_bad + 2.0 ** -137 < abs(t_bad - 0.0)     # the round-to-nearest pair FAILS
    assert 2.0 ** -10 * t_bad + mu_t >= abs(t_bad - 0.0)           # the truncating pair holds
    print(f"  truncating-conversion counterexample to the round-to-nearest eta: t = 2^-136 - 2^-149 truncates to 0,"
          f" error {t_bad:.4e}; RN allowance {2.0**-10*t_bad + 2.0**-137:.4e}"
          f" (ratio {(2.0**-10*t_bad + 2.0**-137)/t_bad:.3f} < 1 => unsafe);"
          f" truncating allowance {2.0**-10*t_bad + mu_t:.4e}: OK")

    print("\nUNFUSED first pass (fp16-written Gram), envelope, d = 768:")
    print(f"  fp32/fp32 {eps_cos(768, U16, U16, half_out=True, eta_q=ETA16, eta_c=ETA16):.4e}"
          f"   fp32 q / fp16 c {eps_cos(768, U16, 0.0, half_out=True, eta_q=ETA16):.4e}")
    for uc, lab in [(0.0, "fp32 q / fp16-stored c"), (U16, "fp32 q / fp32 c")]:
        print(f"\ncomposition of the bracket at d = 768, {lab}, envelope (additive pieces):")
        d = 768
        th = theta(d)
        inp = U16 + uc + U16 * uc
        parts = {"conversion (u_q+u_c+u_q u_c)(1+theta)": inp * (1 + th), "accumulation theta": th,
                 "exact path gamma_d (1+u)^4/(1-u)^2": gamma(d) * ((1 + U) * (1 + U)) * ((1 + U) * (1 + U)) / ((1 - U) * (1 - U)),
                 "scalings + mismatch 6u+22u^2": 6 * U + 22 * U * U}
        tot = sum(parts.values())
        for k, v in parts.items():
            print(f"  {k:40s} {v:.3e}  ({100*v/tot:5.1f}%)")
        print(f"  bracket total {tot:.4e}; x Lambda = {Lambda(d):.8f}; + floor const {FLOOR_COS:.1e}")
    print("\ndot metric: E = Qn * P * bracket (+ absolute terms); the bracket, envelope:")
    for d in dims:
        print(f"  d={d:5d}  fp32/fp32 {bracket(d, U16, U16):.4e}   fp32 q / fp16 c {bracket(d, U16, 0.0):.4e}")
    print("\nlargest d with eps_cos <= budget E (fp32->fp16 both, envelope); conversion floor = %.4e" % (2 * U16 + U16 * U16))
    print("  (the exact search is normative; the first-order formula (E - floor - 8u)/(5u) errs on the permissive side)")
    print("  (the search is clamped to the bound's range 64 <= d <= 2^20; 'none' = no admissible d)")
    for budget in [9.9e-4, 1.0e-3, 1.5e-3, 2.0e-3, 5.0e-3, 1.0e-2]:
        d = D_MIN - 1
        while d < D_MAX and eps_cos(d + 1, U16, U16, eta_q=ETA16, eta_c=ETA16) <= budget:
            d += 1
        approx = (budget - (2 * U16 + U16 * U16) - 8 * U) / (5 * U)
        if d < D_MIN:
            print(f"  E = {budget:.1e}: exact search: none (eps(64) = {eps_cos(64, U16, U16, eta_q=ETA16, eta_c=ETA16):.4e} > E); first-order formula {approx:9.0f}")
        else:
            print(f"  E = {budget:.1e}: exact search d <= {d:6d}; first-order formula {approx:9.0f}  ({100*(approx-d)/d:+.1f}%)")
    print("\nPsi (absolute term), fp32 q / fp16-stored c, d = 768, envelope:")
    for nq, sm in [(1.0, 1.0), (0.1, 1.0), (1e-2, 1.0), (1e-3, 1.0), (1.0, 1e3), (1.0, 1e5)]:
        ps = Psi(768, U16, 0.0, ETA16, 0.0, 1 / nq, sm)
        print(f"  N_q = {nq:6g}, sigma_max = {sm:6g}: Psi = {ps:.3e}   (C(768) = {C_const(768, U16, 0.0):.3e})")
    print("\ncrossover d where the accumulation term equals the conversion term:")
    for label, inp in [("fp32 both", 2 * U16), ("fp16-stored corpus", U16)]:
        print(f"  {label:20s}: envelope 4du -> d = {inp/(4*U):.0f};  Ampere model 1.375du -> d = {inp/(1.375*U):.0f}")


# --------------------------------------------------------------------------- emulation
def fl16(x: float) -> float:
    return float(np.float16(np.float32(x)))


def ilog2(x: Fraction) -> int:
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if Fraction(2) ** e > x:
        e -= 1
    return e


def operand_exponent(x: Fraction, lam_t: float) -> int:
    """Format exponent of a nonzero operand: floor(log2|x|), floored at the format's
    minimum normal exponent (denormalised significand below that)."""
    return max(ilog2(abs(x)), ilog2(Fraction(lam_t)))


def truncate_to_quantum(x: Fraction, quantum: Fraction) -> Fraction:
    n = abs(x) / quantum
    n = n.numerator // n.denominator
    return n * quantum if x >= 0 else -n * quantum


def truncate_to_p_bits(x: Fraction, p: int = 24) -> Fraction:
    if x == 0:
        return x
    e = ilog2(abs(x))
    if e < -126:                      # binary32 subnormal output: quantum 2^-149
        return truncate_to_quantum(x, Fraction(2) ** -149)
    return truncate_to_quantum(x, Fraction(2) ** (e - (p - 1)))


FLOOR_HITS = {"-126": 0, "e_min": 0}


def block_fma_dot(a: list[float], b: list[float], nfma: int, nu: int, lam_t: float,
                  zero_anchors: bool, e_min: int | None) -> Fraction:
    """Emulated tensor-core inner product. Anchor = largest of the product exponents
    e_a + e_b and the accumulator's binary32 exponent; a nonzero-subnormal accumulator
    floors the anchor at -126; with acc = 0 and all products below 2^e_min the anchor is
    e_min (Khattak & Mikaitis Sec. 4.1.2). `zero_anchors` decides whether a product with
    a zero operand still contributes its exponent pair to the anchor (the papers do not
    say; the accumulation bound covers both). Shifted-out bits below 2^(anchor-23-nu) are discarded; the
    aligned addends are summed exactly; the result is normalised and truncated to 24
    significant bits."""
    acc = Fraction(0)
    lam = Fraction(lam_t)
    for j in range(0, len(a), nfma):
        idx = range(j, min(j + nfma, len(a)))
        prods = [(Fraction(a[i]) * Fraction(b[i]), Fraction(a[i]), Fraction(b[i])) for i in idx]
        anchors = []
        for _, x, y in prods:
            if x != 0 and y != 0:
                anchors.append(operand_exponent(x, lam_t) + operand_exponent(y, lam_t))
            elif zero_anchors:
                ex = operand_exponent(x, lam_t) if x != 0 else ilog2(lam)
                ey = operand_exponent(y, lam_t) if y != 0 else ilog2(lam)
                anchors.append(ex + ey)
        if acc != 0:
            ea = ilog2(abs(acc))
            if ea < -126:
                FLOOR_HITS["-126"] += 1
            anchors.append(max(ea, -126))
        elif anchors and e_min is not None:
            if max(anchors) < e_min:
                FLOOR_HITS["e_min"] += 1
            anchors = [max(max(anchors), e_min)]
        if not anchors:
            continue
        anchor = max(anchors)
        quantum = Fraction(2) ** (anchor - 23 - nu)
        total = sum(truncate_to_quantum(p, quantum) for p, _, _ in prods) + truncate_to_quantum(acc, quantum)
        acc = truncate_to_p_bits(total, 24)
    return acc


def accumulation_bound(a: list[float], b: list[float], nfma: int, nu: int, lam_t: float) -> Fraction:
    d = len(a)
    m = -(-d // nfma)
    beta = beta_per_block(nfma, nu)
    growth = Fraction((1 + beta * U) ** m)
    absum = sum(abs(Fraction(x) * Fraction(y)) for x, y in zip(a, b))
    rel = (growth - 1) * absum
    ainf = max(abs(Fraction(x)) for x in a)
    binf = max(abs(Fraction(x)) for x in b)
    lam = Fraction(lam_t)
    Dq = lam * (ainf + binf) + lam * lam
    anchor_abs = m * (nfma + 1) * Fraction(2) ** (1 - nu) * Fraction(U) * Dq * growth
    episodes = (m + 1) * Fraction(2) ** (1 - nu) * Fraction(U) * Dq * growth   # zero-anchored trivial blocks
    phi = (nfma + 1) * Fraction(2) ** (-126 - 23 - nu) + Fraction(2) ** -149
    return rel + anchor_abs + episodes + m * phi * growth


def check_accumulation_model(trials: int = 280, seed: int = 0):
    print("\nper-block beta and beta/b per published model (d = 768; 's_x adds' = d/16 extra binary32 RN additions):")
    for n, (b, nu, _, _) in HW_MODELS.items():
        km = kappa_acc_model(768, b, nu)
        print(f"  {n:18s} b={b:2d} nu={nu}  beta={beta_per_block(b, nu):5.1f}  beta/b={beta_per_block(b, nu)/b:.4f}"
              f"  model={km:.3e}  +adds={km + 48*U:.3e}  envelope={kappa_acc(768):.3e}"
              f"  used: {100*km/kappa_acc(768):.0f}% / {100*(km+48*U)/kappa_acc(768):.0f}%")
    rng = random.Random(seed)
    worst_ratio = {n: 0.0 for n in HW_MODELS}
    worst_rel = {n: 0.0 for n in HW_MODELS}
    rel_only_violations = 0
    for t in range(trials):
        d = rng.choice([16, 64, 256, 768])
        kind = t % 9
        if kind == 0:      # iid normal
            a = [fl16(rng.gauss(0, 1)) for _ in range(d)]
            b = [fl16(rng.gauss(0, 1)) for _ in range(d)]
        elif kind == 1:    # same sign, descending magnitudes (truncation drift accumulates)
            a = sorted((fl16(abs(rng.gauss(0, 1)) * 2.0 ** -rng.randint(0, 6)) for _ in range(d)), reverse=True)
            b = [fl16(1.0 + rng.random()) for _ in range(d)]
        elif kind == 2:    # one dominant product, the rest tiny
            a = [fl16(1.0)] + [fl16(2.0 ** -12 * (1 + rng.random())) for _ in range(d - 1)]
            b = [fl16(1.0 + rng.random() * 0.5) for _ in range(d)]
        elif kind == 3:    # cancellation
            half = d // 2
            x = [fl16(rng.gauss(0, 1)) for _ in range(half)]
            a = x + [-v for v in x]
            b = [fl16(1.0 + 2.0 ** -10 * rng.randint(0, 3)) for _ in range(d)]
        elif kind == 4:    # SUBNORMAL operands with large partners: anchor >> magnitude
            a = [2.0 ** -24] * d
            b = [fl16(2.0 ** 15)] + [1.0] * (d - 1)
        elif kind == 5:    # ZERO operand paired with the largest fp16 value
            a = [0.0] + [2.0 ** -24] * (d - 1)
            b = [65504.0] + [1.0] * (d - 1)
        elif kind == 6:    # TINY values (bf16/tf32 rows): products below 2^-126, accumulator subnormal -> floors bind
            a = [float(np.float32(2.0 ** -70 * (1 + rng.randint(0, 127) / 128))) for _ in range(d)]
            b = [float(np.float32(2.0 ** -60 * (1 + rng.randint(0, 127) / 128) * (1 if rng.random() < 0.7 else -1))) for _ in range(d)]
        elif kind == 7:    # ALL products below 2^e_min with acc = 0 at the start: the e_min floor binds (bf16/tf32 rows)
            a = [float(np.float32(2.0 ** -70 * (1 + rng.randint(0, 127) / 128))) for _ in range(d)]
            b = [float(np.float32(2.0 ** -70 * (1 + rng.randint(0, 127) / 128) * (1 if rng.random() < 0.7 else -1))) for _ in range(d)]
        else:              # mixed: a few subnormal operands inside otherwise unit-scale data
            a = [fl16(rng.gauss(0, 0.04)) for _ in range(d)]
            for i in rng.sample(range(d), max(1, d // 16)):
                a[i] = 2.0 ** -24 * rng.randint(1, 500)
            b = [fl16(rng.gauss(0, 0.04)) for _ in range(d)]
            b[rng.randrange(d)] = fl16(60000.0)
        exact = sum(Fraction(x) * Fraction(y) for x, y in zip(a, b))
        absum = sum(abs(Fraction(x) * Fraction(y)) for x, y in zip(a, b))
        for name, (nfma, nu, lam_t, e_min) in HW_MODELS.items():
            if lam_t == LAMBDA16 and kind in (6, 7):
                continue          # fp16 inputs cannot be that small; kinds 6-7 are for the bf16/tf32 rows
            if lam_t == LAMBDA32 and any(abs(v) < LAMBDA32 and v != 0 for v in a + b):
                continue          # fp16 subnormals do not exist in bf16/tf32
            bnd = accumulation_bound(a, b, nfma, nu, lam_t)
            rel_only = Fraction(kappa_acc_model(d, nfma, nu)) * absum
            for zero_anchors in (False, True):
                got = block_fma_dot(a, b, nfma, nu, lam_t, zero_anchors, e_min)
                err = abs(got - exact)
                assert err <= bnd, (name, d, kind, zero_anchors, float(err), float(bnd))
                worst_ratio[name] = max(worst_ratio[name], float(err / bnd) if bnd else 0.0)
                if err > rel_only:
                    rel_only_violations += 1
                if absum:
                    worst_rel[name] = max(worst_rel[name], float(err / rel_only))
    print("\nAccumulation-model emulation (exact rational, both zero-anchor variants): worst |err| / bound (must be <= 1):")
    for k in HW_MODELS:
        print(f"  {k:18s} full bound: {worst_ratio[k]:.3f}    relative-only part: {worst_rel[k]:.3f}")
    print(f"  cases where the RELATIVE-ONLY bound (no absolute anchor term) is violated: {rel_only_violations}"
          " -- all from the subnormal/zero-operand adversaries; the absolute anchor term covers them")
    print(f"  floor branches exercised: -126 floor in {FLOOR_HITS['-126']} blocks, e_min floor in {FLOOR_HITS['e_min']} blocks (both must be > 0)")
    assert FLOOR_HITS["-126"] > 0 and FLOOR_HITS["e_min"] > 0


def envelope_check():
    """Exhaustive over every d in [64, 2^20], every model, with m = ceil(d/b) + s_x
    non-trivial blocks and s_x = d/16 extra binary32 RN additions:
      (a) relative:  (1+beta u)^m (1+u)^s_x - 1 <= 4du(1+4du)             [accumulation, relative]
      (b) anchor:    m (b+1) 2^(1-nu) u G       <= 4du(1+4du)             [cosine bound, Psi]
      (c) floor:     m ((b+1) 2^(-149-nu) + 2^-149) G <= 3 (d+8) 2^-149  [cosine bound, 2^-44 floor]
      (d) episodes:  (m+s_x+1) 2^(1-nu) u G             <= d u                [zero-anchored trivial blocks, s_x+1 chains]"""
    d = np.arange(D_MIN, D_MAX + 1, dtype=np.float64)
    env = 4 * d * U * (1 + 4 * d * U)
    sx = d / 16
    ok = True
    for n, (b, nu, _, _) in HW_MODELS.items():
        beta = beta_per_block(b, nu)
        m = np.ceil(d / b) + sx
        G = np.exp(m * np.log1p(beta * U) + sx * np.log1p(U))
        rel = G - 1
        anchor = m * (b + 1) * 2.0 ** (1 - nu) * U * G
        floor = m * ((b + 1) * 2.0 ** (-149 - nu) + 2.0 ** -149) * G
        episodes = (m + sx + 1) * 2.0 ** (1 - nu) * U * G     # one leading/trailing episode per chain
        bad_a = int((rel > env).sum()); bad_b = int((anchor > env).sum()); bad_c = int((floor > 3 * (d + 8) * 2.0 ** -149).sum())
        bad_d = int((episodes > d * U).sum())
        ok &= (bad_a == 0 and bad_b == 0 and bad_c == 0 and bad_d == 0)
        print(f"  {n:18s}: violations relative {bad_a}, anchor {bad_b}, floor {bad_c}, episodes {bad_d};"
              f"  max ratios {np.max(rel/env):.3f} / {np.max(anchor/env):.3f} / {np.max(floor/(3*(d+8)*2.0**-149)):.3f}"
              f" / {np.max(episodes/(d*U)):.3f}")
    assert ok
    print("envelope check: all four coefficient claims (a-d) hold for every model and every 64 <= d <= 2^20: OK")
    dm = D_MAX
    # The cosine bound's absolute pieces at the worst admissible d.
    # The binary32 exact pass's underflow allowance carries a PROPAGATION factor as well as a rounding count: an
    # absolute error committed early is carried through every later rounding, which costs
    # (1+u)^(d-1) <= 1/(1-du) <= 16/15 in the admitted range (4du <= 1/4).  Summing the local errors
    # alone understates zeta by 6.7% at d = 2^20.
    prop = 1.0 / (1.0 - dm * U)
    zeta_nofma = 1.01 * (2 * dm - 1) * 2.0 ** -70 * prop  # binary32 RN, multiply-then-add: d products + d-1 adds
    zeta_fma = 1.01 * dm * 2.0 ** -70 * prop              # binary32 RN with FMA, for comparison
    floor_l2 = 3 * (dm + 8) * 2.0 ** -69 * (1 + U) ** 2  # accumulation floor
    scalings = 2.0 ** -108                               # four subnormal scaling roundings
    splitk = (dm // 16) * 2.0 ** -150 * 2.0 ** 80 * (1 + U) ** 2   # s_x subnormal split-K additions
    tot = zeta_nofma + floor_l2 + scalings + splitk
    print(f"  cosine-bound floor constant at d = 2^20 [binary32 propagation 1/(1-du) = {prop:.4f}]:"
          f" zeta(no FMA) {zeta_nofma/FLOOR_COS:.4f}"
          f" + accumulation floor {floor_l2/FLOOR_COS:.4f} + scalings {scalings/FLOOR_COS:.2e}"
          f" + split-K {splitk/FLOOR_COS:.4f} = {tot/FLOOR_COS:.3f} x 2^-44 (must be < 1;"
          f" {(tot - zeta_nofma + zeta_fma)/FLOOR_COS:.3f} with FMA)")
    assert tot < FLOOR_COS


def sharpness(d: int = 768):
    x = np.float32(1.0 + 2.0 ** -11 - 2.0 ** -23)       # binary32-exact, rounds DOWN to 1.0 in fp16
    q = np.full(d, x, dtype=np.float32)
    qh = q.astype(np.float16).astype(np.float64)
    exact = float(np.dot(q.astype(np.float64), q.astype(np.float64)))
    approx = float(np.dot(qh, qh))                      # integer: every accumulator is exact
    rel = (exact - approx) / exact
    print(f"\nSharpness at d={d}: (q.q - q_h.q_h)/(q.q) = {rel:.6e}  vs  2 u16 = {2*U16:.6e}  ratio {rel/(2*U16):.4f}")
    print(f"  fp16 value of x: {float(np.float16(x))}, relative rounding error {(float(x)-1)/float(x):.6e} (u16 = {U16:.6e})")
    c = qh                                              # fp16-stored corpus case: u_c = 0
    rel_c = (float(np.dot(q.astype(np.float64), c)) - float(np.dot(qh, c))) / (float(np.linalg.norm(q)) * float(np.linalg.norm(c)))
    print(f"  fp16-stored corpus (c = q_h): (q.c - q_h.c)/(|q||c|) = {rel_c:.6e}  vs  u16 = {U16:.6e}  ratio {rel_c/U16:.4f}")


def composition_check():
    """The four weight-composition inequalities, in EXACT RATIONAL arithmetic.

    A node's weight w bounds its relative error by gamma_w = w u64/(1 - w u64):

        leaf (exactly representable)   w = 0
        fl(a+b), a, b > 0              w = max(w_a, w_b) + 1
        fl(a*b)                        w = w_a + w_b + 1
        fl(a/b)                        w = w_a + 2 w_b + 1     <-- inexact DENOMINATOR costs twice
        fl(sqrt(a))                    w = w_a + 1

    The division rule is the one that needs care: charging a quotient w_a + w_b + 1 is
    FALSE.  Counterexample (j = 0, k = 2), exhibited below in exact rationals.
    Then the quotient-aware weight of C, Psi, E, B and the wrapper is recounted from the
    formulas as this file writes them (with the pow-free kappa).  The root of the tree
    is the wrapper's argument WITH the (1 + 2^-30) inflation folded in, which is what
    fl^ rounds toward +inf.  The weight IS 113; n <= 256 is a looser constant quoted for
    convenience, and doubling it is optional conservatism against a miscount, not a
    step of the argument."""
    u = Fraction(1, 2 ** 53)                 # u64
    g = lambda n: Fraction(n) * u / (1 - Fraction(n) * u)

    # the counterexample to a j + k + 1 division rule: |alpha| <= gamma_0 = 0,
    # |beta| <= gamma_2 with the denominator at its LOW extreme, final rounding upward.
    excess = ((1 + g(0)) / (1 - g(2)) * (1 + u) - 1) - g(3)
    assert excess == u * u * (1 + 6 * u) / ((1 - 3 * u) * (1 - 4 * u))
    assert excess > 0
    print(f"\nweight composition: a j + k + 1 division rule fails at (j, k) = (0, 2):"
          f" attained relative error - gamma_3 = +{float(excess):.4e} > 0"
          f"  [= u^2 (1+6u)/((1-3u)(1-4u)), exact]")

    ws = list(range(0, 65)) + [100, 127, 128, 255, 256, 300, 511, 512, 700, 1000, 1023, 1024,
                               1500, 2047, 2048]
    for j in ws:
        assert (1 + g(j)) * (1 + u) <= 1 + g(j + 1)                 # addition, a, b > 0
        assert (1 - g(j)) * (1 - u) >= 1 - g(j + 1)
        assert (1 + g(j)) * (1 + u) * (1 + u) <= (1 + g(j + 1)) ** 2   # sqrt, squared
        assert (1 - g(j)) * (1 - u) * (1 - u) >= (1 - g(j + 1)) ** 2
        assert 1 / (1 - g(j)) - 1 <= g(2 * j)                       # the crux of the division rule
        for k in ws:
            assert (1 + g(j)) * (1 + g(k)) * (1 + u) <= 1 + g(j + k + 1)          # multiplication
            assert (1 - g(j)) * (1 - g(k)) * (1 - u) >= 1 - g(j + k + 1)
            assert (1 + g(j)) / (1 - g(k)) * (1 + u) <= 1 + g(j + 2 * k + 1)      # division
            assert (1 - g(j)) / (1 + g(k)) * (1 - u) >= 1 - g(j + 2 * k + 1)
    print(f"  all four composition inequalities hold in exact rationals for every j, k in"
          f" {len(ws)} weights up to {ws[-1]} (u64 = 2^-53): OK")

    class W:                                  # weight of an expression tree node
        def __init__(s, w=0): s.w = w
        def __add__(a, b): return W(max(a.w, b.w) + 1)      # positive addition; also the 1 - x sites,
        def __sub__(a, b): return W(max(a.w, b.w) + 1)      # which attenuate their input error by >= 14x
        def __mul__(a, b): return W(a.w + b.w + 1)
        def __truediv__(a, b): return W(a.w + 2 * b.w + 1)
        def sqrt(a): return W(a.w + 1)
    L = lambda: W(0)
    one, uu, d, i = L(), L(), L(), L()        # 1, u, d, small integer constants
    uq, uc, eq, ec, lam, rho, sm, Nq, Nc = (L() for _ in range(9))
    c44, c70, c30, c150, c149 = (L() for _ in range(5))
    g1 = (d - one) * uu / (one - (d - one) * uu)                            # gamma_{d-1}
    kap = one / ((one - uu) * (one - uu).sqrt()) * (one / (one - g1).sqrt()) * (one + d * c70)
    Lam = kap * kap * ((one + uu) * (one + uu))
    x = (i * d) * uu
    kacc = x * (one + x)
    th = kacc + (uu * i + uu * uu) * (one + kacc)
    gd = d * uu / (one - d * uu)
    mism = (((one + uu) * (one + uu)) * ((one + uu) * (one + uu))) / ((one - uu) * (one - uu))
    inp = (uq + uc) + uq * uc
    core = mism * gd + uu * i + (uu * uu) * i + th + inp * (one + th)
    C = Lam * core + c44
    sd, k1 = d.sqrt(), kap * (one + uu)
    conv = (one + th) * ((k1 * sd) * ((eq * (one + uc)) * rho + (ec * (one + uq)) * sm)
                         + (((d * eq) * ec) * rho) * sm)
    anch = ((kacc + d * uu) * ((one + uu) * (one + uu))) * (
        lam * (k1 * ((one + uq) * sm + (one + uc) * rho) + ((sd * (eq + ec)) * rho) * sm)
        + (lam * lam * rho) * sm)
    Psi = conv + anch
    E = C + Psi
    B = ((Lam * (one + th)) * (one + uq)) * (one + uc) + E
    eps_w = (E * (one + uu) + uu * B) / (one - uu) * (one + c30)
    Qn, Cn = kap * Nq, kap * Nc
    zeta = ((i * d - one) * c150) / (one - d * uu)
    Ed = ((Qn * Cn) * core
          + (one + th) * (sd * ((eq * (one + uc)) * Cn + (ec * (one + uq)) * Qn) + (d * eq) * ec)
          + ((kacc + d * uu) * ((one + uu) * (one + uu)))
          * (lam * (((one + uq) * Qn + (one + uc) * Cn) + sd * (eq + ec)) + lam * lam)
          + (i * (d + i)) * c149
          + mism * zeta)
    Bd = (((Qn * Cn) * (one + th)) * (one + uq)) * (one + uc) + Ed
    epsd_w = (Ed * (one + uu) + uu * Bd) / (one - uu) * (one + c30)
    print(f"  quotient-aware weights: gamma_d {gd.w}, kappa {kap.w}, Lambda {Lam.w},"
          f" kappa_acc {kacc.w}, theta {th.w}, bracket {core.w}, C {C.w}, Psi {Psi.w},"
          f" E {E.w}, B {B.w}")
    n = max(eps_w.w, epsd_w.w)
    print(f"  root weight (incl. the (1 + 2^-30) inflation): cosine"
          f" {eps_w.w}, dot {epsd_w.w} -- so n = {n} <= 256 and"
          f" gamma_256^(64) = {256 * 2.0 ** -53 / (1 - 256 * 2.0 ** -53):.4e} < 2^-44"
          f" = {2.0 ** -44:.4e}")
    print(f"  (doubling the enumeration, 2 x {n} = {2 * n} <= 256, is optional conservatism"
          " against a miscount, NOT a step of the argument)")
    assert n <= 256 and 2 * n <= 256
    assert 256 * 2.0 ** -53 / (1 - 256 * 2.0 ** -53) < 2.0 ** -44


def overflow_check():
    """The cosine bound's no-overflow step, for the SCALED intermediates.

    It is tempting to dismiss them as "smaller still" than the raw values; that is FALSE:
    a factor 1/N with N < 1 ENLARGES what it multiplies, and at the smallest admitted
    norm N = 2^-40 it does so by 2^40, turning a raw value near 2^-40 into one of order
    1.  A scaled value can therefore be the pass's LARGEST binary32 intermediate.

    Each scaled intermediate is bounded here at the worst admitted norms
    (2^-40 <= N <= 2^126 on each axis, N_q max_S N_c <= 2^126) and the worst admitted
    dimension d = 2^20, in BOTH scaling orders -- the scale rules permit either -- and asserted to
    be < 2^127, itself a factor ~2 inside binary32's overflow threshold ~2^128.  The
    cancellation that makes it work: the divisor sigma_a(c) <= (1+u)/N_c is THAT
    COLUMN'S OWN N_c, not the slice's max_S N_c that appears in the aggregate bound on
    Pi, so it cancels the N_c inside Pi_c and leaves a bound proportional to N_q alone.
    Under DOT (the dot bound) rho_a = sigma_a = rho_e = sigma_e = 1 exactly, so no scaled
    intermediate exists there and the raw bounds already suffice.
    """
    d = D_MAX                       # 2^20, the worst admitted dimension
    N_HI, N_LO = 2.0 ** 126, 2.0 ** -40          # per-axis norm guards
    k, gd, kacc = kappa_norm(d), gamma(d), kappa_acc(d)
    zeta = (2 * d - 1) * 2.0 ** -150 / (1 - d * U)   # binary32 RN, non-FMA order, with propagation
    sd = math.sqrt(d)                                # = 2^10 at d = 2^20
    ut = UBF                                         # (1+2^-8) is the envelope over every u_t
    lg = lambda x: math.log2(x)
    print("\nno-overflow check for the SCALED intermediates (cosine bound), d = 2^20,")
    print(f"  guards 2^-40 <= N <= 2^126 per axis and N_q max_S N_c <= 2^126;"
          f"  gamma_d = {gd:.7f}, kappa = {k:.7f}, kappa_acc = {kacc:.4f}")
    print(f"  binary32 overflows above ~2^128; every line below must be < 2^127")

    # ---- exact pass: |g_e| <= (1+gamma_d) kappa^2 N_q N_c + zeta   [binary32 RN exact pass and norms]
    coef_e = (1 + gd) * k * k * (1 + U)              # (1+gamma_d) kappa^2 (1+u)
    e_one = coef_e * N_HI + (1 + U) * zeta / N_LO    # one scaling: either axis, by symmetry
    e_both = (1 + gd) * k * k * (1 + U) * (1 + U) + (1 + U) * (1 + U) * zeta / (N_LO * N_LO)
    print(f"  exact pass, coefficient (1+gamma_d) kappa^2 (1+u) = {coef_e:.7f};"
          f" underflow piece (1+u) zeta/N_c = {(1 + U) * zeta / N_LO:.3e} = 2^{lg((1 + U) * zeta / N_LO):.1f}")
    print(f"    |sigma_e g_e| and |rho_e g_e| <= {coef_e:.4f} x 2^126 + tiny = 2^{lg(e_one):.3f}"
          f"   (2^127/that = {2.0 ** 127 / e_one:.2f}x, 2^128/that = {2.0 ** 128 / e_one:.2f}x = 2^{lg(2.0 ** 128 / e_one):.3f})")
    print(f"    |rho_e sigma_e g_e| (both scalings) <= {e_both:.7f} -- O(1), as a cosine must be")
    assert e_one < 2.0 ** 127 and e_both < 2.0 ** 127

    # ---- first pass: |g_a| <= (1+kappa_acc) Pi_c + (absolute terms)   [accumulation model]
    #      Pi_c <= ||q_h|| ||c_h||; scale by sigma_a(c) <= (1+u)/N_c, that column's own N_c.
    #      The eta_c sqrt(d)/N_c piece is tamed by ||q_h|| <= sqrt(d) Omega_t: every
    #      component of q_h is a value of the first-pass format, by the overflow guard
    #      kappa(d) N_q <= Omega_t on a converted axis and by format membership otherwise.
    for name, eta_t, om_t, lam_t in [("binary16", ETA16, 65504.0, LAMBDA16),
                                     ("bfloat16", 2.0 ** -134, 3.3895313892515355e38, LAMBDA32),
                                     ("tf32 RN", 2.0 ** -137, 3.4028234663852886e38, LAMBDA32),
                                     ("tf32 trunc", 2.0 ** -136, 3.4028234663852886e38, LAMBDA32)]:
        main = (1 + U) * (1 + ut) * (1 + ut) * k * k * N_HI            # <= 1.08 x 2^126
        cross_q = (1 + U) * (1 + ut) * k * eta_t * sd                  # eta_q sqrt(d) piece
        cross_c = (1 + U) * eta_t * d * om_t * 2.0 ** 40               # eta_t Omega_t <= 2^-6
        Dq = 2 * lam_t * om_t + lam_t * lam_t                          # lambda_t Omega_t <= 4
        absol = ((kacc + d * U) * (1 + U) * (1 + U) * Dq * 2.0 ** 40
                 + 3.01 * (d + 8) * 2.0 ** -149 * 2.0 ** 40)           # anchor, episodes, floor
        a_one = (1 + kacc) * (main + cross_q + cross_c) + absol
        # both scalings: each axis contributes (1+u_t) kappa + eta_t sqrt(d)/N <= (1+u_t) kappa + eta_t sqrt(d) 2^40
        ax = (1 + ut) * k + eta_t * sd * 2.0 ** 40
        a_both = (1 + kacc) * (1 + U) * (1 + U) * ax * ax + ((kacc + d * U) * (1 + U) * (1 + U) * Dq
                                                             + 3.01 * (d + 8) * 2.0 ** -149) * 2.0 ** 80
        print(f"  first pass, {name:10s} eta_t Omega_t = 2^{lg(eta_t * om_t):.2f}, lambda_t Omega_t = {lam_t * om_t:.3f}:"
              f" main {main / 2.0 ** 126:.4f} x 2^126, residuals 2^{lg(cross_q + cross_c + absol):.1f}")
        print(f"    |sigma_a(c) g_a| and |rho_a g_a| <= {a_one / 2.0 ** 126:.4f} x 2^126 = 2^{lg(a_one):.3f}"
              f"   (2^127/that = {2.0 ** 127 / a_one:.2f}x, 2^128/that = {2.0 ** 128 / a_one:.2f}x = 2^{lg(2.0 ** 128 / a_one):.3f})")
        print(f"    both scalings <= 2^{lg(a_both):.2f}  (the absolute part is Psi's degenerate-norm"
              f" cliff, valid but useless -- not an overflow)")
        assert a_one < 2.0 ** 127 and a_both < 2.0 ** 127
    print("  every scaled intermediate is < 2^127 at the worst admitted norms and d = 2^20,"
          " in both scaling orders: OK")


def evaluation_error_check():
    """CORROBORATION for the a-priori bound on the binary64 evaluation error of eps
    (C, Psi, B and the wrapper): gamma_256^(64) < 2^-44, from the quotient-aware
    weight of the expression (root weight 113, doubled to n <= 226 <= 256, u64 = 2^-53) and
    the fact that the formulas are positive arithmetic apart from four 1 - x sites with
    x <= 1/15.  This function checks the same
    quantity against 60-digit decimal arithmetic at a handful of points and asserts
    that bound.  It is not the justification -- a finite set of test points cannot be.
    Uses the standard library only."""
    from decimal import Decimal as D, getcontext
    getcontext().prec = 60
    u = D(2) ** -24
    worst = 0.0
    for d in [64, 65, 128, 768, 4096, 2 ** 20]:
        for uq, uc, eq, ec in [(U16, U16, ETA16, ETA16), (U16, 0.0, ETA16, 0.0), (0.0, 0.0, 0.0, 0.0), (UBF, UBF, 2.0 ** -134, 2.0 ** -134)]:
            for rho, sm in [(1.0, 1.0), (1e3, 1e5)]:
                Dd = D(d)
                gm = Dd * u / (1 - Dd * u); gm1 = (Dd - 1) * u / (1 - (Dd - 1) * u)
                k = 1 / ((1 - u) * (1 - u).sqrt()) / (1 - gm1).sqrt() * (1 + Dd * D(2) ** -70)
                lam = k ** 2 * (1 + u) ** 2
                x = 4 * Dd * u; ka = x * (1 + x); th = ka + (2 * u + u * u) * (1 + ka)
                inp = D(uq) + D(uc) + D(uq) * D(uc)
                core = (1 + u) ** 4 / (1 - u) ** 2 * gm + 6 * u + 22 * u * u + th + inp * (1 + th)
                C = lam * core + D(2) ** -44
                sd = Dd.sqrt(); lt = D(LAMBDA16)
                conv = (1 + th) * (k * (1 + u) * sd * (D(eq) * (1 + D(uc)) * D(rho) + D(ec) * (1 + D(uq)) * D(sm)) + Dd * D(eq) * D(ec) * D(rho) * D(sm))
                anc = (ka + Dd * u) * (1 + u) ** 2 * (lt * (k * (1 + u) * ((1 + D(uq)) * D(sm) + (1 + D(uc)) * D(rho)) + sd * (D(eq) + D(ec)) * D(rho) * D(sm)) + lt ** 2 * D(rho) * D(sm))
                E = C + conv + anc
                B = lam * (1 + th) * (1 + D(uq)) * (1 + D(uc)) + E
                eps_exact = (E * (1 + u) + u * B) / (1 - u)
                E64 = C_const(d, uq, uc) + Psi(d, uq, uc, eq, ec, rho, sm)
                B64 = Lambda(d) * (1 + theta(d)) * (1 + uq) * (1 + uc) + E64
                eps64 = (E64 * (1 + U) + U * B64) / (1 - U)
                rel = float((D(eps64) - eps_exact) / eps_exact)
                worst = max(worst, abs(rel))
    print(f"\nbinary64 evaluation of eps (C, Psi, B, wrapper) vs 60-digit decimal arithmetic: worst |relative error| {worst:.2e}"
          f"  = {worst/2.0**-30:.2e} x 2^-30 (the wrapper's inflation; must be << 1)")
    print(f"  the a-priori evaluation-error bound is gamma_256^(64) = {256*2.0**-53/(1-256*2.0**-53):.3e} < 2^-44 = {2.0**-44:.3e};"
          f" the measured worst is {worst/2.0**-44:.2e} x that")
    assert worst < 2.0 ** -44


if __name__ == "__main__":
    table()
    check_accumulation_model()
    print()
    envelope_check()
    sharpness()
    composition_check()
    overflow_check()
    evaluation_error_check()
