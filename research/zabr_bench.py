"""
Realistic 100k IR options benchmark:
  - 3 indexes: USD SOFR, USD Term SOFR, EUR EURIBOR 3M
  - Instrument types: cap/floor (payer/receiver on caplets) + swaption payer/receiver
  - ZABR closed-form normal vol  →  Bachelier price + 4 Greeks
  - QuantLib used ONLY for: curve bootstrap + forward/annuity extraction
  - Everything else: fully vectorized numpy
"""

import numpy as np
from scipy.stats import norm
from scipy.special import ndtr
import time
import QuantLib as ql

N = 100_000
rng = np.random.default_rng(42)

# ── 0. QuantLib date setup ────────────────────────────────────────────────────
today = ql.Date(16, 4, 2026)
ql.Settings.instance().evaluationDate = today

t0 = time.perf_counter()

# ── 1. Curve bootstrap ────────────────────────────────────────────────────────
# USD SOFR OIS
sofr_inst = [
    ("1W", 0.0528), ("1M", 0.0525), ("3M", 0.0510), ("6M", 0.0498),
    ("1Y", 0.0480), ("2Y", 0.0450), ("3Y", 0.0430), ("5Y", 0.0410),
    ("7Y", 0.0400), ("10Y", 0.0390), ("15Y", 0.0385), ("20Y", 0.0380),
    ("30Y", 0.0375),
]
sofr_helpers = [
    ql.OISRateHelper(2, ql.Period(t), ql.QuoteHandle(ql.SimpleQuote(r)), ql.Sofr())
    for t, r in sofr_inst
]
sofr_curve = ql.PiecewiseLogCubicDiscount(today, sofr_helpers, ql.Actual360())
sofr_curve.enableExtrapolation()

# EUR EURIBOR 3M — simplified flat (good enough for benchmark)
eur_curve = ql.FlatForward(today, 0.0245, ql.Actual360())
eur_curve.enableExtrapolation()

# USD Term SOFR — flat (quoted like a fixed rate, same discount as SOFR)
tsofr_curve = ql.FlatForward(today, 0.0490, ql.Actual360())
tsofr_curve.enableExtrapolation()

t_curves = time.perf_counter()
print(f"\n[1] Curve bootstrap        : {(t_curves - t0)*1000:6.1f} ms")

# ── 2. Generate synthetic book ────────────────────────────────────────────────
# Instrument types: 0=cap/floor USD SOFR, 1=cap/floor USD TermSOFR, 2=swaption EUR
inst_type = rng.integers(0, 3, N)   # 0,1,2
is_cap     = rng.integers(0, 2, N).astype(bool)   # cap vs floor / payer vs receiver

# Option expiries and underlying tenors (years)
T_exp    = rng.uniform(0.25, 10.0, N)           # option expiry
tenor    = rng.choice([1,2,3,5,7,10], N)         # swap/cap tenor in years
T_end    = T_exp + tenor

# Strikes relative to forward (will be set after forward extraction)
strike_spread = rng.uniform(-0.015, 0.015, N)    # +/- 150bps moneyness
notional = rng.uniform(1e6, 5e7, N)

t_book = time.perf_counter()
print(f"[2] Book generation        : {(t_book - t_curves)*1000:6.1f} ms")

# ── 3. Curve lookup: forward rates + annuities (vectorized via QL) ────────────
#   For cap/floor caplets: forward = simple forward rate over [T_exp, T_exp+0.25]
#   For swaptions:         forward = par swap rate,  annuity = risky annuity
#
#   We precompute discount factors on numpy arrays → fast enough for 100k.

def discount_array(curve, year_fracs):
    """Vectorized DF lookup from a QuantLib curve."""
    cal_dates = [today + ql.Period(int(round(y * 365)), ql.Days) for y in year_fracs]
    return np.array([curve.discount(d) for d in cal_dates])

# QL DF vectorised for our three curves
print("   Computing discount factors (3 curves × 2 pillars)...", end=" ", flush=True)

# We need DF(T_exp) and DF(T_end) per instrument
mask_sofr  = (inst_type == 0)
mask_tsofr = (inst_type == 1)
mask_eur   = (inst_type == 2)

df_start = np.empty(N)
df_end   = np.empty(N)

for mask, curve in [(mask_sofr, sofr_curve), (mask_tsofr, tsofr_curve), (mask_eur, eur_curve)]:
    if mask.sum() == 0:
        continue
    df_start[mask] = discount_array(curve, T_exp[mask])
    df_end[mask]   = discount_array(curve, T_end[mask])

t_df = time.perf_counter()
print(f"done — {(t_df - t_book)*1000:.1f} ms")

# ── 4. Forward rate + annuity (pure numpy from DFs) ──────────────────────────
# Cap/floor: forward = (DF_start/DF_end - 1) / tenor  (approximation)
# Swaption:  forward = (DF_start - DF_end) / annuity
#            annuity  = tenor * (DF_start + DF_end) / 2  (simplified trap rule)
#
# In production you'd sum over coupon dates, but the shape is the same.

annuity = tenor * 0.5 * (df_start + df_end)
forward = (df_start - df_end) / annuity       # par swap rate / fwd rate

strike = forward + strike_spread

t_fwd = time.perf_counter()
print(f"[3] Fwd rate + annuity     : {(t_fwd - t_df)*1000:6.1f} ms  (incl. DF lookup)")

# ── 5. ZABR normal vol (vectorized closed-form, Antonov-Piterbarg approx) ────
#   Parameters: alpha, beta, nu, rho — calibrated per index/tenor bucket
#   Here we simulate per-instrument params as if from a vol surface.

# Simulated ZABR params (would come from surface calibration in prod)
alpha = rng.uniform(0.003, 0.010, N)    # normal vol ATM ~30-100bps
beta  = np.zeros(N)                     # normal backbone (β=0) — common for rates
nu    = rng.uniform(0.10, 0.60, N)      # vol-of-vol
rho   = rng.uniform(-0.40, 0.10, N)    # correlation

def zabr_normal_vol_vec(F, K, T, alpha, beta, nu, rho):
    """
    ZABR / normal-SABR closed-form normal vol approximation (Hagan et al. β=0 case).
    Fully vectorized. Returns σ_N(F,K,T).
    """
    F = np.clip(F, 1e-6, None)
    K = np.clip(K, 1e-6, None)

    atm = np.abs(F - K) < 1e-9

    # β=0: C(F) = 1 (normal backbone)
    # z = ν/α * (F - K)
    z = np.where(atm, 0.0, nu / alpha * (F - K))

    # χ(z) = log[(√(1-2ρz+z²) + z - ρ) / (1-ρ)]
    chi = np.where(
        atm, 1.0,
        np.log((np.sqrt(np.clip(1 - 2*rho*z + z**2, 1e-12, None)) + z - rho)
               / np.clip(1 - rho, 1e-12, None))
    )
    z_chi = np.where(atm, 1.0, z / np.where(np.abs(chi) < 1e-12, 1e-12, chi))

    # Leading term: σ_N ≈ α * z/χ(z)
    sigma_lead = alpha * z_chi

    # O(T) correction (standard Hagan expansion, β=0)
    correction = 1 + T * (2 - 3*rho**2) / 24 * nu**2

    return sigma_lead * correction

t5a = time.perf_counter()
sigma_n = zabr_normal_vol_vec(forward, strike, T_exp, alpha, beta, nu, rho)
sigma_n = np.clip(sigma_n, 1e-6, None)   # floor at 0.01bps
t5b = time.perf_counter()
print(f"[4] ZABR normal vol (100k) : {(t5b - t5a)*1000:6.1f} ms")

# ── 6. Bachelier price + 4 Greeks (vectorized) ───────────────────────────────
t6a = time.perf_counter()

sqrt_T = np.sqrt(T_exp)
d      = (forward - strike) / (sigma_n * sqrt_T)
pdf_d  = norm.pdf(d)
cdf_d  = ndtr(d)            # scipy.special — faster than norm.cdf

# Payer swaption / cap = call on forward rate
call_price = notional * annuity * ((forward - strike) * cdf_d + sigma_n * sqrt_T * pdf_d)
put_price  = notional * annuity * ((strike - forward) * ndtr(-d) + sigma_n * sqrt_T * pdf_d)
price      = np.where(is_cap, call_price, put_price)

delta = notional * annuity * np.where(is_cap, cdf_d, cdf_d - 1)
gamma = notional * annuity * pdf_d / (sigma_n * sqrt_T)
vega  = notional * annuity * sqrt_T * pdf_d     # dV/dσ_N
theta = -0.5 * notional * annuity * sigma_n * pdf_d / sqrt_T

t6b = time.perf_counter()
print(f"[5] Bachelier px + Greeks  : {(t6b - t6a)*1000:6.1f} ms")

# ── Summary ───────────────────────────────────────────────────────────────────
t_total = t6b - t0
t_compute = t5b - t5a + t6b - t6a

print(f"\n{'─'*52}")
print(f"  Total (incl. curve bootstrap)  : {t_total*1000:6.1f} ms")
print(f"  Pure compute (ZABR + Bachelier): {t_compute*1000:6.1f} ms")
print(f"  DF lookup overhead (100k×2)    : {(t_df - t_book)*1000:6.1f} ms  ← bottleneck")
print(f"{'─'*52}")
print(f"\nSample results (idx=0, {'cap' if is_cap[0] else 'floor'}, type={inst_type[0]}):")
print(f"  Forward : {forward[0]*100:.4f}%   Strike: {strike[0]*100:.4f}%")
print(f"  σ_N     : {sigma_n[0]*100:.4f}%   T={T_exp[0]:.2f}y")
print(f"  Price   : {price[0]:>12,.0f}   Delta: {delta[0]:>12,.0f}")
print(f"  Vega    : {vega[0]:>12,.0f}   Theta: {theta[0]:>12,.0f}")
print(f"\nBook PV  : {price.sum():>18,.0f}")
print(f"Net Delta: {delta.sum():>18,.0f}")

print(f"""
─── What drives timing ───────────────────────────────────────────────────────
  ZABR formula  : O(N) vectorized — negligible vs pure Bachelier
  QuantLib DF   : Python loop over 100k dates — this IS the bottleneck
  Fix in prod   : interpolate your own log-linear DF array (numpy), call QL
                  only to rebuild after a curve shift (~few ms)
──────────────────────────────────────────────────────────────────────────────
""")
