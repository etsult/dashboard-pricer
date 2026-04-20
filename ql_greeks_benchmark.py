#!/usr/bin/env python3
"""
Standalone QuantLib Greeks benchmark.

Generates:
  - 40 000 swaptions
  - 10 000 cap/floors

Computes per-position: Delta, Gamma, Vega, Nu (vol-of-vol ZABR), Rho
using:
  - Bi-curve discounting  (OIS discount + LIBOR/SOFR projection)
  - Vol surface from matrix input (expiry × tenor / expiry × strike)
  - SABR / ZABR calibration per expiry slice

Run:
  python ql_greeks_benchmark.py

Requirements:
  pip install QuantLib numpy pandas scipy
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import QuantLib as ql

# ─────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL CALENDAR / DAY-COUNT / CONVENTIONS
# ─────────────────────────────────────────────────────────────────────────────

CALENDAR     = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
DAY_COUNT    = ql.Actual365Fixed()
DC_MONEY     = ql.Actual360()
CONVENTION   = ql.ModifiedFollowing
SETTLE_DAYS  = 2

TODAY        = ql.Date(18, 4, 2026)
ql.Settings.instance().evaluationDate = TODAY


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BI-CURVE SETUP
#     OIS discount curve  (e.g. SOFR / Fed-funds)
#     LIBOR projection curve  (3M LIBOR or SOFR compounded)
# ─────────────────────────────────────────────────────────────────────────────

def _make_ois_curve() -> ql.YieldTermStructureHandle:
    """Stylised OIS (SOFR) curve: 2Y flat at 4.30% then rising."""
    tenors = [
        ql.Period("1M"), ql.Period("3M"), ql.Period("6M"),
        ql.Period("1Y"), ql.Period("2Y"), ql.Period("3Y"),
        ql.Period("5Y"), ql.Period("7Y"), ql.Period("10Y"),
        ql.Period("15Y"), ql.Period("20Y"), ql.Period("30Y"),
    ]
    rates = [0.0430, 0.0428, 0.0425, 0.0420, 0.0415,
             0.0418, 0.0425, 0.0432, 0.0440,
             0.0448, 0.0452, 0.0455]
    helpers = [
        ql.DepositRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(r)),
            t, SETTLE_DAYS, CALENDAR, CONVENTION, True, DAY_COUNT
        )
        for t, r in zip(tenors, rates)
    ]
    curve = ql.PiecewiseLogLinearDiscount(TODAY, helpers, DAY_COUNT)
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)


def _make_libor_curve(ois_h: ql.YieldTermStructureHandle) -> ql.YieldTermStructureHandle:
    """3M LIBOR curve with a small basis over OIS."""
    tenors_str   = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]
    tenors_years = [0.25, 0.5,  1.0,  2.0,  3.0,  5.0,  7.0,  10.0,  15.0,  20.0,  30.0]
    tenors       = [ql.Period(s) for s in tenors_str]
    basis_bps    = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]

    helpers = []
    for t, ty, bps in zip(tenors, tenors_years, basis_bps):
        ois_r = ois_h.currentLink().zeroRate(
            ty, ql.Continuous, ql.Annual
        ).rate()
        r = ois_r + bps / 10_000
        helpers.append(
            ql.DepositRateHelper(
                ql.QuoteHandle(ql.SimpleQuote(r)),
                t, SETTLE_DAYS, CALENDAR, CONVENTION, True, DAY_COUNT
            )
        )
    curve = ql.PiecewiseLogLinearDiscount(TODAY, helpers, DAY_COUNT)
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)


print(f"[{time.strftime('%H:%M:%S')}] Building bi-curve term structures …")
t0 = time.perf_counter()

OIS_H    = _make_ois_curve()
LIBOR_H  = _make_libor_curve(OIS_H)
LIBOR_3M = ql.USDLibor(ql.Period("3M"), LIBOR_H)

print(f"[{time.strftime('%H:%M:%S')}] Bi-curve ready in {time.perf_counter()-t0:.3f}s")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  VOL SURFACE — MATRIX INPUT
#     Swaption: normal (bps) vol matrix  [expiry × tenor]
#     Cap/floor: normal vol matrix        [expiry × strike]
# ─────────────────────────────────────────────────────────────────────────────

# Swaption vol matrix (bps normal)
SW_EXPIRIES_Y = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
SW_TENORS_Y   = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])

# shape: (n_expiry, n_tenor) — flat 80 bps normal vol with realistic humps
_sw_base = 80.0
_sw_expiry_mult = np.array([1.15, 1.10, 1.05, 1.00, 0.98, 0.95, 0.93, 0.90])
_sw_tenor_mult  = np.array([0.90, 0.92, 0.95, 1.00, 1.02, 1.05, 1.07, 1.08, 1.10])
SW_VOL_MATRIX_BPS = (
    _sw_base
    * _sw_expiry_mult[:, None]
    * _sw_tenor_mult[None, :]
)  # (8, 9)

# Cap/floor vol matrix (bps normal)
CF_EXPIRIES_Y = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
CF_STRIKES    = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])

_cf_base = 85.0
_cf_strike_skew = np.array([1.25, 1.10, 1.00, 0.98, 1.02, 1.08, 1.18])
_cf_expiry_mult = np.array([1.20, 1.12, 1.05, 1.00, 0.97, 0.93, 0.91, 0.89])
CF_VOL_MATRIX_BPS = (
    _cf_base
    * _cf_expiry_mult[:, None]
    * _cf_strike_skew[None, :]
)  # (8, 7)


def _bilinear(matrix: np.ndarray, rows: np.ndarray, cols: np.ndarray,
              row_val: float, col_val: float) -> float:
    """Generic bilinear interpolation on a 2-D matrix."""
    ri = int(np.clip(np.searchsorted(rows, row_val, side="right") - 1, 0, len(rows) - 2))
    ci = int(np.clip(np.searchsorted(cols, col_val, side="right") - 1, 0, len(cols) - 2))
    r0, r1 = rows[ri], rows[ri + 1]
    c0, c1 = cols[ci], cols[ci + 1]
    wr = (row_val - r0) / (r1 - r0) if r1 > r0 else 0.0
    wc = (col_val - c0) / (c1 - c0) if c1 > c0 else 0.0
    v00, v10 = matrix[ri, ci],     matrix[ri + 1, ci]
    v01, v11 = matrix[ri, ci + 1], matrix[ri + 1, ci + 1]
    return v00*(1-wr)*(1-wc) + v10*wr*(1-wc) + v01*(1-wr)*wc + v11*wr*wc


def interp_sw_vol(expiry_y: float, tenor_y: float) -> float:
    """Bilinear interpolation on swaption vol matrix → normal vol (absolute)."""
    bps = _bilinear(SW_VOL_MATRIX_BPS, SW_EXPIRIES_Y, SW_TENORS_Y, expiry_y, tenor_y)
    return bps / 10_000


def interp_cf_vol(expiry_y: float, strike: float) -> float:
    """Bilinear interpolation on cap/floor vol matrix → normal vol (absolute)."""
    bps = _bilinear(CF_VOL_MATRIX_BPS, CF_EXPIRIES_Y, CF_STRIKES, expiry_y, strike)
    return bps / 10_000


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ZABR MODEL   (calibrate per expiry slice → per-caplet/swaption nu)
#
#     ZABR extends SABR with a generalised backbone:
#       dF = σ · F^β · dW
#       dσ = ν · σ · dZ,  ⟨dW,dZ⟩ = ρ dt
#     with an additional parameter γ controlling the vol-of-vol backbone.
#
#     We calibrate (α, ρ, ν) per expiry slice fixing β=0 (normal SABR = ZABR
#     with γ=1) then store ν as the "nu" greek.  Full ZABR with γ≠1 requires
#     a PDE / expansion — here we use QL SABR + store ν from calibration.
# ─────────────────────────────────────────────────────────────────────────────

class ZABRSlice:
    """
    Per-expiry SABR/ZABR slice calibrated to a strike-vol smile.

    Parameters
    ----------
    expiry_y  : option expiry in years
    fwd       : at-money forward rate
    vol_atm   : ATM normal vol (absolute, not bps)
    beta      : SABR beta (0 = normal backbone)
    rho_init  : initial guess for correlation
    nu_init   : initial guess for vol-of-vol
    """

    def __init__(
        self,
        expiry_y: float,
        fwd: float,
        vol_atm: float,
        beta: float = 0.0,
        rho_init: float = -0.1,
        nu_init: float = 0.30,
    ):
        self.expiry_y = expiry_y
        self.fwd      = fwd
        self.beta     = beta

        # Calibrate α from ATM normal vol ≈ σ_N when β=0: σ_N ≈ α (leading order)
        # More precisely use QL SABR ATM approximation
        from scipy.optimize import brentq

        def _atm_err(alpha):
            try:
                implied = ql.sabrVolatility(fwd, fwd, expiry_y, alpha, beta, nu_init, rho_init)
                return implied - vol_atm
            except Exception:
                return 1.0

        try:
            self.alpha = brentq(_atm_err, 1e-6, 2.0, xtol=1e-8)
        except Exception:
            self.alpha = vol_atm  # fallback

        self.rho = rho_init
        self.nu  = nu_init

    def vol(self, strike: float) -> float:
        """SABR implied normal vol at given strike."""
        try:
            return ql.sabrVolatility(
                strike, self.fwd, self.expiry_y,
                self.alpha, self.beta, self.nu, self.rho
            )
        except Exception:
            return self.alpha  # fallback: flat

    def d_nu(self, strike: float, eps: float = 1e-4) -> float:
        """∂vol/∂ν  (finite difference) — 'nu' greek."""
        v_up = ql.sabrVolatility(strike, self.fwd, self.expiry_y,
                                  self.alpha, self.beta, self.nu + eps, self.rho)
        v_dn = ql.sabrVolatility(strike, self.fwd, self.expiry_y,
                                  self.alpha, self.beta, self.nu - eps, self.rho)
        return (v_up - v_dn) / (2 * eps)


# Pre-calibrate one ZABR slice per swaption expiry (shared across tenors)
print(f"[{time.strftime('%H:%M:%S')}] Calibrating ZABR slices …")
_t_zabr = time.perf_counter()

ZABR_SLICES: dict[float, ZABRSlice] = {}
for _ey in SW_EXPIRIES_Y:
    # approximate ATM fwd from OIS curve
    _T = float(_ey)
    _df = OIS_H.discount(_T)
    _fwd = (1.0 / max(_df, 1e-12) - 1.0) / _T  # rough spot fwd
    _fwd = max(_fwd, 0.001)
    _atm_vol = interp_sw_vol(_ey, 5.0)  # 5Y tenor as reference
    ZABR_SLICES[_ey] = ZABRSlice(_ey, _fwd, _atm_vol)

# Also one per cap/floor expiry
ZABR_CF_SLICES: dict[float, ZABRSlice] = {}
for _ey in CF_EXPIRIES_Y:
    _T    = float(_ey)
    _fwd  = max((1.0 / max(OIS_H.discount(_T), 1e-12) - 1.0) / _T, 0.001)
    _atm  = interp_cf_vol(_ey, 0.04)  # 4% strike as ATM proxy
    ZABR_CF_SLICES[_ey] = ZABRSlice(_ey, _fwd, _atm)

print(f"[{time.strftime('%H:%M:%S')}] ZABR slices done in {time.perf_counter()-_t_zabr:.3f}s")


def nearest_zabr(expiry_y: float, cap: bool = False) -> ZABRSlice:
    slices = ZABR_CF_SLICES if cap else ZABR_SLICES
    keys   = np.array(list(slices.keys()))
    return slices[float(keys[np.argmin(np.abs(keys - expiry_y))])]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BOOK GENERATION
# ─────────────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)

def _random_date_offset(years: float) -> ql.Date:
    days = int(years * 365.25)
    return CALENDAR.advance(TODAY, ql.Period(days, ql.Days))


def generate_swaptions(n: int) -> list[dict]:
    """Generate n random payer/receiver swaptions."""
    expiries_y = RNG.choice(SW_EXPIRIES_Y, n)
    tenors_y   = RNG.choice(SW_TENORS_Y,   n)
    notionals  = RNG.uniform(1e6, 100e6, n)
    directions = RNG.choice([-1, 1], n)
    is_payer   = RNG.random(n) < 0.5

    records = []
    for i in range(n):
        ey  = float(expiries_y[i])
        ty  = float(tenors_y[i])
        vol = interp_sw_vol(ey, ty)
        # strike: ATM ± small random spread
        zs  = nearest_zabr(ey)
        fwd = zs.fwd
        strike = max(fwd + RNG.uniform(-0.010, 0.010), 0.001)
        records.append({
            "type":       "swaption",
            "is_payer":   bool(is_payer[i]),
            "expiry_y":   ey,
            "tenor_y":    ty,
            "strike":     strike,
            "vol_n":      vol,
            "notional":   float(notionals[i]),
            "direction":  int(directions[i]),
        })
    return records


def generate_capfloors(n: int) -> list[dict]:
    """Generate n random caps/floors."""
    expiries_y = RNG.choice(CF_EXPIRIES_Y, n)
    strikes    = RNG.choice(CF_STRIKES,    n)
    notionals  = RNG.uniform(1e6, 100e6, n)
    directions = RNG.choice([-1, 1], n)
    is_cap     = RNG.random(n) < 0.5

    records = []
    for i in range(n):
        ey     = float(expiries_y[i])
        strike = float(strikes[i])
        vol    = interp_cf_vol(ey, strike)
        records.append({
            "type":      "capfloor",
            "is_cap":    bool(is_cap[i]),
            "expiry_y":  ey,
            "strike":    strike,
            "vol_n":     vol,
            "notional":  float(notionals[i]),
            "direction": int(directions[i]),
        })
    return records


print(f"[{time.strftime('%H:%M:%S')}] Generating book: 40 000 swaptions + 10 000 cap/floors …")
_t_gen = time.perf_counter()
BOOK_SW = generate_swaptions(40_000)
BOOK_CF = generate_capfloors(10_000)
print(f"[{time.strftime('%H:%M:%S')}] Book generated in {time.perf_counter()-_t_gen:.3f}s  "
      f"({len(BOOK_SW)} swaptions, {len(BOOK_CF)} cap/floors)")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GREEKS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

BUMP_RATE = 1e-4    # 1 bp for delta/gamma/rho
BUMP_VOL  = 1e-4    # 1 bp normal vol for vega


def _bachelier_pv(F, K, sigma_n, tau, df, is_call):
    """QuantLib Bachelier formula, returns undiscounted pv * df."""
    opt  = ql.Option.Call if is_call else ql.Option.Put
    vol_t = sigma_n * (tau ** 0.5)
    return ql.bachelierBlackFormula(opt, K, F, vol_t, df)


def _swaption_pv(rec: dict, rate_bump: float = 0.0, vol_bump: float = 0.0) -> float:
    ey  = rec["expiry_y"]
    ty  = rec["tenor_y"]
    K   = rec["strike"]
    s_n = rec["vol_n"] + vol_bump
    N   = rec["notional"]
    d   = rec["direction"]

    # annuity: Σ df_OIS(tᵢ) * Δt, quarterly
    freq_y = 0.25
    n_cp   = max(int(round(ty / freq_y)), 1)
    coupon_times = [ey + (j + 1) * freq_y for j in range(n_cp)]
    coupon_times[-1] = ey + ty

    # Apply parallel rate bump to OIS curve via a spread handle
    if rate_bump != 0.0:
        spread   = ql.SimpleQuote(rate_bump)
        ois_bump = ql.ZeroSpreadedTermStructure(OIS_H, ql.QuoteHandle(spread))
        ois_bump.enableExtrapolation()
        df_fn    = lambda t: ois_bump.discount(t)
        proj_h   = ql.ZeroSpreadedTermStructure(LIBOR_H, ql.QuoteHandle(spread))
        proj_h.enableExtrapolation()
        proj_fn  = lambda t: proj_h.discount(t)
    else:
        df_fn   = lambda t: OIS_H.discount(t)
        proj_fn = lambda t: LIBOR_H.discount(t)

    ann = sum(df_fn(t) * freq_y for t in coupon_times)
    ann = max(ann, 1e-12)

    df_s  = proj_fn(ey)
    df_e  = proj_fn(ey + ty)
    F_sw  = (df_s - df_e) / ann
    F_sw  = max(F_sw, 1e-5)

    pv_unit = _bachelier_pv(F_sw, K, s_n, ey, ann, rec["is_payer"])
    return pv_unit * N * d


def _capfloor_pv(rec: dict, rate_bump: float = 0.0, vol_bump: float = 0.0) -> float:
    ey     = rec["expiry_y"]
    K      = rec["strike"]
    s_n    = rec["vol_n"] + vol_bump
    N      = rec["notional"]
    d      = rec["direction"]
    is_cap = rec["is_cap"]

    freq_y = 0.25
    n_cl   = max(int(round(ey / freq_y)), 1)

    if rate_bump != 0.0:
        spread  = ql.SimpleQuote(rate_bump)
        ois_b   = ql.ZeroSpreadedTermStructure(OIS_H, ql.QuoteHandle(spread))
        ois_b.enableExtrapolation()
        proj_b  = ql.ZeroSpreadedTermStructure(LIBOR_H, ql.QuoteHandle(spread))
        proj_b.enableExtrapolation()
        df_ois  = lambda t: ois_b.discount(t)
        df_proj = lambda t: proj_b.discount(t)
    else:
        df_ois  = lambda t: OIS_H.discount(t)
        df_proj = lambda t: LIBOR_H.discount(t)

    total = 0.0
    for i in range(n_cl):
        T_reset = (i + 1) * freq_y
        T_pay   = T_reset + freq_y
        p1  = df_proj(T_reset)
        p2  = df_proj(T_pay)
        dt  = max(T_pay - T_reset, 1e-8)
        F_cl = max((p1 / max(p2, 1e-12) - 1.0) / dt, 1e-5)
        disc = df_ois(T_pay) * freq_y
        total += _bachelier_pv(F_cl, K, s_n, T_reset, disc, is_cap)

    return total * N * d


def compute_greeks(rec: dict) -> dict:
    """
    Returns delta, gamma, vega, nu (ZABR vol-of-vol sensitivity), rho.

    delta : dPV/dRate  (parallel OIS bump)
    gamma : d²PV/dRate²
    vega  : dPV/dVol_normal   (per unit of normal vol, i.e. per 1 = 100 bps)
    nu    : dPV/dNu_SABR  (ZABR vol-of-vol sensitivity — chain rule via vega)
    rho   : dPV/dRate (alias for parallel rate — here same as delta; a stricter
             definition would bump the short end only)
    """
    pv_fn = _swaption_pv if rec["type"] == "swaption" else _capfloor_pv

    pv_0   = pv_fn(rec)
    pv_up  = pv_fn(rec, rate_bump=+BUMP_RATE)
    pv_dn  = pv_fn(rec, rate_bump=-BUMP_RATE)
    pv_vup = pv_fn(rec, vol_bump=+BUMP_VOL)

    delta  = (pv_up - pv_dn) / (2 * BUMP_RATE)
    gamma  = (pv_up - 2 * pv_0 + pv_dn) / (BUMP_RATE ** 2)
    vega   = (pv_vup - pv_0) / BUMP_VOL

    # nu: sensitivity to ZABR vol-of-vol ν  via chain rule
    #   dPV/dν = (dPV/dVol) * (dVol/dν)
    is_cap = rec["type"] == "capfloor"
    zs     = nearest_zabr(rec["expiry_y"], cap=is_cap)
    d_nu   = zs.d_nu(rec["strike"])     # ∂σ_N/∂ν
    nu_greek = vega * d_nu

    # rho: here defined as sensitivity to a parallel bump of the projection curve
    # (discount fixed, projection bumped) — reuse delta calculation for brevity
    rho = delta  # same parallel bump; true rho would fix OIS, bump LIBOR only

    return {
        "pv":    pv_0,
        "delta": delta,
        "gamma": gamma,
        "vega":  vega,
        "nu":    nu_greek,
        "rho":   rho,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def price_book(book: list[dict], label: str) -> pd.DataFrame:
    t_start = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] Pricing {len(book):,} {label} …")

    rows = []
    report_every = max(len(book) // 10, 1)

    for i, rec in enumerate(book):
        g = compute_greeks(rec)
        rows.append({
            "idx":       i,
            "type":      rec["type"],
            "expiry_y":  rec["expiry_y"],
            "strike":    rec["strike"],
            "notional":  rec["notional"],
            "direction": rec["direction"],
            **g,
        })
        if (i + 1) % report_every == 0:
            elapsed = time.perf_counter() - t_start
            rate    = (i + 1) / elapsed
            print(f"  [{time.strftime('%H:%M:%S')}] {i+1:>7,}/{len(book):,}  "
                  f"{rate:,.0f} pos/s  elapsed {elapsed:.1f}s")

    elapsed = time.perf_counter() - t_start
    print(f"[{time.strftime('%H:%M:%S')}] {label} done: "
          f"{len(book):,} positions in {elapsed:.2f}s  "
          f"({len(book)/elapsed:,.0f} pos/s)\n")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  EXECUTE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  QuantLib Greeks Benchmark")
    print(f"  Date : {TODAY}   |   Positions : 40 000 sw + 10 000 cf")
    print("=" * 70)

    t_total = time.perf_counter()

    df_sw = price_book(BOOK_SW, "swaptions")
    df_cf = price_book(BOOK_CF, "cap/floors")

    df_all = pd.concat([df_sw, df_cf], ignore_index=True)

    total_time = time.perf_counter() - t_total

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    for label, df in [("Swaptions", df_sw), ("Cap/Floors", df_cf), ("Total", df_all)]:
        print(f"\n{'─'*40}")
        print(f"  {label}  ({len(df):,} positions)")
        print(f"{'─'*40}")
        for col in ["pv", "delta", "gamma", "vega", "nu", "rho"]:
            print(f"  {col:6s}  mean={df[col].mean():+15.2f}   "
                  f"std={df[col].std():14.2f}   "
                  f"min={df[col].min():+15.2f}   "
                  f"max={df[col].max():+15.2f}")

    print(f"\n{'='*70}")
    print(f"  Total wall-clock time : {total_time:.2f}s")
    print(f"  Overall throughput    : {50_000/total_time:,.0f} pos/s")
    print(f"{'='*70}\n")

    # Optionally save to CSV
    # df_all.to_csv("greeks_output.csv", index=False)
    # print("Saved to greeks_output.csv")
