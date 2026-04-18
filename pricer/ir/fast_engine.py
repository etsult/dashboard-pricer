"""
Fast vectorized IR option book pricer.

Prices N positions simultaneously using pure numpy — no Python loops over
positions at pricing time.  Suitable for books of 10k–500k positions.

Approximations used (acceptable for portfolio risk):
  • Annuity: trapezoidal rule  ann ≈ (T_end − T_start) × df_mid
  • Cap/floor: priced as a single representative caplet at T_mid
    scaled by the number of caplets in the strip.
  • Greeks (delta, vega): Bachelier analytical.
  • DV01: Bachelier delta × annuity × notional × 1e-4 (1bp ≈ linear approx).

For small books (<500 positions) the exact engine in engine.py is preferred.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from market_data.curves.rate_curve import RateCurve
from .indexes import INDEX_CATALOG
from .instruments import Book

_SQRT_2PI = np.sqrt(2 * np.pi)

# ── Curve helpers (vectorized) ─────────────────────────────────────────────────

class _CurveGrid:
    """Pre-computed dense DF grid for O(1) vectorized lookup."""

    def __init__(self, curve: RateCurve, n: int = 20_000):
        self._T  = np.linspace(1e-4, 31.0, n)
        self._z  = np.interp(self._T, curve._tenors, curve._zero_rates)
        self._df = np.exp(-self._z * self._T)

    def df(self, T: np.ndarray) -> np.ndarray:
        return np.interp(T, self._T, self._df)

    def zero(self, T: np.ndarray) -> np.ndarray:
        return np.interp(T, self._T, self._z)

    def forward_rate(self, T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        """Simply compounded forward rate f(T1, T2)."""
        df1 = self.df(T1)
        df2 = self.df(T2)
        dt  = np.clip(T2 - T1, 1e-8, None)
        return (df1 / np.clip(df2, 1e-12, None) - 1.0) / dt


# ── Bachelier (fully vectorized) ──────────────────────────────────────────────

def _bach_price_greeks(F, K, sigma_n, tau, is_call):
    tau     = np.clip(tau, 1e-8, None)
    vol_t   = np.clip(sigma_n * np.sqrt(tau), 1e-12, None)
    d       = (F - K) / vol_t
    pdf     = np.exp(-0.5 * d ** 2) / _SQRT_2PI
    cdf     = ndtr(d)
    price   = np.where(is_call,
                       (F - K) * cdf + vol_t * pdf,
                       (K - F) * ndtr(-d) + vol_t * pdf)
    delta   = np.where(is_call, cdf, cdf - 1.0)
    vega    = np.sqrt(tau) * pdf   # ∂price/∂sigma_n
    gamma   = pdf / vol_t
    return price, delta, vega, gamma


# ── Main fast engine ──────────────────────────────────────────────────────────

class FastBookEngine:
    """
    Vectorized pricer for large IR option books.

    Usage
    -----
    eng = FastBookEngine(curve, book)
    pv_df   = eng.price_book()         # DataFrame with PV per position
    risk_df = eng.risk_book()          # DV01, vega, delta per position
    agg     = eng.aggregate_risk()     # book-level aggregated risk tables
    """

    def __init__(self, curve: RateCurve, book: Book):
        self.book  = book
        self._grid = _CurveGrid(curve)
        self._curve = curve

    # ------------------------------------------------------------------

    def _extract_arrays(self):
        """Pull all position params into numpy arrays in one pass."""
        pos = self.book.positions
        n   = len(pos)

        instr    = np.array([p.instrument for p in pos])
        idx_keys = [p.index_key  for p in pos]
        notional = np.array([p.notional   for p in pos], dtype=float)
        strike   = np.array([p.strike     for p in pos], dtype=float)
        expiry   = np.array([p.expiry_y   for p in pos], dtype=float)
        tenor    = np.array([p.tenor_y    for p in pos], dtype=float)
        sigma_n  = np.array([p.sigma_n    for p in pos], dtype=float)
        direction= np.array([p.direction  for p in pos], dtype=float)

        freq     = np.array([1.0 / INDEX_CATALOG[k]["reset_freq"] for k in idx_keys])
        basis    = np.array([INDEX_CATALOG[k]["basis_bps"] / 10_000.0 for k in idx_keys])

        is_swaption  = np.isin(instr, ["payer_swaption", "receiver_swaption"])
        is_payer     = instr == "payer_swaption"
        is_call_cap  = instr == "cap"
        is_cap_floor = np.isin(instr, ["cap", "floor"])

        return {
            "n": n, "instr": instr, "notional": notional, "strike": strike,
            "expiry": expiry, "tenor": tenor, "sigma_n": sigma_n,
            "direction": direction, "freq": freq, "basis": basis,
            "is_swaption": is_swaption, "is_payer": is_payer,
            "is_call_cap": is_call_cap, "is_cap_floor": is_cap_floor,
            "idx_keys": idx_keys,
        }

    # ------------------------------------------------------------------

    def price_book(self) -> pd.DataFrame:
        """Return per-position PV. O(N) vectorized."""
        if self.book.is_empty():
            return pd.DataFrame()

        a = self._extract_arrays()
        g = self._grid
        n = a["n"]

        pv      = np.zeros(n)
        F_fwd   = np.zeros(n)
        annuity = np.zeros(n)

        # ── Swaptions ─────────────────────────────────────────────────────────
        sw = a["is_swaption"]
        if sw.any():
            T_exp = a["expiry"][sw]
            T_end = T_exp + a["tenor"][sw]
            bas   = a["basis"][sw]
            frq   = a["freq"][sw]

            # Two-curve: index DF = OIS DF * exp(-basis * T)
            df_exp_idx = g.df(T_exp) * np.exp(-bas * T_exp)
            df_end_idx = g.df(T_end) * np.exp(-bas * T_end)

            # Trapezoidal annuity (OIS-discounted)
            T_mid  = (T_exp + T_end) / 2
            ann_sw = (T_end - T_exp) * g.df(T_mid)
            ann_sw = np.clip(ann_sw, 1e-10, None)

            F_sw   = (df_exp_idx - df_end_idx) / ann_sw

            price_sw, _, _, _ = _bach_price_greeks(
                F_sw, a["strike"][sw], a["sigma_n"][sw],
                T_exp, a["is_payer"][sw],
            )
            pv[sw]      = price_sw * ann_sw * a["notional"][sw] * a["direction"][sw]
            F_fwd[sw]   = F_sw
            annuity[sw] = ann_sw

        # ── Caps / Floors ──────────────────────────────────────────────────────
        cf = a["is_cap_floor"]
        if cf.any():
            mat  = a["tenor"][cf]           # cap maturity
            frq  = a["freq"][cf]
            bas  = a["basis"][cf]

            n_caplets = np.maximum(np.round(mat / frq), 1)  # number of caplets
            T_mid_reset = mat / 2                             # mid-strip reset time
            T_mid_pay   = T_mid_reset + frq

            # Forward rate at mid-strip on index curve
            z1 = g.zero(T_mid_reset) + bas
            z2 = g.zero(T_mid_pay)   + bas
            df1 = np.exp(-z1 * T_mid_reset)
            df2 = np.exp(-z2 * T_mid_pay)
            F_cf = np.clip((df1 / np.clip(df2, 1e-12, None) - 1.0) / frq, 1e-5, None)

            df_pay = g.df(T_mid_pay)
            tau_opt = np.clip(T_mid_reset, 1e-6, None)

            price_cf, _, _, _ = _bach_price_greeks(
                F_cf, a["strike"][cf], a["sigma_n"][cf],
                tau_opt, a["is_call_cap"][cf],
            )
            pv[cf]    = price_cf * frq * df_pay * n_caplets * a["notional"][cf] * a["direction"][cf]
            F_fwd[cf] = F_cf

        return pd.DataFrame({
            "label":       [p.label     for p in self.book.positions],
            "instrument":  a["instr"],
            "index_key":   a["idx_keys"],
            "ccy":         [INDEX_CATALOG[k]["ccy"] for k in a["idx_keys"]],
            "expiry_y":    a["expiry"],
            "tenor_y":     a["tenor"],
            "strike_pct":  a["strike"] * 100,
            "atm_pct":     F_fwd * 100,
            "sigma_bps":   a["sigma_n"] * 10_000,
            "notional":    a["notional"],
            "direction":   a["direction"],
            "pv":          pv,
        })

    # ------------------------------------------------------------------

    def risk_book(self, bump_bp: float = 1.0) -> pd.DataFrame:
        """
        Per-position DV01 and vega using analytical Greek approximation.
        Much faster than full bump-and-reprice for large books.
        """
        if self.book.is_empty():
            return pd.DataFrame()

        a  = self._extract_arrays()
        g  = self._grid
        n  = a["n"]

        dv01  = np.zeros(n)
        vega  = np.zeros(n)
        delta = np.zeros(n)
        pv    = np.zeros(n)

        bump = bump_bp / 10_000.0

        # ── Swaptions ─────────────────────────────────────────────────────────
        sw = a["is_swaption"]
        if sw.any():
            T_exp = a["expiry"][sw]
            T_end = T_exp + a["tenor"][sw]
            bas   = a["basis"][sw]

            df_exp_idx = g.df(T_exp) * np.exp(-bas * T_exp)
            df_end_idx = g.df(T_end) * np.exp(-bas * T_end)
            T_mid      = (T_exp + T_end) / 2
            ann_sw     = np.clip((T_end - T_exp) * g.df(T_mid), 1e-10, None)
            F_sw       = (df_exp_idx - df_end_idx) / ann_sw

            price_sw, delta_sw, vega_sw, _ = _bach_price_greeks(
                F_sw, a["strike"][sw], a["sigma_n"][sw],
                T_exp, a["is_payer"][sw],
            )

            scale_sw = ann_sw * a["notional"][sw] * a["direction"][sw]
            pv[sw]    = price_sw * scale_sw
            delta[sw] = delta_sw * scale_sw
            vega[sw]  = vega_sw  * scale_sw * 1e-4   # per 1bp vol
            dv01[sw]  = delta_sw * scale_sw * bump    # delta × bp = DV01

        # ── Caps / Floors ──────────────────────────────────────────────────────
        cf = a["is_cap_floor"]
        if cf.any():
            mat  = a["tenor"][cf]
            frq  = a["freq"][cf]
            bas  = a["basis"][cf]
            n_cl = np.maximum(np.round(mat / frq), 1)
            T_mr = mat / 2
            T_mp = T_mr + frq

            z1 = g.zero(T_mr) + bas
            z2 = g.zero(T_mp) + bas
            df1 = np.exp(-z1 * T_mr)
            df2 = np.exp(-z2 * T_mp)
            F_cf = np.clip((df1 / np.clip(df2, 1e-12, None) - 1.0) / frq, 1e-5, None)
            df_pay = g.df(T_mp)

            price_cf, delta_cf, vega_cf, _ = _bach_price_greeks(
                F_cf, a["strike"][cf], a["sigma_n"][cf],
                np.clip(T_mr, 1e-6, None), a["is_call_cap"][cf],
            )

            scale_cf = frq * df_pay * n_cl * a["notional"][cf] * a["direction"][cf]
            pv[cf]    = price_cf * scale_cf
            delta[cf] = delta_cf * scale_cf
            vega[cf]  = vega_cf  * scale_cf * 1e-4
            dv01[cf]  = delta_cf * scale_cf * bump

        return pd.DataFrame({
            "label":      [p.label for p in self.book.positions],
            "instrument": a["instr"],
            "index_key":  a["idx_keys"],
            "ccy":        [INDEX_CATALOG[k]["ccy"] for k in a["idx_keys"]],
            "expiry_y":   a["expiry"],
            "tenor_y":    a["tenor"],
            "notional":   a["notional"],
            "direction":  a["direction"],
            "pv":         pv,
            "delta":      delta,
            "dv01":       dv01,
            "vega":       vega,
        })

    # ------------------------------------------------------------------

    def aggregate_risk(self) -> dict[str, pd.DataFrame]:
        """
        Aggregate risk by instrument, index, CCY, and maturity bucket.
        Returns a dict of DataFrames for display.
        """
        df = self.risk_book()
        if df.empty:
            return {}

        def _agg(by):
            return (
                df.groupby(by, observed=True)[["pv", "dv01", "vega"]]
                .sum()
                .rename(columns={"pv": "PV ($)", "dv01": "DV01 ($)", "vega": "Vega/bp ($)"})
                .sort_values("PV ($)", ascending=False)
            )

        bins = [0, 1/12, 3/12, 1, 2, 5, 10, 30]
        lbls = ["<1M", "1-3M", "3M-1Y", "1-2Y", "2-5Y", "5-10Y", ">10Y"]
        df["exp_bucket"] = pd.cut(df["expiry_y"], bins=bins, labels=lbls)

        return {
            "by_instrument": _agg("instrument"),
            "by_index":      _agg("index_key"),
            "by_ccy":        _agg("ccy"),
            "by_expiry":     _agg("exp_bucket"),
        }
