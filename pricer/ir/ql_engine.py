"""
QuantLib benchmark pricer for IR books (options + swaps).

Uses ql.bachelierBlackFormula + exact coupon schedules + OIS discounting
with per-index projection curves.  Used as validation benchmark against
FastBookEngine — ~20× slower due to Python loops over positions.

Accepts either a RateCurve (auto-wrapped as CurveSet) or a CurveSet.

What this validates vs FastBookEngine
──────────────────────────────────────
  ✓ Annuity: exact coupon-date sum vs any remaining approximation
  ✓ Cap strip: full caplet schedule
  ✓ Bachelier formula: ql.bachelierBlackFormula vs scipy ndtr
  ✓ Multi-curve: projection curve used for forwards, OIS for discounting
  ✗ Calendar / business-day adjustments (both use continuous-time years)
  ✗ Day-count conventions (both use ACT/365-equivalent continuous compounding)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import QuantLib as ql
    _HAS_QL = True
except ImportError:
    _HAS_QL = False

from market_data.curves.rate_curve import RateCurve
from market_data.curves.curve_set import CurveSet
from .instruments import Book
from .indexes import INDEX_CATALOG


# ── Guard ─────────────────────────────────────────────────────────────────────

def _require_ql() -> None:
    if not _HAS_QL:
        raise ImportError(
            "QuantLib is not installed.\n"
            "Install it with:  pip install QuantLib\n"
            "Then restart the Python session."
        )


def ql_available() -> bool:
    return _HAS_QL


# ── Curve helpers ─────────────────────────────────────────────────────────────

class _CurveInterp:
    """Thin wrapper around CurveSet for scalar lookups (used in Python loop)."""

    def __init__(self, cs: CurveSet):
        self._cs = cs

    def df_ois(self, T: float) -> float:
        z = float(np.interp(T, self._cs.ois._tenors, self._cs.ois._zero_rates))
        return float(np.exp(-z * T))

    # backward-compat alias
    def df(self, T: float) -> float:
        return self.df_ois(T)

    def df_proj(self, index_key: str, T: float) -> float:
        return self._cs.proj_df(index_key, float(T))

    def forward(self, index_key: str, T1: float, T2: float) -> float:
        """Simply-compounded forward rate on projection curve."""
        df1 = self.df_proj(index_key, T1)
        df2 = self.df_proj(index_key, T2)
        dt  = max(T2 - T1, 1e-8)
        return max((df1 / max(df2, 1e-12) - 1.0) / dt, 1e-5)


# ── Exact annuity ─────────────────────────────────────────────────────────────

def _exact_annuity(crv: _CurveInterp, T_start: float, T_end: float, freq: float) -> float:
    """Σ df_OIS(tᵢ) × freq over all coupon dates from T_start to T_end."""
    n = max(int(round((T_end - T_start) / freq)), 1)
    coupon_times = [T_start + (i + 1) * freq for i in range(n)]
    coupon_times[-1] = T_end
    return sum(crv.df_ois(t) * freq for t in coupon_times)


# ── Bachelier via QuantLib ─────────────────────────────────────────────────────

def _ql_bachelier(F: float, K: float, sigma_n: float, tau: float,
                  is_call: bool, discount: float) -> float:
    opt_type = ql.Option.Call if is_call else ql.Option.Put
    vol_t    = sigma_n * np.sqrt(max(tau, 1e-8))
    return ql.bachelierBlackFormula(opt_type, K, F, vol_t, discount)


# ── Main benchmark engine ─────────────────────────────────────────────────────

class QLBookEngine:
    """
    QuantLib benchmark pricer (options + vanilla IRS).

    Parameters
    ----------
    curve_or_set : RateCurve | CurveSet
        RateCurve is auto-wrapped as CurveSet (flat basis fallback).
    book : Book
    """

    def __init__(self, curve_or_set: "RateCurve | CurveSet", book: Book):
        _require_ql()
        if isinstance(curve_or_set, RateCurve):
            cs = CurveSet.from_single_curve(curve_or_set)
        else:
            cs = curve_or_set
        self._crv  = _CurveInterp(cs)
        self._book = book

    # ── Pricers ───────────────────────────────────────────────────────────────

    def _price_swaption(self, p) -> tuple[float, float]:
        meta     = INDEX_CATALOG[p.index_key]
        freq     = 1.0 / meta["reset_freq"]
        T_exp    = p.expiry_y
        T_end    = T_exp + p.tenor_y
        is_payer = p.instrument == "payer_swaption"

        ann = _exact_annuity(self._crv, T_exp, T_end, freq)
        ann = max(ann, 1e-10)

        df_exp = self._crv.df_proj(p.index_key, T_exp)
        df_end = self._crv.df_proj(p.index_key, T_end)
        F_sw   = (df_exp - df_end) / ann

        pv_unit = _ql_bachelier(float(F_sw), p.strike, p.sigma_n, T_exp, is_payer, float(ann))
        return pv_unit * p.notional * p.direction, float(F_sw)

    def _price_capfloor(self, p) -> tuple[float, float]:
        meta   = INDEX_CATALOG[p.index_key]
        freq   = 1.0 / meta["reset_freq"]
        is_cap = p.instrument == "cap"
        n_cl   = max(int(round(p.tenor_y / freq)), 1)

        total_pv = 0.0
        F_first  = None
        for i in range(n_cl):
            T_reset = (i + 1) * freq
            T_pay   = T_reset + freq
            F_cl    = self._crv.forward(p.index_key, T_reset, T_pay)
            if F_first is None:
                F_first = F_cl
            df_pay = self._crv.df_ois(T_pay)
            total_pv += _ql_bachelier(
                float(F_cl), p.strike, p.sigma_n, T_reset, is_cap,
                float(freq * df_pay),
            )

        return total_pv * p.notional * p.direction, float(F_first or 0.0)

    def _price_irs(self, p) -> tuple[float, float]:
        meta     = INDEX_CATALOG[p.index_key]
        freq     = 1.0 / meta["reset_freq"]
        T_start  = p.expiry_y
        T_end    = T_start + p.tenor_y
        is_payer = p.instrument == "payer_irs"

        ann    = _exact_annuity(self._crv, T_start, T_end, freq)
        ann    = max(ann, 1e-10)
        df_s   = self._crv.df_proj(p.index_key, T_start)
        df_e   = self._crv.df_proj(p.index_key, T_end)
        F_sw   = (df_s - df_e) / ann

        sign   = 1.0 if is_payer else -1.0
        pv     = sign * (F_sw - p.strike) * ann * p.notional * p.direction
        return pv, float(F_sw)

    # ── Public interface ──────────────────────────────────────────────────────

    def price_book(self) -> pd.DataFrame:
        if self._book.is_empty():
            return pd.DataFrame()

        rows = []
        for p in self._book.positions:
            meta = INDEX_CATALOG[p.index_key]
            instr = p.instrument
            if instr in ("payer_swaption", "receiver_swaption"):
                pv, F_fwd = self._price_swaption(p)
            elif instr in ("cap", "floor"):
                pv, F_fwd = self._price_capfloor(p)
            else:
                pv, F_fwd = self._price_irs(p)

            rows.append({
                "label":      p.label,
                "instrument": instr,
                "index_key":  p.index_key,
                "ccy":        meta["ccy"],
                "expiry_y":   p.expiry_y,
                "tenor_y":    p.tenor_y,
                "strike_pct": p.strike * 100,
                "atm_pct":    F_fwd * 100,
                "sigma_bps":  p.sigma_n * 10_000,
                "notional":   p.notional,
                "direction":  p.direction,
                "pv":         pv,
            })

        return pd.DataFrame(rows)

    def compare(self, fast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare QL prices against FastBookEngine output.

        Returns columns: pv_fast, pv_ql, diff, diff_bps, diff_pct.
        """
        ql_df = self.price_book()
        out   = fast_df[["label", "instrument", "notional", "pv"]].copy()
        out   = out.rename(columns={"pv": "pv_fast"})
        out["pv_ql"]    = ql_df["pv"].values
        out["diff"]     = out["pv_ql"] - out["pv_fast"]
        out["diff_bps"] = out["diff"] / out["notional"].abs() * 10_000
        out["diff_pct"] = out["diff"] / out["pv_fast"].abs().replace(0, np.nan) * 100
        return out
