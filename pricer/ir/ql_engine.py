"""
QuantLib benchmark pricer for IR options.

Mixes QuantLib (formula + exact coupon schedules) with numpy (curve
interpolation, array aggregation).  Used exclusively as a validation
benchmark against FastBookEngine — it is ~100× slower because it
loops over positions in Python.

What this tests vs FastBookEngine
──────────────────────────────────
  ✓ Annuity accuracy   — exact coupon-date sum vs trapezoidal midpoint rule
  ✓ Cap strip accuracy — all individual caplets vs single mid-strip caplet
  ✓ Bachelier formula  — ql.bachelierBlackFormula vs our scipy ndtr impl
  ✗ Calendar / business-day adjustments (both use continuous-time T in years)
  ✗ Day-count conventions (both use ACT/365-equivalent continuous compounding)

Dependency
──────────
  pip install QuantLib        (C++ binding, ~50MB wheel on PyPI)

The module degrades gracefully: if QuantLib is not installed, importing it
raises ImportError with an install hint — no other code is affected.
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
    """Return True if QuantLib is importable."""
    return _HAS_QL


# ── Curve helpers ─────────────────────────────────────────────────────────────

class _CurveInterp:
    """Thin wrapper — interpolates RateCurve at arbitrary T (years)."""

    def __init__(self, curve: RateCurve):
        self._T  = curve._tenors
        self._z  = curve._zero_rates

    def zero(self, T: float) -> float:
        return float(np.interp(T, self._T, self._z))

    def df(self, T: float) -> float:
        return float(np.exp(-self.zero(T) * T))

    def forward(self, T1: float, T2: float, basis: float = 0.0) -> float:
        """Simply-compounded forward on the index curve (OIS + flat basis)."""
        df1 = np.exp(-(self.zero(T1) + basis) * T1)
        df2 = np.exp(-(self.zero(T2) + basis) * T2)
        dt  = max(T2 - T1, 1e-8)
        return max((df1 / max(df2, 1e-12) - 1.0) / dt, 1e-5)


# ── Exact annuity (coupon-date summation) ─────────────────────────────────────

def _exact_annuity(
    crv: _CurveInterp,
    T_start: float,
    T_end: float,
    freq: float,          # coupon period in years (e.g. 0.25 for quarterly)
) -> float:
    """
    Exact swaption annuity: Σ df(t_i) × Δt_i  over all coupon dates.

    FastBookEngine uses trapezoidal: ann ≈ (T_end − T_start) × df(T_mid).
    The exact sum matters most for long tenors on steep curves (can differ
    by 0.5–2%).
    """
    n = max(int(round((T_end - T_start) / freq)), 1)
    coupon_times = [T_start + (i + 1) * freq for i in range(n)]
    coupon_times[-1] = T_end  # pin last coupon to exact maturity
    return sum(crv.df(t) * freq for t in coupon_times)


# ── Bachelier pricing via QuantLib formula ────────────────────────────────────

def _ql_bachelier(
    F: float, K: float, sigma_n: float, tau: float,
    is_call: bool, discount_factor: float,
) -> float:
    """
    Price a single Bachelier option using ql.bachelierBlackFormula.

    Parameters
    ----------
    F            forward rate (decimal)
    K            strike (decimal)
    sigma_n      normal vol (decimal / year^0.5  ·  units: e.g. 0.006 = 60bps)
    tau          time to expiry in years
    is_call      True → call/payer/cap;  False → put/receiver/floor
    discount_factor  df(T_pay) — the price incorporates this DF
    """
    opt_type = ql.Option.Call if is_call else ql.Option.Put
    vol_t    = sigma_n * np.sqrt(max(tau, 1e-8))   # stddev = σ_N × √τ
    return ql.bachelierBlackFormula(opt_type, K, F, vol_t, discount_factor)


# ── Main benchmark engine ─────────────────────────────────────────────────────

class QLBookEngine:
    """
    QuantLib benchmark pricer.

    Uses:
      • ql.bachelierBlackFormula   for option pricing (exact Bachelier)
      • Exact coupon-date annuity  for swaptions   (vs trapezoidal in fast_engine)
      • Full caplet strip          for caps/floors  (vs mid-strip in fast_engine)

    Returns a DataFrame compatible with FastBookEngine.price_book() so the
    two can be compared column-by-column.

    Parameters
    ----------
    curve   RateCurve — OIS discount curve
    book    Book      — positions to price
    """

    def __init__(self, curve: RateCurve, book: Book):
        _require_ql()
        self._crv  = _CurveInterp(curve)
        self._book = book

    # ── Internal pricers ──────────────────────────────────────────────────

    def _price_swaption(self, p) -> tuple[float, float]:
        """Return (pv, F_swap) for one swaption position."""
        meta  = INDEX_CATALOG[p.index_key]
        basis = meta["basis_bps"] / 10_000.0
        freq  = 1.0 / meta["reset_freq"]

        T_exp = p.expiry_y
        T_end = T_exp + p.tenor_y

        # Forward swap rate on index curve
        df_exp_idx = np.exp(-(self._crv.zero(T_exp) + basis) * T_exp)
        df_end_idx = np.exp(-(self._crv.zero(T_end) + basis) * T_end)

        # EXACT annuity (key difference vs fast_engine trapezoidal)
        ann = _exact_annuity(self._crv, T_exp, T_end, freq)
        ann = max(ann, 1e-10)

        F_sw = (df_exp_idx - df_end_idx) / ann
        is_payer = p.instrument == "payer_swaption"

        # ql.bachelierBlackFormula(type, K, F, stddev, discount)
        # For swaptions "discount" = annuity × notional
        pv_unit = _ql_bachelier(
            float(F_sw), p.strike, p.sigma_n, T_exp,
            is_payer, float(ann),
        )
        return pv_unit * p.notional * p.direction, float(F_sw)

    def _price_capfloor(self, p) -> tuple[float, float]:
        """
        Sum all individual caplets/floorlets (exact strip).

        FastBookEngine uses a single representative caplet at T_mid × n_caplets.
        Here we price each caplet individually — bigger difference the more
        the vol term structure or forward curve is non-flat.
        """
        meta  = INDEX_CATALOG[p.index_key]
        basis = meta["basis_bps"] / 10_000.0
        freq  = 1.0 / meta["reset_freq"]
        is_cap = p.instrument == "cap"
        mat    = p.tenor_y

        n_caplets = max(int(round(mat / freq)), 1)
        total_pv  = 0.0
        F_first   = None   # ATM fwd of first caplet (for display)

        for i in range(n_caplets):
            T_reset = (i + 1) * freq          # fixing time
            T_pay   = T_reset + freq           # payment time

            # Forward rate for this caplet on index curve
            F_cl = self._crv.forward(T_reset, T_pay, basis)
            if F_first is None:
                F_first = F_cl

            df_pay = self._crv.df(T_pay)

            # Caplet PV — discount = freq × df(T_pay)
            pv_cl = _ql_bachelier(
                float(F_cl), p.strike, p.sigma_n, T_reset,
                is_cap, float(freq * df_pay),
            )
            total_pv += pv_cl

        return total_pv * p.notional * p.direction, float(F_first or 0.0)

    # ── Public interface ──────────────────────────────────────────────────

    def price_book(self) -> pd.DataFrame:
        """
        Price all positions. Returns DataFrame with same schema as
        FastBookEngine.price_book(), plus column 'pv_ql' for clarity.
        """
        if self._book.is_empty():
            return pd.DataFrame()

        rows = []
        for p in self._book.positions:
            meta = INDEX_CATALOG[p.index_key]
            is_sw = p.instrument in ("payer_swaption", "receiver_swaption")

            if is_sw:
                pv, F_fwd = self._price_swaption(p)
            else:
                pv, F_fwd = self._price_capfloor(p)

            rows.append({
                "label":      p.label,
                "instrument": p.instrument,
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

        Returns a DataFrame with columns:
          pv_fast   PV from FastBookEngine
          pv_ql     PV from QLBookEngine
          diff      pv_ql − pv_fast
          diff_bps  diff / notional × 10_000  (in bps of notional)
          diff_pct  diff / |pv_fast| × 100    (relative %)
        """
        ql_df   = self.price_book()
        out     = fast_df[["label", "instrument", "notional", "pv"]].copy()
        out     = out.rename(columns={"pv": "pv_fast"})
        out["pv_ql"]    = ql_df["pv"].values
        out["diff"]     = out["pv_ql"] - out["pv_fast"]
        out["diff_bps"] = out["diff"] / out["notional"].abs() * 10_000
        notional_pv     = out["pv_fast"].abs().replace(0, np.nan)
        out["diff_pct"] = out["diff"] / notional_pv * 100
        return out
