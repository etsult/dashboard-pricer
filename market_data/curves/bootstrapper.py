"""
Curve stripping / bootstrapping for multi-curve IRD pricing.

OISBootstrapper
  Strips an OIS discount curve from OIS swap par rates.
  Inputs: {tenor: rate} where tenors ≤ 1Y are deposits, > 1Y are OIS swap par rates.

IBORBootstrapper
  Strips a floating-index projection curve from a mix of:
    - Money market deposits  {T: rate}
    - FRA forward rates      {(T1, T2): rate}
    - IR futures prices      {(T1, T2): price}   (100 − rate; convexity adj pre-applied)
    - IRS par rates          {T: swap_par_rate}   (fixed vs the index, OIS-discounted)

Both return a RateCurve compatible with CurveSet.

Multi-curve bootstrap reference
─────────────────────────────────
Andersen & Piterbarg, "Interest Rate Modeling", Vol. 1 Ch. 6.
For the IRS bootstrap under dual-curve discounting the float leg PV is:

  Float PV = Σᵢ [ P_proj(t_{i-1}) / P_proj(tᵢ) − 1 ] × P_ois(tᵢ)

At par (Float PV = Fixed PV = s × ann_ois):

  P_proj(T_N) = P_proj(T_{N-1}) × P_ois(T_N) / (s × ann_ois − A + P_ois(T_N))

where A = float PV of already-known caplets (t_1 … t_{N-1}).
"""

from __future__ import annotations

import numpy as np
from market_data.curves.rate_curve import RateCurve


# ══════════════════════════════════════════════════════════════════════════════
# OIS bootstrapper
# ══════════════════════════════════════════════════════════════════════════════

class OISBootstrapper:
    """
    Bootstrap an OIS discount curve from OIS swap par rates.

    Parameters
    ----------
    quotes : {tenor_years: rate_decimal}
        Tenors ≤ period  — treated as deposit rates (direct zero conversion).
        Tenors >  period — treated as OIS swap par rates (fixed vs overnight).
    freq   : coupon frequency of fixed leg (per year).
              Default 1 (annual, standard for SOFR, €STR, SONIA OIS).
    """

    def __init__(self, quotes: dict[float, float], freq: float = 1.0):
        self._tenors = np.array(sorted(quotes.keys()), dtype=float)
        self._rates  = np.array([quotes[t] for t in self._tenors], dtype=float)
        self._dt     = 1.0 / freq   # coupon period in years

    def build(self) -> RateCurve:
        """Strip and return a RateCurve of continuously-compounded OIS zero rates."""
        zeros = self._strip()
        return RateCurve._from_zeros(self._tenors, zeros)

    def _strip(self) -> np.ndarray:
        zeros = np.zeros_like(self._rates)

        for i, (T, r) in enumerate(zip(self._tenors, self._rates)):
            if T <= self._dt + 1e-9:
                # Deposit: convert simple rate to continuously-compounded zero
                zeros[i] = np.log(1.0 + r * T) / max(T, 1e-9)
            else:
                # OIS swap: fixed_rate × Σ df(t_k) × dt = 1 − df(T)  (par floater)
                # Bootstrap sequentially given all previous zeros.
                coupon_dates = np.arange(self._dt, T + 1e-9, self._dt)
                coupon_dates[-1] = T

                pv_fixed_intermediate = 0.0
                for t_c in coupon_dates[:-1]:
                    z_c = float(np.interp(t_c, self._tenors[:i], zeros[:i]))
                    pv_fixed_intermediate += r * np.exp(-z_c * t_c) * self._dt

                # 1 - pv_fixed_intermediate = df(T) × (1 + r × dt)  [last coupon + principal]
                df_T = (1.0 - pv_fixed_intermediate) / (1.0 + r * self._dt)
                df_T = max(df_T, 1e-9)
                zeros[i] = -np.log(df_T) / T

        return zeros


# ══════════════════════════════════════════════════════════════════════════════
# IBOR / term-rate projection curve bootstrapper
# ══════════════════════════════════════════════════════════════════════════════

class IBORBootstrapper:
    """
    Bootstrap a floating-index projection curve from market instruments.

    Instruments are processed in order of increasing maturity:
      1. Deposits   (short end, direct zero rates)
      2. FRAs       (mid range, forward rates → zeros)
      3. Futures    (mid range, same as FRAs; convexity adj must be pre-applied)
      4. IRS        (long end, sequential dual-curve bootstrap)

    Parameters
    ----------
    ois_curve : RateCurve — OIS discount curve (already stripped)
    deposits  : {T: deposit_rate}
    fras      : {(T1, T2): fra_rate}   — simply-compounded forward rate
    futures   : {(T1, T2): price}      — 100 − rate (convexity adjusted)
    swaps     : {T: par_swap_rate}     — fixed vs this index, OIS-discounted
    freq      : floating leg frequency (payments per year, e.g. 4 for quarterly)
    """

    def __init__(
        self,
        ois_curve: RateCurve,
        deposits:  dict[float, float]         | None = None,
        fras:      dict[tuple[float,float], float] | None = None,
        futures:   dict[tuple[float,float], float] | None = None,
        swaps:     dict[float, float]         | None = None,
        freq:      float = 4.0,
    ):
        self._ois   = ois_curve
        self._deps  = sorted((deposits or {}).items())
        self._fras  = sorted((fras     or {}).items(), key=lambda kv: kv[0][1])
        self._futs  = sorted((futures  or {}).items(), key=lambda kv: kv[0][1])
        self._swaps = sorted((swaps    or {}).items())
        self._dt    = 1.0 / freq

        # Running pillar set (time → zero rate on projection curve)
        # T=0 anchor: projection DF(0) = 1 → z(0) = 0
        self._T_pil: list[float] = [0.0]
        self._z_pil: list[float] = [0.0]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _proj_df(self, T: float) -> float:
        """Current best estimate of projection DF at T (interpolated from pillars)."""
        if T <= 1e-9:
            return 1.0
        z = float(np.interp(T, self._T_pil, self._z_pil))
        return float(np.exp(-z * T))

    def _add_pillar(self, T: float, df: float) -> None:
        df = max(float(df), 1e-9)
        z  = -np.log(df) / max(float(T), 1e-9)
        idx = int(np.searchsorted(self._T_pil, float(T)))
        self._T_pil.insert(idx, float(T))
        self._z_pil.insert(idx, float(z))

    # ── Bootstrap ─────────────────────────────────────────────────────────────

    def build(self) -> RateCurve:
        """Strip and return a projection RateCurve."""

        # 1. Deposits → direct zero rates
        for T, r in self._deps:
            z  = np.log(1.0 + r * T) / max(T, 1e-9)
            self._add_pillar(T, float(np.exp(-z * T)))

        # 2. FRAs: f(T1, T2) → df_proj(T2) = df_proj(T1) / (1 + f × dt)
        for (T1, T2), f in self._fras:
            df1 = self._proj_df(T1)
            dt  = max(T2 - T1, 1e-9)
            self._add_pillar(T2, df1 / (1.0 + f * dt))

        # 3. Futures (pre-convexity-adjusted; same formula as FRA)
        for (T1, T2), price in self._futs:
            f   = (100.0 - price) / 100.0
            df1 = self._proj_df(T1)
            dt  = max(T2 - T1, 1e-9)
            self._add_pillar(T2, df1 / (1.0 + f * dt))

        # 4. IRS par rates: dual-curve sequential bootstrap
        #    Float PV = Σᵢ (p_{i-1}/pᵢ − 1) × d_i  where pᵢ = proj_df(tᵢ), dᵢ = ois_df(tᵢ)
        #    At par: s × ann_ois = Float PV
        #    Solve for last pillar: p_N = p_{N-1} × d_N / (s × ann_ois − A + d_N)
        for T_N, s in self._swaps:
            coupon_dates = np.arange(self._dt, T_N + 1e-9, self._dt)
            coupon_dates[-1] = T_N  # pin last coupon to exact maturity

            # Fixed leg annuity (OIS discounting)
            ann_ois = sum(
                self._ois.discount_factor(t) * self._dt for t in coupon_dates
            )

            # Float leg PV from already-known projection DFs (all but last coupon)
            A      = 0.0
            p_prev = 1.0   # df_proj(0) = 1
            for t_i in coupon_dates[:-1]:
                p_i = self._proj_df(t_i)
                d_i = self._ois.discount_factor(t_i)
                A      += (p_prev / max(p_i, 1e-12) - 1.0) * d_i
                p_prev  = p_i
            # p_prev is now df_proj(coupon_dates[-2]) = p_{N-1}

            d_N  = self._ois.discount_factor(T_N)
            denom = s * ann_ois - A + d_N
            denom = max(denom, 1e-12)
            df_N  = p_prev * d_N / denom
            self._add_pillar(T_N, df_N)

        # Return RateCurve excluding the T=0 anchor
        Ts = np.array(self._T_pil[1:], dtype=float)
        zs = np.array(self._z_pil[1:], dtype=float)
        return RateCurve._from_zeros(Ts, zs)
