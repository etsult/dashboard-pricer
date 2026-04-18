# market_data/curves/rate_curve.py
"""
Bootstrap a piecewise-linear zero rate curve from US Treasury par yields.

  - T <= 1Y : par yield treated as zero rate (coupon effect negligible)
  - T >  1Y : iterative coupon-stripping bootstrap (semi-annual coupon bonds)

Provides:
    zero_rate(T)          -> continuously compounded zero rate
    discount_factor(T)    -> exp(-z * T)
    forward_rate(T1, T2)  -> simply compounded fwd rate for period [T1, T2]
    par_swap_rate(T_start, T_end, freq) -> par swap rate given the curve
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


class RateCurve:
    """
    Zero-rate curve bootstrapped from par yields (e.g. US Treasuries from FRED).

    Parameters
    ----------
    par_yields : dict
        {tenor_years: rate_decimal}  e.g. {1.0: 0.042, 2.0: 0.041, ...}
    """

    def __init__(self, par_yields: Dict[float, float]):
        tenors = np.array(sorted(par_yields.keys()), dtype=float)
        rates  = np.array([par_yields[t] for t in tenors], dtype=float)
        self._tenors     = tenors
        self._zero_rates = self._bootstrap(tenors, rates)

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    def _bootstrap(self, tenors: np.ndarray, par_yields: np.ndarray) -> np.ndarray:
        zero_rates = np.zeros_like(par_yields)

        for i, (T, y) in enumerate(zip(tenors, par_yields)):
            if T <= 1.0:
                # Short end: convert simple par yield to continuously compounded zero
                zero_rates[i] = np.log(1.0 + y * T) / T
            else:
                # Bootstrap: semi-annual coupon bond must price at par (= 1)
                # 1 = c * Σ df(t_k)  +  df(T)   where c = y/2
                coupon      = y / 2.0
                coupon_dates = np.arange(0.5, T + 1e-9, 0.5)
                pv_coupons  = 0.0
                for t_c in coupon_dates[:-1]:  # all coupon dates except final
                    z_c = float(np.interp(t_c, tenors[:i], zero_rates[:i]))
                    pv_coupons += coupon * np.exp(-z_c * t_c)
                df_T = (1.0 - pv_coupons) / (1.0 + coupon)
                df_T = max(df_T, 1e-9)
                zero_rates[i] = -np.log(df_T) / T

        return zero_rates

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def zero_rate(self, T: float) -> float:
        """Continuously compounded zero rate for maturity T (years)."""
        T = max(float(T), 1e-6)
        return float(np.interp(T, self._tenors, self._zero_rates))

    def discount_factor(self, T: float) -> float:
        """Risk-neutral discount factor for maturity T."""
        return float(np.exp(-self.zero_rate(T) * T))

    def forward_rate(self, T1: float, T2: float) -> float:
        """
        Simply compounded forward rate for period [T1, T2].
        f(T1,T2) = (df(T1)/df(T2) - 1) / (T2 - T1)
        """
        if T2 <= T1 + 1e-9:
            raise ValueError(f"T2 ({T2}) must be > T1 ({T1})")
        df1 = self.discount_factor(T1)
        df2 = self.discount_factor(T2)
        return (df1 / df2 - 1.0) / (T2 - T1)

    def par_swap_rate(self, T_start: float, T_end: float, freq: float = 0.5) -> float:
        """
        Par swap rate: the fixed rate that makes a swap have zero value.

        S = (df(T_start) - df(T_end)) / annuity
        annuity = freq * Σ df(T_i)  for payment dates T_start+freq, ..., T_end
        """
        payment_dates = np.arange(T_start + freq, T_end + 1e-9, freq)
        annuity = freq * sum(self.discount_factor(t) for t in payment_dates)
        if annuity < 1e-10:
            return 0.0
        return (self.discount_factor(T_start) - self.discount_factor(T_end)) / annuity

    def annuity(self, T_start: float, T_end: float, freq: float = 0.5) -> float:
        """Dollar annuity (PV01 per unit notional * freq) for a swap."""
        payment_dates = np.arange(T_start + freq, T_end + 1e-9, freq)
        return freq * sum(self.discount_factor(t) for t in payment_dates)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Return curve as a tidy DataFrame for display."""
        return pd.DataFrame({
            "Tenor (Y)":          self._tenors,
            "Par Yield (%)":      np.interp(self._tenors, self._tenors, self._tenors) * 0,  # placeholder
            "Zero Rate (%)":      np.round(self._zero_rates * 100, 4),
            "Discount Factor":    np.round(np.exp(-self._zero_rates * self._tenors), 6),
        })

    def zero_curve_df(self) -> pd.DataFrame:
        """Tenor vs zero rate for plotting."""
        return pd.DataFrame({
            "Tenor": self._tenors,
            "Zero Rate (%)": self._zero_rates * 100,
        })

    # ------------------------------------------------------------------
    # Bond analytics  (Hull, Ch. 4)
    # ------------------------------------------------------------------

    def price_bond(
        self,
        coupon_rate: float,   # annual coupon rate, decimal
        maturity: float,       # years to maturity
        face: float = 100.0,
        freq: int   = 2,       # coupon payments per year
    ) -> float:
        """Dirty price of a fixed-coupon bond discounted off the zero curve."""
        period  = 1.0 / freq
        coupon  = face * coupon_rate / freq
        t_grid  = np.arange(period, maturity + 1e-9, period)
        pv      = float(sum(coupon * self.discount_factor(t) for t in t_grid))
        pv     += face * self.discount_factor(maturity)
        return pv

    def bond_yield(
        self,
        coupon_rate: float,
        maturity: float,
        face: float  = 100.0,
        freq: int    = 2,
        price: float = None,
    ) -> float:
        """
        Continuously-compounded YTM by Brent root-finding.
        If price is None, uses self.price_bond().
        """
        from scipy.optimize import brentq
        p      = price if price is not None else self.price_bond(coupon_rate, maturity, face, freq)
        period = 1.0 / freq
        coupon = face * coupon_rate / freq
        t_grid = np.arange(period, maturity + 1e-9, period)
        cfs    = [coupon] * len(t_grid)
        cfs[-1] += face

        def _pv(y: float) -> float:
            return sum(cf * np.exp(-y * t) for cf, t in zip(cfs, t_grid)) - p

        return brentq(_pv, -0.5, 5.0, xtol=1e-10)

    def macaulay_duration(
        self,
        coupon_rate: float,
        maturity: float,
        face: float = 100.0,
        freq: int   = 2,
    ) -> float:
        """Macaulay duration (years) = Σ t × PV(CF) / P."""
        p       = self.price_bond(coupon_rate, maturity, face, freq)
        period  = 1.0 / freq
        coupon  = face * coupon_rate / freq
        t_grid  = np.arange(period, maturity + 1e-9, period)
        cfs     = [coupon] * len(t_grid)
        cfs[-1] += face
        num = sum(t * cf * self.discount_factor(t) for t, cf in zip(t_grid, cfs))
        return num / p

    def modified_duration(
        self,
        coupon_rate: float,
        maturity: float,
        face: float = 100.0,
        freq: int   = 2,
    ) -> float:
        """
        Modified duration (years).
        Under continuous compounding: D_mod = D_mac (= -dP/dy / P).
        """
        return self.macaulay_duration(coupon_rate, maturity, face, freq)

    def convexity_measure(
        self,
        coupon_rate: float,
        maturity: float,
        face: float = 100.0,
        freq: int   = 2,
    ) -> float:
        """Convexity (continuous compounding) = Σ t² × PV(CF) / P."""
        p       = self.price_bond(coupon_rate, maturity, face, freq)
        period  = 1.0 / freq
        coupon  = face * coupon_rate / freq
        t_grid  = np.arange(period, maturity + 1e-9, period)
        cfs     = [coupon] * len(t_grid)
        cfs[-1] += face
        return sum(t**2 * cf * self.discount_factor(t) for t, cf in zip(t_grid, cfs)) / p

    def dv01(
        self,
        coupon_rate: float,
        maturity: float,
        face: float = 100.0,
        freq: int   = 2,
    ) -> float:
        """
        DV01 = D_mod × P × 0.0001.
        Dollar change in *full* bond price per 1 bp parallel yield shift.
        """
        p     = self.price_bond(coupon_rate, maturity, face, freq)
        d_mod = self.modified_duration(coupon_rate, maturity, face, freq)
        return d_mod * p * 0.0001

    # ------------------------------------------------------------------
    # Forward rate curve  (useful alongside zero curve)
    # ------------------------------------------------------------------

    def instantaneous_forward(self, T: float, dt: float = 1e-4) -> float:
        """Instantaneous forward rate f(T) ≈ d/dT [z(T) × T]."""
        T = max(T, dt)
        z1 = self.zero_rate(T - dt) * (T - dt)
        z2 = self.zero_rate(T + dt) * (T + dt)
        return (z2 - z1) / (2 * dt)

    def forward_curve_df(self) -> pd.DataFrame:
        """Zero, forward, and par curves at pillar tenors."""
        rows = []
        for i, T in enumerate(self._tenors):
            fwd = self.instantaneous_forward(T) if T > 1e-3 else self._zero_rates[0]
            # Par yield: y such that bond priced at par (annuity formula)
            freq   = 2 if T > 1.0 else 1
            period = 1.0 / freq
            t_grid = np.arange(period, T + 1e-9, period)
            ann    = sum(period * self.discount_factor(t) for t in t_grid) if len(t_grid) else period
            par    = (1.0 - self.discount_factor(T)) / ann if ann > 1e-10 else float(self._zero_rates[i])
            rows.append({
                "Tenor":          T,
                "Zero Rate (%)":  self._zero_rates[i] * 100,
                "Forward Rate (%)": fwd * 100,
                "Par Yield (%)":  par * 100,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Scenario curves  (Hull, Ch. 4 / risk management)
    # ------------------------------------------------------------------

    @classmethod
    def _from_zeros(cls, tenors: np.ndarray, zero_rates: np.ndarray) -> "RateCurve":
        """Construct directly from zero rates (skip bootstrap)."""
        obj              = cls.__new__(cls)
        obj._tenors      = tenors.copy()
        obj._zero_rates  = zero_rates.copy()
        return obj

    def shifted(self, bp: float) -> "RateCurve":
        """Parallel shift: all zero rates ± bp basis points."""
        return RateCurve._from_zeros(
            self._tenors,
            np.maximum(self._zero_rates + bp / 10_000.0, -0.10),
        )

    def steepened(self, front_bp: float, back_bp: float) -> "RateCurve":
        """
        Linear tilt.  Short end shifts by front_bp, long end by back_bp.
        Positive front_bp + negative back_bp = bear flattener.
        """
        T_min  = self._tenors[0]
        T_max  = self._tenors[-1]
        alpha  = (self._tenors - T_min) / max(T_max - T_min, 1e-9)
        shifts = (front_bp + alpha * (back_bp - front_bp)) / 10_000.0
        return RateCurve._from_zeros(self._tenors, self._zero_rates + shifts)

    def twisted(self, belly_bp: float) -> "RateCurve":
        """
        Butterfly twist: short/long ends unchanged, belly (5Y area) shifts by belly_bp.
        Quadratic bump centred at mid-tenor.
        """
        T_mid  = (self._tenors[0] + self._tenors[-1]) / 2.0
        T_half = (self._tenors[-1] - self._tenors[0]) / 2.0 + 1e-9
        bump   = belly_bp / 10_000.0 * np.maximum(0.0, 1.0 - ((self._tenors - T_mid) / T_half)**2)
        return RateCurve._from_zeros(self._tenors, self._zero_rates + bump)
