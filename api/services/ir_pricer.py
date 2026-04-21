"""
Interest Rate options and swap pricing functions.

No Streamlit dependency — pure Python + numpy/scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ─── Atomic pricers ──────────────────────────────────────────────────────────

def bachelier_call(F: float, K: float, sigma_n: float, tau: float) -> float:
    """Undiscounted Bachelier (normal vol) call. sigma_n in decimal."""
    if tau <= 0:
        return max(F - K, 0.0)
    vol_t = sigma_n * np.sqrt(tau)
    if vol_t < 1e-12:
        return max(F - K, 0.0)
    d = (F - K) / vol_t
    return float((F - K) * norm.cdf(d) + vol_t * norm.pdf(d))


def bachelier_put(F: float, K: float, sigma_n: float, tau: float) -> float:
    if tau <= 0:
        return max(K - F, 0.0)
    vol_t = sigma_n * np.sqrt(tau)
    if vol_t < 1e-12:
        return max(K - F, 0.0)
    d = (F - K) / vol_t
    return float((K - F) * norm.cdf(-d) + vol_t * norm.pdf(d))


def black76_call(F: float, K: float, sigma: float, tau: float) -> float:
    """Undiscounted Black-76 call."""
    if tau <= 0:
        return max(F - K, 0.0)
    if F <= 0 or K <= 0:
        return 0.0
    vol_t = sigma * np.sqrt(tau)
    if vol_t < 1e-12:
        return max(F - K, 0.0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / vol_t
    d2 = d1 - vol_t
    return float(F * norm.cdf(d1) - K * norm.cdf(d2))


def black76_put(F: float, K: float, sigma: float, tau: float) -> float:
    if tau <= 0:
        return max(K - F, 0.0)
    if F <= 0 or K <= 0:
        return 0.0
    vol_t = sigma * np.sqrt(tau)
    if vol_t < 1e-12:
        return max(K - F, 0.0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / vol_t
    d2 = d1 - vol_t
    return float(K * norm.cdf(-d2) - F * norm.cdf(-d1))


# ─── Caplet / cap / floor ────────────────────────────────────────────────────

def price_caplet(
    F: float,
    K: float,
    sigma: float,
    tau_reset: float,
    df_pay: float,
    tau_accrual: float,
    notional: float,
    vol_type: str,
    instrument_type: str,
) -> float:
    if vol_type == "normal":
        fn = bachelier_call if instrument_type == "cap" else bachelier_put
    else:
        fn = black76_call if instrument_type == "cap" else black76_put
    return fn(F, K, sigma, tau_reset) * tau_accrual * df_pay * notional


def price_cap_floor(
    curve,
    K: float,
    maturity: float,
    freq: float,
    sigma: float,
    notional: float,
    vol_type: str,
    instrument_type: str,
    start_shift: float = 0.0,
) -> tuple[float, list[dict]]:
    """
    Returns (total_price, list_of_caplet_dicts).
    start_shift > 0 gives a forward-starting cap/floor.
    """
    T_start = start_shift
    payment_dates = np.arange(T_start + freq, T_start + maturity + 1e-9, freq)
    details: list[dict] = []
    total = 0.0

    for T_pay in payment_dates:
        T_reset   = T_pay - freq
        tau_reset = max(T_reset, 0.0)
        F_fwd     = curve.forward_rate(max(T_reset, 1e-4), T_pay)
        df_pay    = curve.discount_factor(T_pay)
        pv        = price_caplet(
            F_fwd, K, sigma, tau_reset, df_pay, freq, notional, vol_type, instrument_type
        )
        total += pv
        details.append({
            "reset_years":     round(T_reset, 4),
            "pay_years":       round(T_pay, 4),
            "fwd_rate_pct":    round(F_fwd * 100, 4),
            "discount_factor": round(df_pay, 6),
            "pv":              round(pv, 2),
        })

    return total, details


# ─── Swaption ────────────────────────────────────────────────────────────────

def price_swaption(
    curve,
    T_expiry: float,
    T_end: float,
    K: float,
    sigma: float,
    notional: float,
    vol_type: str,
    swaption_type: str,
    freq: float,
    start_shift: float = 0.0,
) -> tuple[float, float, float]:
    """
    Returns (price, par_swap_rate, annuity).
    start_shift offsets T_expiry and T_end for forward-starting swaptions.
    """
    T0  = T_expiry + start_shift
    T1  = T_end    + start_shift
    S0  = curve.par_swap_rate(T0, T1, freq)
    ann = curve.annuity(T0, T1, freq)
    tau = max(T_expiry, 0.0)

    if vol_type == "normal":
        fn = bachelier_call if swaption_type == "payer" else bachelier_put
    else:
        fn = black76_call if swaption_type == "payer" else black76_put

    price = fn(S0, K, sigma, tau) * ann * notional
    return price, S0, ann


# ─── Vanilla IRS ─────────────────────────────────────────────────────────────

def price_irs(
    curve,
    T_start: float,
    T_end: float,
    fixed_rate: float,
    fixed_freq: float,
    float_freq: float,
    notional: float,
    irs_type: str,
    basis_spread_bps: float = 0.0,
) -> tuple[float, float, float, float, float, list[dict]]:
    """
    Price a vanilla IRS (or basic XCcy with basis spread).

    Returns (price, par_swap_rate, annuity, fixed_leg_pv, float_leg_pv, leg_details).

    Float leg PV = notional × (df_start − df_end) + basis_pv
    Fixed leg PV = notional × fixed_rate × annuity
    Payer (pay fixed):   price = float_leg_pv − fixed_leg_pv
    Receiver (recv fixed): price = fixed_leg_pv − float_leg_pv
    """
    df_start = curve.discount_factor(T_start) if T_start > 1e-6 else 1.0
    df_end   = curve.discount_factor(T_end)
    ann      = curve.annuity(T_start, T_end, fixed_freq)

    basis = basis_spread_bps / 10_000.0

    # Standard float leg PV + basis carried on each float reset period
    float_payment_dates = np.arange(T_start + float_freq, T_end + 1e-9, float_freq)
    basis_pv = float(sum(
        basis * float_freq * curve.discount_factor(t) * notional
        for t in float_payment_dates
    ))
    float_leg_pv = notional * (df_start - df_end) + basis_pv
    fixed_leg_pv = notional * fixed_rate * ann

    # Par swap rate (fair fixed rate given the float leg)
    S0 = float_leg_pv / (notional * ann) if ann > 1e-12 else fixed_rate

    price = (float_leg_pv - fixed_leg_pv) if irs_type == "payer" else (fixed_leg_pv - float_leg_pv)

    # Per-period cashflow breakdown on fixed payment dates
    fixed_payment_dates = np.arange(T_start + fixed_freq, T_end + 1e-9, fixed_freq)
    leg_details: list[dict] = []
    for T_pay in fixed_payment_dates:
        T_reset    = T_pay - fixed_freq
        fwd        = curve.forward_rate(max(T_reset, 1e-4), T_pay)
        df         = curve.discount_factor(T_pay)
        fix_cf     = notional * fixed_rate * fixed_freq
        flt_cf     = notional * (fwd + basis) * fixed_freq
        net_pv_row = df * (flt_cf - fix_cf) if irs_type == "payer" else df * (fix_cf - flt_cf)
        leg_details.append({
            "pay_years":       round(T_pay, 4),
            "fixed_cashflow":  round(fix_cf, 2),
            "float_cashflow":  round(flt_cf, 2),
            "discount_factor": round(df, 6),
            "net_pv":          round(net_pv_row, 2),
        })

    return price, S0, ann, fixed_leg_pv, float_leg_pv, leg_details
