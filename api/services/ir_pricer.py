"""
Interest Rate options pricing functions.

These were previously embedded in pages/5_IROptions.py.
Extracting them here makes them reusable by the FastAPI router
(and by any future frontend, notebook, or test).

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
    vol_type: str,          # "normal" | "lognormal"
    instrument_type: str,   # "cap" | "floor"
) -> float:
    if vol_type == "normal":
        fn = bachelier_call if instrument_type == "cap" else bachelier_put
    else:
        fn = black76_call if instrument_type == "cap" else black76_put
    return fn(F, K, sigma, tau_reset) * tau_accrual * df_pay * notional


def price_cap_floor(
    curve,              # RateCurve instance
    K: float,
    maturity: float,
    freq: float,
    sigma: float,
    notional: float,
    vol_type: str,
    instrument_type: str,
) -> tuple[float, list[dict]]:
    """
    Returns (total_price, list_of_caplet_dicts).
    Each dict has: reset_years, pay_years, fwd_rate_pct, discount_factor, pv.
    """
    payment_dates = np.arange(freq, maturity + 1e-9, freq)
    details: list[dict] = []
    total = 0.0

    for T_pay in payment_dates:
        T_reset     = T_pay - freq
        tau_reset   = max(T_reset, 0.0)
        F_fwd       = curve.forward_rate(max(T_reset, 1e-4), T_pay)
        df_pay      = curve.discount_factor(T_pay)
        pv          = price_caplet(
            F_fwd, K, sigma, tau_reset, df_pay, freq, notional, vol_type, instrument_type
        )
        total += pv
        details.append({
            "reset_years":      round(T_reset, 4),
            "pay_years":        round(T_pay, 4),
            "fwd_rate_pct":     round(F_fwd * 100, 4),
            "discount_factor":  round(df_pay, 6),
            "pv":               round(pv, 2),
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
    swaption_type: str,   # "payer" | "receiver"
    freq: float,
) -> tuple[float, float, float]:
    """Returns (price, par_swap_rate, annuity)."""
    S0  = curve.par_swap_rate(T_expiry, T_end, freq)
    ann = curve.annuity(T_expiry, T_end, freq)
    tau = max(T_expiry, 0.0)

    if vol_type == "normal":
        fn = bachelier_call if swaption_type == "payer" else bachelier_put
    else:
        fn = black76_call if swaption_type == "payer" else black76_put

    price = fn(S0, K, sigma, tau) * ann * notional
    return price, S0, ann
