# pricer/backtest/dh_straddle.py
"""
Delta-hedged short ATM straddle backtest engine.

Strategy
────────
Every T_days sell a new ATM straddle (K = S at inception).
Rebalance the delta hedge every `rebalance_freq` days using spot.
Mark option to market daily using DVOL as repricing vol.

P&L decomposition (short straddle)
────────────────────────────────────
  Total P&L  = -ΔV_option  +  hedge × ΔS          ← exact (MTM)

  Approximation via Greeks (Taylor, sums ≈ total):
    Theta  = +|Θ| × Δt        positive  — time decay collected
    Gamma  = -½ Γ S² r²       negative  — cost of spot moves
    Vega   = -𝒱 × Δσ          ±          — cost/gain from vol changes

  The edge:  E[Gamma] < E[Theta]  ⟺  σ_realized < σ_implied
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# Black-76  option primitives  (r = 0, i.e. F = S)
# ─────────────────────────────────────────────────────────────────────

def _d1(S: float, K: float, sigma: float, tau: float) -> float:
    if tau <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))


def straddle_price(S: float, K: float, sigma: float, tau: float) -> float:
    """ATM straddle price (call + put), r = 0."""
    if tau <= 0:
        return abs(S - K)
    d1 = _d1(S, K, sigma, tau)
    d2 = d1 - sigma * np.sqrt(tau)
    call = S * norm.cdf(d1)  - K * norm.cdf(d2)
    put  = K * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call + put


def straddle_delta(S: float, K: float, sigma: float, tau: float) -> float:
    """Net delta of the straddle (call Δ + put Δ = 2N(d1) − 1)."""
    if tau <= 0:
        return float(np.sign(S - K))
    d1 = _d1(S, K, sigma, tau)
    return 2.0 * norm.cdf(d1) - 1.0


def straddle_gamma(S: float, K: float, sigma: float, tau: float) -> float:
    """Gamma of straddle = 2 × call gamma."""
    if tau <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = _d1(S, K, sigma, tau)
    return 2.0 * norm.pdf(d1) / (S * sigma * np.sqrt(tau))


def straddle_theta(S: float, K: float, sigma: float, tau: float) -> float:
    """Theta of straddle per year (negative = long loses time value)."""
    if tau <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, sigma, tau)
    return -S * sigma * norm.pdf(d1) / np.sqrt(tau)   # per year, for long


def straddle_vega(S: float, K: float, sigma: float, tau: float) -> float:
    """Vega of straddle = 2 × call vega (per unit of sigma, not per %)."""
    if tau <= 0:
        return 0.0
    d1 = _d1(S, K, sigma, tau)
    return 2.0 * S * np.sqrt(tau) * norm.pdf(d1)


# ─────────────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────────────

def run(
    spot: pd.Series,            # daily close prices (index = date)
    dvol_pct: pd.Series,        # DVOL in percent (e.g. 60.0)
    T_days: int       = 30,     # straddle maturity in calendar days
    rebalance_freq: int = 1,    # delta rebalance every N days  (1 = daily)
    notional_usd: float = 100_000,  # USD notional per trade
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Roll and delta-hedge a short ATM straddle.

    Returns
    -------
    daily : DataFrame  indexed by date, one row per active day
        total_pnl, theta_pnl, gamma_pnl, vega_pnl,
        hedge_pnl, option_pnl,
        S, iv, tau, delta, gamma, vega, V (option value)

    trades : DataFrame  one row per completed trade
        entry/exit date, strike, entry IV, premium received, P&L
    """
    # Align on common dates ────────────────────────────────────────────
    combined = pd.concat(
        [spot.rename("S"), (dvol_pct / 100).rename("iv")], axis=1
    ).dropna()
    S_ser  = combined["S"]
    iv_ser = combined["iv"]

    daily_rows  = []
    trade_rows  = []

    # Roll a new straddle every T_days ─────────────────────────────────
    for i_start in range(0, len(combined) - T_days, T_days):

        S0   = S_ser.iloc[i_start]
        iv0  = iv_ser.iloc[i_start]
        K    = S0                                   # ATM at inception
        tau0 = T_days / 365.0

        # BTC notional: how many BTC the straddle covers
        btc_qty = notional_usd / S0

        # Entry: receive straddle premium (we are short)
        premium_usd = straddle_price(S0, K, iv0, tau0) * btc_qty
        V_prev      = straddle_price(S0, K, iv0, tau0) * btc_qty
        hedge       = straddle_delta(S0, K, iv0, tau0) * btc_qty  # BTC held

        trade_entry = {
            "Entry Date":    combined.index[i_start].date(),
            "K (Strike)":    round(K, 0),
            "Entry S":       round(S0, 0),
            "Entry IV (%)":  round(iv0 * 100, 1),
            "Premium ($)":   round(premium_usd, 0),
            "BTC qty":       round(btc_qty, 4),
        }
        trade_pnl = 0.0

        for t in range(1, T_days + 1):
            idx = i_start + t
            if idx >= len(combined):
                break

            date  = combined.index[idx]
            S_t   = S_ser.iloc[idx]
            iv_t  = iv_ser.iloc[idx]
            S_tm1 = S_ser.iloc[idx - 1]
            iv_tm1 = iv_ser.iloc[idx - 1]
            tau_t = max((T_days - t) / 365.0, 0.0)
            dt    = 1.0 / 365.0

            # Mark option to market
            V_new = straddle_price(S_t, K, iv_t, tau_t) * btc_qty

            # ── Exact P&L ─────────────────────────────────────────
            option_pnl = -(V_new - V_prev)          # short: gain if V drops
            hedge_pnl  = hedge * (S_t - S_tm1)      # gain if S moves with hedge
            total_pnl  = option_pnl + hedge_pnl

            # ── Greek decomposition (approximate) ─────────────────
            # Use greeks at start of day (t-1 values)
            tau_prev = (T_days - t + 1) / 365.0
            G  = straddle_gamma(S_tm1, K, iv_tm1, tau_prev) * btc_qty
            Th = straddle_theta(S_tm1, K, iv_tm1, tau_prev) * btc_qty
            Ve = straddle_vega( S_tm1, K, iv_tm1, tau_prev) * btc_qty

            dS        = S_t - S_tm1
            dsigma    = iv_t - iv_tm1

            theta_pnl = -Th * dt                    # short θ → positive
            gamma_pnl = -0.5 * G * dS ** 2          # short γ → negative
            vega_pnl  = -Ve * dsigma                # ± depending on Δσ
            rv_daily  = abs(dS / S_tm1)             # |daily return|

            daily_rows.append({
                "date":        date,
                "trade_start": combined.index[i_start].date(),
                "S":           S_t,
                "iv":          iv_t * 100,
                "tau":         tau_t,
                "V":           V_new,
                "delta":       straddle_delta(S_t, K, iv_t, tau_t) * btc_qty,
                "gamma":       G,
                "vega":        Ve,
                "total_pnl":   total_pnl,
                "option_pnl":  option_pnl,
                "hedge_pnl":   hedge_pnl,
                "theta_pnl":   theta_pnl,
                "gamma_pnl":   gamma_pnl,
                "vega_pnl":    vega_pnl,
                "rv_daily_pct": rv_daily * 100,
            })
            trade_pnl += total_pnl

            # ── Rebalance delta ────────────────────────────────────
            if t % rebalance_freq == 0:
                hedge = straddle_delta(S_t, K, iv_t, tau_t) * btc_qty

            V_prev = V_new

        trade_entry.update({
            "Exit Date":  combined.index[min(i_start + T_days, len(combined) - 1)].date(),
            "Exit IV (%)": round(iv_ser.iloc[min(i_start + T_days, len(combined) - 1)] * 100, 1),
            "P&L ($)":    round(trade_pnl, 0),
            "Win":        trade_pnl > 0,
        })
        trade_rows.append(trade_entry)

    daily_df  = pd.DataFrame(daily_rows).set_index("date") if daily_rows else pd.DataFrame()
    trades_df = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame()
    return daily_df, trades_df
