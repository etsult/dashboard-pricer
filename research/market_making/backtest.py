"""
Market Making Backtest Engine
==============================

Simulates an Avellaneda-Stoikov market maker on historical OHLCV data.

Fill model (touch model, Cont & Kukanov 2013 approximation)
------------------------------------------------------------
  At each bar [open, high, low, close]:
    - If low  ≤ bid_quote → bid fill at bid_quote
    - If high ≥ ask_quote → ask fill at ask_quote
  Both can fill in the same bar when the full bar range > full spread.

Calibration — fractional space
-------------------------------
  σ̃       = per-bar log-return std  (passed directly, NOT annualised)
  κ       = 1 / (target half-spread fraction)
              → 3333 for 3 bps target, 1666 for 6 bps, etc.
  T_bars  = look-ahead window (bars), typically 30-120
  γ       = dimensionless risk aversion, typically 0.05-0.3

Fill rate intuition (touch model, symmetric quotes):
  For a bar with range R bps, a fill occurs when
  mid ± half_spread lies within the bar.
  Expected fill rate per side ≈ P(|Z| > half_spread / (R/2))
  → for 3 bps hs and 9 bps range: fill ~66% of bars.

Known optimistic biases vs live trading:
  1. Queue position: assumes immediate priority at price → overestimates fills
  2. Zero market impact
  3. Zero latency
  Use a discount factor of 0.3-0.5 on simulated fills for live estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .models import ASParams, AvellanedaStoikov, AvellanedaStoikovGueant
from .metrics import BacktestResult, Fill


@dataclass
class BacktestConfig:
    symbol:    str   = "SOL/USDT"
    exchange:  str   = "binance"
    model:     str   = "AS_Gueant"      # "AS_basic" | "AS_Gueant"

    # AS model — fractional space
    gamma:           float = 0.10       # dimensionless risk aversion
    kappa:           float = 3333.0     # 1/(target half-spread fraction); 3333→3 bps
    T_bars:          float = 60.0       # rolling horizon in bars
    max_inventory:   float = 5.0        # base-asset units
    order_size:      float = 0.1
    maker_fee:       float = -0.0001    # negative = rebate
    taker_fee:       float = 0.0004
    max_half_spread_frac: float = 0.005 # 50 bps hard cap

    # Volatility
    vol_window_bars: int   = 60
    vol_floor_frac:  float = 1e-5       # min per-bar σ̃ (prevents zero-div)

    flat_at_end:     bool  = True
    initial_cash:    float = 0.0


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    """
    Core simulation loop.

    Parameters
    ----------
    df  : OHLCV DataFrame [timestamp, open, high, low, close, volume]
    cfg : BacktestConfig

    Returns
    -------
    BacktestResult with time series, fills, and metrics.
    """
    result = BacktestResult(symbol=cfg.symbol, model=cfg.model)

    cash      = cfg.initial_cash
    inventory = 0.0

    # Per-bar log-return std (fractional, not annualised)
    log_rets   = np.log(df["close"] / df["close"].shift(1)).fillna(0).values
    roll_std   = pd.Series(log_rets).rolling(cfg.vol_window_bars, min_periods=2).std().fillna(0).values

    params = ASParams(
        gamma               = cfg.gamma,
        kappa               = cfg.kappa,
        T_bars              = cfg.T_bars,
        max_inventory       = cfg.max_inventory,
        order_size          = cfg.order_size,
        maker_fee           = cfg.maker_fee,
        taker_fee           = cfg.taker_fee,
        max_half_spread_frac= cfg.max_half_spread_frac,
    )

    model_cls = AvellanedaStoikovGueant if cfg.model == "AS_Gueant" else AvellanedaStoikov
    model     = model_cls(params)

    n = len(df)
    for i in range(n):
        row = df.iloc[i]
        ts  = float(row["timestamp"])
        mid = (float(row["high"]) + float(row["low"])) / 2.0

        # ── Volatility (fractional per-bar std) ──────────────────────────────
        sigma_pb = max(float(roll_std[i]), cfg.vol_floor_frac)

        # ── Time remaining (decay toward 0 over T_bars window) ──────────────
        t_remaining = max(cfg.T_bars - (i % cfg.T_bars), 1.0)

        # ── Quotes ───────────────────────────────────────────────────────────
        bid_q, ask_q = model.quotes(mid, inventory, sigma_pb, t_remaining)

        # ── Fill simulation (touch model) ────────────────────────────────────
        lo, hi = float(row["low"]), float(row["high"])

        # 5-bar forward mid for adverse selection estimate
        look      = min(i + 5, n - 1)
        mid_5     = (float(df.iloc[look]["high"]) + float(df.iloc[look]["low"])) / 2.0

        spread_bps = 0.0
        if bid_q and ask_q:
            spread_bps = (ask_q - bid_q) / mid * 10_000

        if bid_q is not None and lo <= bid_q:
            cash      -= bid_q * cfg.order_size * (1.0 + cfg.maker_fee)
            inventory += cfg.order_size
            result.fills.append(Fill(
                timestamp=ts, side="bid", price=bid_q, qty=cfg.order_size,
                mid_at_fill=mid, mid_5s_after=mid_5, spread_quoted=spread_bps,
            ))

        if ask_q is not None and hi >= ask_q:
            cash      += ask_q * cfg.order_size * (1.0 - cfg.maker_fee)
            inventory -= cfg.order_size
            result.fills.append(Fill(
                timestamp=ts, side="ask", price=ask_q, qty=cfg.order_size,
                mid_at_fill=mid, mid_5s_after=mid_5, spread_quoted=spread_bps,
            ))

        # ── Mark-to-market ───────────────────────────────────────────────────
        result.timestamps.append(ts)
        result.mid_prices.append(mid)
        result.inventories.append(inventory)
        result.running_pnl.append(cash + inventory * float(row["close"]))
        result.bid_quotes.append(bid_q)
        result.ask_quotes.append(ask_q)

    # ── Flatten at end ────────────────────────────────────────────────────────
    if cfg.flat_at_end and abs(inventory) > 1e-9:
        last_mid = result.mid_prices[-1]
        cash    += inventory * last_mid * (1.0 - cfg.taker_fee)
        inventory = 0.0
        result.running_pnl[-1] = cash + inventory * last_mid

    result.compute_metrics()
    return result


# ── Sensitivity sweeps ────────────────────────────────────────────────────────

def gamma_sensitivity(
    df: pd.DataFrame, cfg: BacktestConfig,
    gammas: Sequence[float] = (0.01, 0.05, 0.10, 0.20, 0.40, 0.80),
) -> list[dict]:
    rows = []
    for g in gammas:
        c = BacktestConfig(**{**cfg.__dict__, "gamma": g})
        r = run_backtest(df, c)
        m = {**r.metrics, "gamma": g}
        rows.append(m)
    return rows


def kappa_sensitivity(
    df: pd.DataFrame, cfg: BacktestConfig,
    kappas: Sequence[float] = (500, 1000, 1666, 3333, 5000, 10000),
) -> list[dict]:
    rows = []
    for k in kappas:
        c = BacktestConfig(**{**cfg.__dict__, "kappa": k})
        r = run_backtest(df, c)
        target_hs_bps = round(10000 / k, 1)
        m = {**r.metrics, "kappa": k, "target_hs_bps": target_hs_bps}
        rows.append(m)
    return rows
