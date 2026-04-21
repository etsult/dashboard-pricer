"""
Market Making Backtest Engine
==============================

Simulates an Avellaneda-Stoikov market maker on historical OHLCV data.

Fill model (touch model, Cont & Kukanov 2013 approximation)
------------------------------------------------------------
  At each bar [open, high, low, close]:
    - If low  ≤ bid_quote → bid fill at bid_quote
    - If high ≥ ask_quote → ask fill at ask_quote
  Both can fill in the same bar (crossed bar).

Limitations of the touch model vs L2 data
------------------------------------------
  1. Fill priority: in reality, queue position matters.
     We assume immediate fill when price touches — optimistic.
  2. Market impact: large orders move the market.
     We assume zero impact — reasonable for small order sizes.
  3. Latency: we assume zero latency — optimistic.
  These biases make the backtest an *upper bound* on live performance.

For a production system, switch to a queue-reactive model
(Stoikov & Waeber, 2012) using actual order book data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .models import ASParams, AvellanedaStoikov, AvellanedaStoikovGueant, rolling_sigma
from .metrics import BacktestResult, Fill


@dataclass
class BacktestConfig:
    """All parameters needed to run a single backtest."""
    symbol:    str   = "SOL/USDT"
    exchange:  str   = "binance"
    model:     str   = "AS_Gueant"      # "AS_basic" | "AS_Gueant"

    # AS model params
    gamma:           float = 0.05
    kappa:           float = 1.5
    T_hours:         float = 1.0        # rolling time horizon
    max_inventory:   float = 5.0        # max position in base asset
    order_size:      float = 0.1        # per quote
    maker_fee:       float = -0.0001    # rebate
    taker_fee:       float = 0.0004

    # Volatility
    vol_window_bars: int  = 60          # bars for rolling σ estimate
    vol_floor:       float = 0.10       # minimum annualised vol (10%)

    # Execution
    flat_at_end:     bool = True        # flatten inventory at last bar close
    initial_cash:    float = 0.0        # USD starting cash (relative to 0)


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    """
    Core backtest loop.

    Parameters
    ----------
    df  : OHLCV DataFrame with columns [timestamp, open, high, low, close, volume]
    cfg : BacktestConfig

    Returns
    -------
    BacktestResult with full time series and fills.
    """
    # ── Infer bar duration ────────────────────────────────────────────────────
    dt_secs        = float(df["timestamp"].diff().median())
    bars_per_year  = 365 * 24 * 3600 / dt_secs
    bars_per_hour  = 3600 / dt_secs
    ann_factor     = bars_per_year

    # ── Initialise model ─────────────────────────────────────────────────────
    closes = df["close"].tolist()
    result = BacktestResult(symbol=cfg.symbol, model=cfg.model)

    cash      = cfg.initial_cash
    inventory = 0.0
    fills: list[Fill] = []

    # Pre-compute rolling vol
    log_rets   = np.log(df["close"] / df["close"].shift(1)).fillna(0).values
    roll_std   = pd.Series(log_rets).rolling(cfg.vol_window_bars).std().fillna(0).values

    for i, row in enumerate(df.itertuples(index=False)):
        ts    = row.timestamp
        mid   = (row.high + row.low) / 2.0   # use bar midpoint as mid price

        # ── Volatility ────────────────────────────────────────────────────────
        sigma_per_bar = roll_std[i] if i >= cfg.vol_window_bars else roll_std[cfg.vol_window_bars]
        sigma_ann     = max(sigma_per_bar * math.sqrt(ann_factor), cfg.vol_floor)

        # ── Build AS model params for this bar ───────────────────────────────
        t_remaining = cfg.T_hours / bars_per_hour   # time left in "hours" unit

        params = ASParams(
            gamma        = cfg.gamma,
            sigma        = sigma_ann,
            kappa        = cfg.kappa,
            T            = cfg.T_hours,
            dt           = dt_secs / 3600,
            max_inventory= cfg.max_inventory,
            order_size   = cfg.order_size,
            maker_fee    = cfg.maker_fee,
            taker_fee    = cfg.taker_fee,
        )

        model_cls = AvellanedaStoikovGueant if cfg.model == "AS_Gueant" else AvellanedaStoikov
        model     = model_cls(params)

        bid_q, ask_q = model.quotes(mid, inventory, t_remaining)

        # ── Fill simulation (touch model) ────────────────────────────────────
        filled_bid = bid_q is not None and row.low  <= bid_q
        filled_ask = ask_q is not None and row.high >= ask_q

        # Look-ahead for adverse selection (5 bars forward)
        look = min(i + 5, len(df) - 1)
        mid_5 = (df.iloc[look]["high"] + df.iloc[look]["low"]) / 2.0

        spread_bps = 0.0
        if bid_q and ask_q:
            spread_bps = (ask_q - bid_q) / mid * 10_000

        if filled_bid:
            exec_price = bid_q            # type: ignore[assignment]
            cash      -= exec_price * cfg.order_size * (1 + cfg.maker_fee)
            inventory += cfg.order_size
            fills.append(Fill(
                timestamp    = ts,
                side         = "bid",
                price        = exec_price,
                qty          = cfg.order_size,
                mid_at_fill  = mid,
                mid_5s_after = mid_5,
                spread_quoted= spread_bps,
            ))

        if filled_ask:
            exec_price = ask_q            # type: ignore[assignment]
            cash      += exec_price * cfg.order_size * (1 - cfg.maker_fee)
            inventory -= cfg.order_size
            fills.append(Fill(
                timestamp    = ts,
                side         = "ask",
                price        = exec_price,
                qty          = cfg.order_size,
                mid_at_fill  = mid,
                mid_5s_after = mid_5,
                spread_quoted= spread_bps,
            ))

        # ── Mark-to-market P&L ────────────────────────────────────────────────
        pnl = cash + inventory * row.close

        result.timestamps.append(ts)
        result.mid_prices.append(mid)
        result.inventories.append(inventory)
        result.running_pnl.append(pnl)
        result.bid_quotes.append(bid_q)
        result.ask_quotes.append(ask_q)

    # ── Flatten at end ────────────────────────────────────────────────────────
    if cfg.flat_at_end and abs(inventory) > 1e-9:
        last_mid = result.mid_prices[-1]
        cash += inventory * last_mid * (1 - cfg.taker_fee)
        inventory = 0.0
        result.running_pnl[-1] = cash

    result.fills = fills
    result.compute_metrics()
    return result


# ── Parameter sensitivity ─────────────────────────────────────────────────────

def gamma_sensitivity(
    df: pd.DataFrame,
    cfg: BacktestConfig,
    gammas: Sequence[float] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40),
) -> list[dict]:
    """
    Run the backtest across multiple γ values.
    Returns a list of metrics dicts, one per γ.

    Use this to calibrate γ for the target asset.
    """
    rows = []
    for g in gammas:
        c = BacktestConfig(**{**cfg.__dict__, "gamma": g})
        r = run_backtest(df, c)
        m = r.metrics.copy()
        m["gamma"] = g
        rows.append(m)
    return rows


def kappa_sensitivity(
    df: pd.DataFrame,
    cfg: BacktestConfig,
    kappas: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0),
) -> list[dict]:
    """Run the backtest across multiple κ values."""
    rows = []
    for k in kappas:
        c = BacktestConfig(**{**cfg.__dict__, "kappa": k})
        r = run_backtest(df, c)
        m = r.metrics.copy()
        m["kappa"] = k
        rows.append(m)
    return rows
