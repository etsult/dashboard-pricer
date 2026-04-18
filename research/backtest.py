"""
Research-layer backtest runner.

Wraps the existing pricer/backtest engines and applies:
  - A cost model (spread, commission, slippage, funding)
  - Standardized performance metrics

This is the single entry point for the API and any future notebook/frontend.
The underlying engines (pricer/backtest/*.py) stay untouched.
"""

from __future__ import annotations

import pandas as pd
from dataclasses import asdict

from pricer.backtest.dh_straddle import run as _run_dh_straddle
from market_data.providers.deribit_history import (
    fetch_spot_history,
    fetch_dvol_history,
)
from research.costs import CostModel
from research.performance import compute_metrics, PerformanceMetrics


def run_dh_straddle(
    currency: str,
    history_days: int,
    T_days: int,
    rebalance_freq: int,
    notional_usd: float,
    costs: CostModel,
) -> dict:
    """
    Run the delta-hedged short straddle backtest with realistic transaction costs.

    Returns a dict with:
      - performance: PerformanceMetrics (all standard stats)
      - daily: list of dicts (one per trading day, for charts)
      - trades: list of dicts (one per completed trade)
      - cost_summary: total costs broken down by type
    """
    # ── 1. Fetch market data ───────────────────────────────────────────
    spot = fetch_spot_history(currency, history_days + 40)["close"]
    dvol = fetch_dvol_history(currency, history_days + 40)

    # ── 2. Run the core backtest engine ────────────────────────────────
    daily_df, trades_df = _run_dh_straddle(
        spot=spot,
        dvol_pct=dvol,
        T_days=T_days,
        rebalance_freq=rebalance_freq,
        notional_usd=notional_usd,
    )

    if daily_df.empty or trades_df.empty:
        raise ValueError("Not enough data for the selected parameters.")

    # ── 3. Apply transaction costs ────────────────────────────────────
    # The engine gives us gross P&L. We now subtract realistic frictions.

    total_entry_cost = 0.0
    total_exit_cost  = 0.0
    total_hedge_cost = 0.0
    total_funding    = 0.0

    # Option entry/exit: one straddle (2 legs) per trade, entry + exit
    for _, trade in trades_df.iterrows():
        premium      = abs(trade["Premium ($)"])
        entry_cost   = costs.option_entry(premium, n_legs=2)
        exit_cost    = costs.option_exit(premium * 0.1, n_legs=2)  # approx residual value at exit
        total_entry_cost += entry_cost
        total_exit_cost  += exit_cost

    # Hedge rebalance: one rebalance per rebalance_freq days
    rebalance_days = daily_df[daily_df.index.dayofweek >= 0]  # every day that exists
    for _, row in rebalance_days.iterrows():
        delta_btc   = abs(row["delta"])
        spot_price  = row["S"]
        hedge_cost  = costs.hedge_rebalance(spot_price, delta_btc)
        funding     = costs.daily_funding(abs(delta_btc) * spot_price)
        total_hedge_cost += hedge_cost
        total_funding    += funding

    total_costs = total_entry_cost + total_exit_cost + total_hedge_cost + total_funding

    # Deduct costs proportionally across trading days
    n_days = len(daily_df)
    daily_df = daily_df.copy()
    daily_df["cost"]          = total_costs / n_days  # simple daily amortization
    daily_df["net_total_pnl"] = daily_df["total_pnl"] - daily_df["cost"]

    # ── 4. Compute performance metrics ────────────────────────────────
    gross_metrics = compute_metrics(
        daily_pnl=daily_df["total_pnl"],
        trade_pnl=trades_df["P&L ($)"],
    )
    net_metrics = compute_metrics(
        daily_pnl=daily_df["net_total_pnl"],
        trade_pnl=trades_df["P&L ($)"] - total_costs / max(len(trades_df), 1),
    )

    # ── 5. Serialize ──────────────────────────────────────────────────
    daily_df_out = daily_df.reset_index()
    daily_df_out["date"] = daily_df_out["date"].astype(str)

    trades_df_out = trades_df.copy()
    for col in ["Entry Date", "Exit Date"]:
        if col in trades_df_out.columns:
            trades_df_out[col] = trades_df_out[col].astype(str)

    return {
        "gross_performance": asdict(gross_metrics),
        "net_performance":   asdict(net_metrics),
        "cost_summary": {
            "total_costs":       round(total_costs, 2),
            "entry_exit_costs":  round(total_entry_cost + total_exit_cost, 2),
            "hedge_rebal_costs": round(total_hedge_cost, 2),
            "funding_costs":     round(total_funding, 2),
            "cost_as_pct_gross": round(
                total_costs / abs(gross_metrics.total_pnl) * 100, 2
            ) if gross_metrics.total_pnl != 0 else None,
        },
        "daily":  daily_df_out.to_dict(orient="records"),
        "trades": trades_df_out.to_dict(orient="records"),
    }
