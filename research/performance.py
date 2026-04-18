"""
Standard performance metrics for any P&L series.

All functions take a pd.Series of daily P&L in dollar terms.
Everything is annualized assuming 365 trading days (crypto) unless noted.

Usage:
    from research.performance import compute_metrics
    stats = compute_metrics(daily_pnl_series)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


TRADING_DAYS = 365  # crypto trades 24/7


@dataclass
class PerformanceMetrics:
    # Return
    total_pnl: float
    avg_daily_pnl: float

    # Risk-adjusted
    sharpe: float       # annualized: mean(pnl) / std(pnl) * sqrt(365)
    sortino: float      # like Sharpe but only penalizes downside vol
    calmar: float       # annualized return / max drawdown (magnitude)

    # Drawdown
    max_drawdown: float         # largest peak-to-trough loss in $
    max_drawdown_pct: float     # same, as % of peak cumulative P&L
    avg_drawdown: float         # average drawdown in $

    # Tail risk
    var_95: float       # 1-day 95% Value at Risk (loss threshold)
    cvar_95: float      # Expected Shortfall: mean loss beyond VaR
    var_99: float
    cvar_99: float

    # Trade statistics (if trade_pnl provided, else None)
    win_rate: float | None
    n_trades: int | None
    avg_win: float | None
    avg_loss: float | None
    profit_factor: float | None  # gross wins / gross losses


def compute_metrics(
    daily_pnl: pd.Series,
    trade_pnl: pd.Series | None = None,
) -> PerformanceMetrics:
    """
    Compute the full set of performance metrics from a daily P&L series.

    Parameters
    ----------
    daily_pnl : pd.Series
        One value per day — the net P&L for that day in $.
    trade_pnl : pd.Series, optional
        One value per completed trade. Used for win rate, profit factor, etc.
    """
    pnl = daily_pnl.dropna()

    if len(pnl) < 2:
        raise ValueError("Need at least 2 data points to compute metrics.")

    # ── Return ────────────────────────────────────────────────────────
    total_pnl    = float(pnl.sum())
    avg_daily    = float(pnl.mean())
    std_daily    = float(pnl.std())

    # ── Risk-adjusted ─────────────────────────────────────────────────
    sharpe = float(avg_daily / std_daily * np.sqrt(TRADING_DAYS)) if std_daily > 0 else 0.0

    downside = pnl[pnl < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else std_daily
    sortino = float(avg_daily / downside_std * np.sqrt(TRADING_DAYS)) if downside_std > 0 else 0.0

    # ── Drawdown ──────────────────────────────────────────────────────
    cum_pnl  = pnl.cumsum()
    peak     = cum_pnl.cummax()
    dd       = cum_pnl - peak                   # always <= 0

    max_dd   = float(dd.min())
    avg_dd   = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

    # Drawdown as % of peak (avoid division by zero if strategy is always in loss)
    peak_val = float(peak.max())
    max_dd_pct = (max_dd / peak_val * 100) if peak_val > 0 else 0.0

    # Calmar: annualized total return / |max drawdown|
    n_years = len(pnl) / TRADING_DAYS
    ann_pnl = total_pnl / n_years if n_years > 0 else 0.0
    calmar  = float(ann_pnl / abs(max_dd)) if max_dd < 0 else 0.0

    # ── Tail risk ─────────────────────────────────────────────────────
    var_95  = float(np.percentile(pnl, 5))    # 5th percentile (negative = loss)
    cvar_95 = float(pnl[pnl <= var_95].mean()) if (pnl <= var_95).any() else var_95
    var_99  = float(np.percentile(pnl, 1))
    cvar_99 = float(pnl[pnl <= var_99].mean()) if (pnl <= var_99).any() else var_99

    # ── Trade stats ───────────────────────────────────────────────────
    win_rate = profit_factor = avg_win = avg_loss = None
    n_trades = None

    if trade_pnl is not None and len(trade_pnl) > 0:
        t = trade_pnl.dropna()
        n_trades = len(t)
        wins  = t[t > 0]
        losses = t[t < 0]
        win_rate      = float(len(wins) / n_trades) if n_trades > 0 else 0.0
        avg_win       = float(wins.mean())  if len(wins)   > 0 else 0.0
        avg_loss      = float(losses.mean()) if len(losses) > 0 else 0.0
        gross_wins    = float(wins.sum())
        gross_losses  = abs(float(losses.sum()))
        profit_factor = float(gross_wins / gross_losses) if gross_losses > 0 else 0.0

    return PerformanceMetrics(
        total_pnl=round(total_pnl, 2),
        avg_daily_pnl=round(avg_daily, 2),
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        calmar=round(calmar, 3),
        max_drawdown=round(max_dd, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        avg_drawdown=round(avg_dd, 2),
        var_95=round(var_95, 2),
        cvar_95=round(cvar_95, 2),
        var_99=round(var_99, 2),
        cvar_99=round(cvar_99, 2),
        win_rate=round(win_rate, 4) if win_rate is not None else None,
        n_trades=n_trades,
        avg_win=round(avg_win, 2) if avg_win is not None else None,
        avg_loss=round(avg_loss, 2) if avg_loss is not None else None,
        profit_factor=round(profit_factor, 3) if profit_factor is not None else None,
    )
