"""
Simple equity strategies: Lump Sum and DCA.

Both return the same output shape so they can be compared directly.
Designed to later accept multi-asset weights (All Weather, 60/40, etc.)

Output per strategy:
  daily  : one row per trading day — portfolio value, invested, return %
  trades : one row per buy event
  perf   : standard metrics (CAGR, Sharpe, max drawdown, etc.)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS_PER_YEAR = 252


# ─── Data ─────────────────────────────────────────────────────────────────────

def _fetch_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """
    Download adjusted close prices for one or more tickers.
    Returns a DataFrame with ticker names as columns.
    """
    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),  # end is exclusive in yfinance
        progress=False,
        auto_adjust=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
        if len(tickers) == 1:
            prices = prices.rename(columns={tickers[0]: tickers[0]})
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index).date  # type: ignore
    return prices.dropna()


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _compute_equity_metrics(
    daily_values: pd.Series,      # portfolio value over time
    total_invested: float,
    trade_dates: list[date],
) -> dict:
    """Compute standard metrics from a portfolio value series."""
    values = daily_values.dropna()
    if len(values) < 2:
        return {}

    daily_returns = values.pct_change().dropna()
    n_years = len(values) / TRADING_DAYS_PER_YEAR

    initial   = float(values.iloc[0])
    final     = float(values.iloc[-1])
    total_ret = (final - total_invested) / total_invested

    # CAGR
    cagr = (final / total_invested) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Volatility (annualized)
    vol = float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    # Sharpe (assume 0% risk-free for simplicity; easy to parameterize later)
    sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if daily_returns.std() > 0 else 0.0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    sortino = float(daily_returns.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(downside) > 1 and downside.std() > 0 else 0.0

    # Drawdown
    peak = values.cummax()
    dd   = (values - peak) / peak
    max_dd = float(dd.min())

    # Calmar
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    # Best / worst calendar year
    returns_by_year = (
        values.groupby(lambda d: d.year)
        .apply(lambda g: (g.iloc[-1] - g.iloc[0]) / g.iloc[0])
    )
    best_year  = float(returns_by_year.max()) if len(returns_by_year) > 0 else None
    worst_year = float(returns_by_year.min()) if len(returns_by_year) > 0 else None

    # Positive months rate
    monthly = values.groupby(lambda d: (d.year, d.month)).last()
    monthly_ret = monthly.pct_change().dropna()
    positive_months_pct = float((monthly_ret > 0).mean() * 100) if len(monthly_ret) > 0 else None

    # Ulcer Index: sqrt(mean(drawdown_pct^2)) — penalises depth AND duration
    ulcer = float(np.sqrt((dd ** 2).mean()) * 100)

    # % of days spent in drawdown
    pct_in_drawdown = float((dd < 0).mean() * 100)

    # Max drawdown duration (days from peak to next recovery)
    in_dd = (dd < 0).astype(int)
    streaks, current = [], 0
    for v in in_dd:
        if v:
            current += 1
        else:
            if current:
                streaks.append(current)
            current = 0
    if current:
        streaks.append(current)
    max_dd_duration_days = int(max(streaks)) if streaks else 0

    return {
        "final_value":            round(final, 2),
        "total_invested":         round(total_invested, 2),
        "total_return_pct":       round(total_ret * 100, 2),
        "cagr_pct":               round(cagr * 100, 2),
        "volatility_ann_pct":     round(vol * 100, 2),
        "sharpe":                 round(sharpe, 3),
        "sortino":                round(sortino, 3),
        "calmar":                 round(calmar, 3),
        "max_drawdown_pct":       round(max_dd * 100, 2),
        "max_dd_duration_days":   max_dd_duration_days,
        "ulcer_index":            round(ulcer, 3),
        "pct_in_drawdown":        round(pct_in_drawdown, 1),
        "positive_months_pct":    round(positive_months_pct, 1) if positive_months_pct is not None else None,
        "best_year_pct":          round(best_year * 100, 2) if best_year is not None else None,
        "worst_year_pct":         round(worst_year * 100, 2) if worst_year is not None else None,
        "n_trades":               len(trade_dates),
        "n_years":                round(n_years, 2),
    }


# ─── Lump Sum ─────────────────────────────────────────────────────────────────

def run_lump_sum(
    ticker: str,
    start: date,
    end: date,
    amount: float,
    transaction_cost_pct: float = 0.0,
) -> dict:
    """
    Buy `amount` USD of `ticker` on the first available trading day, hold until `end`.

    Returns
    -------
    dict with keys: performance, daily, trades
    """
    prices = _fetch_prices([ticker], start, end)
    if prices.empty:
        raise ValueError(f"No price data for {ticker} between {start} and {end}")

    col = ticker
    price_series = prices[col]

    # Buy on first day
    entry_price = float(price_series.iloc[0])
    cost        = amount * transaction_cost_pct
    shares      = (amount - cost) / entry_price

    # Daily portfolio value
    portfolio_values = price_series * shares

    trades = [{
        "date":       str(price_series.index[0]),
        "type":       "BUY",
        "shares":     round(shares, 6),
        "price":      round(entry_price, 4),
        "amount_usd": round(amount, 2),
        "cost_usd":   round(cost, 2),
    }]

    daily = []
    for d, val in portfolio_values.items():
        invested = amount
        daily.append({
            "date":          str(d),
            "price":         round(float(price_series[d]), 4),
            "shares":        round(shares, 6),
            "portfolio_value": round(float(val), 2),
            "invested":      round(invested, 2),
            "unrealized_pnl": round(float(val) - invested, 2),
            "return_pct":    round((float(val) - invested) / invested * 100, 3),
        })

    perf = _compute_equity_metrics(portfolio_values, amount, [price_series.index[0]])

    return {"performance": perf, "daily": daily, "trades": trades}


# ─── DCA ──────────────────────────────────────────────────────────────────────

FrequencyT = Literal["weekly", "biweekly", "monthly", "quarterly"]

_FREQ_DAYS: dict[str, int] = {
    "weekly":    7,
    "biweekly":  14,
    "monthly":   30,
    "quarterly": 90,
}


def run_dca(
    ticker: str,
    start: date,
    end: date,
    periodic_amount: float,
    frequency: FrequencyT = "monthly",
    transaction_cost_pct: float = 0.0,
) -> dict:
    """
    Invest `periodic_amount` USD into `ticker` at each `frequency` interval.

    Returns
    -------
    dict with keys: performance, daily, trades
    """
    prices = _fetch_prices([ticker], start, end)
    if prices.empty:
        raise ValueError(f"No price data for {ticker} between {start} and {end}")

    price_series = prices[ticker]
    trading_days = list(price_series.index)
    freq_days    = _FREQ_DAYS[frequency]

    # Determine buy dates: first day, then every freq_days calendar days
    buy_dates: set[date] = set()
    next_buy = trading_days[0]
    for d in trading_days:
        if d >= next_buy:
            buy_dates.add(d)
            next_buy = d + timedelta(days=freq_days)

    # Simulate day by day
    shares_held    = 0.0
    total_invested = 0.0
    trades         = []
    daily          = []

    for d in trading_days:
        price = float(price_series[d])

        if d in buy_dates:
            cost   = periodic_amount * transaction_cost_pct
            bought = (periodic_amount - cost) / price
            shares_held    += bought
            total_invested += periodic_amount
            trades.append({
                "date":       str(d),
                "type":       "BUY",
                "shares":     round(bought, 6),
                "price":      round(price, 4),
                "amount_usd": round(periodic_amount, 2),
                "cost_usd":   round(cost, 2),
                "total_shares": round(shares_held, 6),
                "avg_cost":   round(total_invested / shares_held, 4) if shares_held > 0 else None,
            })

        portfolio_value = shares_held * price
        daily.append({
            "date":            str(d),
            "price":           round(price, 4),
            "shares":          round(shares_held, 6),
            "portfolio_value": round(portfolio_value, 2),
            "invested":        round(total_invested, 2),
            "unrealized_pnl":  round(portfolio_value - total_invested, 2),
            "return_pct":      round((portfolio_value - total_invested) / total_invested * 100, 3)
                               if total_invested > 0 else 0.0,
        })

    # Build portfolio value series for metrics
    pv_series = pd.Series(
        [r["portfolio_value"] for r in daily],
        index=[pd.Timestamp(r["date"]) for r in daily],
    )

    perf = _compute_equity_metrics(pv_series, total_invested, list(buy_dates))
    perf["avg_cost_basis"] = round(
        total_invested / shares_held, 4
    ) if shares_held > 0 else None

    return {"performance": perf, "daily": daily, "trades": trades}


# ─── All Weather Portfolio ────────────────────────────────────────────────────

# Ray Dalio's original allocation (ETF proxies)
ALL_WEATHER_WEIGHTS: dict[str, float] = {
    "SPY": 0.30,   # US large cap equities
    "TLT": 0.40,   # 20+ year US Treasuries (long duration)
    "IEF": 0.15,   # 7-10 year US Treasuries (intermediate)
    "GLD": 0.075,  # Gold
    "GSG": 0.075,  # Broad commodities (iShares S&P GSCI)
}

RebalanceFreqT = Literal["daily", "monthly", "quarterly", "annually"]

_REBAL_DAYS: dict[str, int] = {
    "daily":     1,
    "monthly":   21,   # ~21 trading days
    "quarterly": 63,
    "annually":  252,
}


def run_all_weather(
    start: date,
    end: date,
    initial_amount: float,
    rebalance_frequency: RebalanceFreqT = "quarterly",
    transaction_cost_pct: float = 0.0,
    weights: dict[str, float] | None = None,
) -> dict:
    """
    All Weather Portfolio: multi-asset, periodically rebalanced.

    Default weights follow Ray Dalio's published allocation:
      30% SPY · 40% TLT · 15% IEF · 7.5% GLD · 7.5% GSG

    Parameters
    ----------
    rebalance_frequency : how often to rebalance back to target weights
    weights : override the default allocation (must sum to 1.0)

    Returns
    -------
    dict with keys: performance, daily, trades, weights_used
    """
    target_weights = weights or ALL_WEATHER_WEIGHTS
    tickers = list(target_weights.keys())

    prices = _fetch_prices(tickers, start, end)
    if prices.empty:
        raise ValueError("No price data returned. Check tickers and date range.")

    prices = prices.dropna()
    trading_days = list(prices.index)
    if len(trading_days) < 2:
        raise ValueError("Not enough overlapping price data for all tickers.")

    rebal_every = _REBAL_DAYS[rebalance_frequency]

    # ── Initialise: buy at target weights on day 0 ────────────────────
    day0_prices = prices.iloc[0]
    holdings: dict[str, float] = {}
    trades: list[dict] = []

    for ticker, w in target_weights.items():
        alloc = initial_amount * w
        cost  = alloc * transaction_cost_pct
        holdings[ticker] = (alloc - cost) / float(day0_prices[ticker])
        trades.append({
            "date": str(trading_days[0]),
            "type": "BUY (init)",
            "ticker": ticker,
            "shares": round(holdings[ticker], 6),
            "price": round(float(day0_prices[ticker]), 4),
            "amount_usd": round(alloc, 2),
            "weight_target": round(w, 4),
        })

    total_rebal_cost = initial_amount * transaction_cost_pct * len(tickers)
    total_invested   = initial_amount
    days_since_rebal = 0
    daily: list[dict] = []

    for i, d in enumerate(trading_days):
        day_prices = prices.loc[d]

        asset_values    = {t: holdings[t] * float(day_prices[t]) for t in tickers}
        portfolio_value = sum(asset_values.values())
        current_weights = {t: asset_values[t] / portfolio_value for t in tickers}
        max_drift       = max(abs(current_weights[t] - target_weights[t]) for t in tickers)

        should_rebal = (i > 0) and (days_since_rebal >= rebal_every)

        if should_rebal:
            rebal_cost = 0.0
            for ticker, w in target_weights.items():
                target_value  = portfolio_value * w
                current_value = asset_values[ticker]
                trade_value   = abs(target_value - current_value)
                cost          = trade_value * transaction_cost_pct
                rebal_cost   += cost
                holdings[ticker] = (portfolio_value * w - cost / 2) / float(day_prices[ticker])
                trades.append({
                    "date":          str(d),
                    "type":          "REBAL BUY" if target_value > current_value else "REBAL SELL",
                    "ticker":        ticker,
                    "shares":        round(holdings[ticker], 6),
                    "price":         round(float(day_prices[ticker]), 4),
                    "amount_usd":    round(trade_value, 2),
                    "weight_target": round(w, 4),
                    "weight_before": round(current_weights[ticker], 4),
                    "drift_pct":     round((current_weights[ticker] - w) * 100, 2),
                })
            total_rebal_cost += rebal_cost
            asset_values     = {t: holdings[t] * float(day_prices[t]) for t in tickers}
            portfolio_value  = sum(asset_values.values())
            current_weights  = {t: asset_values[t] / portfolio_value for t in tickers}
            days_since_rebal = 0
        else:
            days_since_rebal += 1

        daily.append({
            "date":            str(d),
            "portfolio_value": round(portfolio_value, 2),
            "invested":        round(total_invested, 2),
            "unrealized_pnl":  round(portfolio_value - total_invested, 2),
            "return_pct":      round((portfolio_value - total_invested) / total_invested * 100, 3),
            "max_drift_pct":   round(max_drift * 100, 2),
            **{f"weight_{t}": round(current_weights[t], 4) for t in tickers},
            **{f"value_{t}":  round(asset_values[t], 2)   for t in tickers},
        })

    pv_series = pd.Series(
        [r["portfolio_value"] for r in daily],
        index=pd.to_datetime([r["date"] for r in daily]),
    )

    perf = _compute_equity_metrics(pv_series, total_invested, [])
    perf["n_trades"]         = len(trades)
    perf["total_rebal_cost"] = round(total_rebal_cost, 2)
    perf["cost_drag_pct"]    = round(total_rebal_cost / initial_amount * 100, 3)

    return {
        "performance":  perf,
        "daily":        daily,
        "trades":       trades,
        "weights_used": target_weights,
    }
