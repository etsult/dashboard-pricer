"""
Market Data Loader for Market Making Backtest
==============================================

Fetches OHLCV + trade data from exchanges via ccxt.

For a rigorous MM backtest we ideally need:
  1. L2 order book snapshots (best bid/ask + depth)
  2. Trade tape (price, qty, side, timestamp)

Since L2 historical data is expensive/hard to obtain, we use a
*simulation model* on top of OHLCV:

  Fill model (Cont & Kukanov, 2013 approximation):
    A limit order at price p gets filled when the mid-price moves
    through p.  This is the "touch model" — simplest valid approximation.
    For a more realistic model, use a queue-reactive model.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import pandas as pd
import numpy as np


class OHLCVBar(NamedTuple):
    timestamp: float   # unix ms → converted to seconds
    open:  float
    high:  float
    low:   float
    close: float
    volume: float      # base-asset volume


def fetch_ohlcv(
    symbol: str,
    exchange_id: str = "binance",
    timeframe: str = "1m",
    limit: int = 1440,      # 1 day at 1m
    since_days_ago: int = 30,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars via ccxt.

    Returns a DataFrame with columns:
        timestamp (unix s), open, high, low, close, volume

    Falls back to synthetic data if exchange is unreachable
    (useful for offline development/testing).
    """
    try:
        import ccxt
        exchange_cls = getattr(ccxt, exchange_id)
        ex = exchange_cls({"enableRateLimit": True})

        since_ms = int((time.time() - since_days_ago * 86_400) * 1_000)
        all_bars: list[list] = []
        while True:
            bars = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=1000)
            if not bars:
                break
            all_bars.extend(bars)
            if len(bars) < 1000:
                break
            since_ms = bars[-1][0] + 1
            time.sleep(ex.rateLimit / 1000)

        df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = df["timestamp"] / 1000.0
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        return df

    except Exception as e:
        print(f"[data_loader] ccxt fetch failed ({e}), using synthetic data.")
        return _synthetic_ohlcv(symbol, n_bars=limit)


def _synthetic_ohlcv(symbol: str, n_bars: int = 1440, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for offline testing.

    Uses geometric Brownian motion with realistic crypto parameters.
    """
    rng = np.random.default_rng(seed)

    # Rough per-symbol parameters
    params = {
        "BTC/USDT":  (60_000, 0.60),
        "ETH/USDT":  (3_000,  0.70),
        "SOL/USDT":  (150,    1.00),
        "AVAX/USDT": (35,     1.10),
        "BNB/USDT":  (400,    0.60),
    }
    s0, ann_vol = params.get(symbol, (100, 0.80))

    dt = 1 / (365 * 24 * 60)   # 1-minute steps
    sigma_dt = ann_vol * np.sqrt(dt)

    rets = rng.normal(0, sigma_dt, n_bars)
    closes = s0 * np.exp(np.cumsum(rets))

    # Construct OHLC from close path
    noise = rng.uniform(0.0001, 0.0005, n_bars)
    opens  = np.roll(closes, 1); opens[0] = s0
    highs  = closes * (1 + noise)
    lows   = closes * (1 - noise)
    volume = rng.lognormal(mean=5, sigma=1, size=n_bars)

    now = time.time()
    timestamps = np.arange(now - n_bars * 60, now, 60)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open":  opens,
        "high":  highs,
        "low":   lows,
        "close": closes,
        "volume": volume,
    })


def compute_realised_vol(
    df: pd.DataFrame,
    window: int = 60,
    annualise_factor: float | None = None,
) -> pd.Series:
    """
    Rolling realised volatility from log returns.

    annualise_factor: bars-per-year.  If None, inferred from median
    bar duration.
    """
    if annualise_factor is None:
        dt_secs = df["timestamp"].diff().median()
        annualise_factor = 365 * 24 * 3600 / dt_secs

    log_ret = np.log(df["close"] / df["close"].shift(1))
    return log_ret.rolling(window).std() * np.sqrt(annualise_factor)


def estimate_spread_bps(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Estimate effective half-spread from OHLCV using the Roll (1984) model.

    Roll's estimator: spread ≈ 2 * sqrt(-Cov(Δp_t, Δp_{t-1}))
    This captures the bid-ask bounce in price changes.

    Returns spread in basis points.
    """
    dp = df["close"].diff()
    cov = dp.rolling(window).cov(dp.shift(1))
    spread = 2 * np.sqrt(np.maximum(-cov, 0))
    mid = df["close"].rolling(window).mean()
    return (spread / mid * 10_000).rename("spread_bps")
