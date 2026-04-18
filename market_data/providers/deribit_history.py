# market_data/providers/deribit_history.py
"""
Historical data from Deribit:
  - DVOL index  (implied vol index, like VIX for crypto)
  - Spot OHLCV  via yfinance (BTC-USD, ETH-USD)

Deribit DVOL response: [timestamp_ms, open, high, low, close]
Values are in percent annualised (e.g. 65.2 = 65.2% IV).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

_BASE = "https://www.deribit.com/api/v2/public"

_YF_TICKER = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}


# ─────────────────────────────────────────────────────────────────────
# DVOL  (Deribit Volatility Index)
# ─────────────────────────────────────────────────────────────────────

def fetch_dvol_history(
    currency: str,
    days: int = 730,
) -> pd.Series:
    """
    Fetch daily DVOL close for *currency* over the last *days* calendar days.

    Returns a pd.Series  name='{currency}_dvol',  index=date (daily),
    values in percent (e.g. 65.2).
    """
    currency  = currency.upper()
    now_ms    = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms  = now_ms - days * 86_400_000

    resp = requests.get(
        f"{_BASE}/get_volatility_index_data",
        params={
            "currency":        currency,
            "start_timestamp": start_ms,
            "end_timestamp":   now_ms,
            "resolution":      "86400",   # daily candles
        },
        timeout=20,
    )
    resp.raise_for_status()
    raw = resp.json()["result"]["data"]  # [[ts_ms, o, h, l, c], ...]

    if not raw:
        return pd.Series(name=f"{currency}_dvol", dtype=float)

    df = pd.DataFrame(raw, columns=["ts_ms", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.date
    s = df.set_index("date")["close"].rename(f"{currency}_dvol")
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def fetch_dvol_pair(days: int = 730) -> pd.DataFrame:
    """
    Fetch BTC and ETH DVOL, align on common dates.

    Returns DataFrame with columns: btc_dvol, eth_dvol, spread, ratio
    """
    btc = fetch_dvol_history("BTC", days)
    eth = fetch_dvol_history("ETH", days)

    df = pd.concat([btc, eth], axis=1).dropna()
    df["spread"] = df["ETH_dvol"] - df["BTC_dvol"]   # ETH premium in vol pts
    df["ratio"]  = df["ETH_dvol"] / df["BTC_dvol"]   # ETH / BTC vol ratio
    return df


# ─────────────────────────────────────────────────────────────────────
# Spot prices + realized vol
# ─────────────────────────────────────────────────────────────────────

def fetch_spot_history(currency: str, days: int = 730) -> pd.DataFrame:
    """Daily OHLCV for currency via yfinance."""
    ticker = _YF_TICKER.get(currency.upper(), f"{currency.upper()}-USD")
    period = "2y" if days >= 700 else f"{days}d"
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].rename(
        columns=str.lower
    )


def fetch_realized_vol(
    currency: str,
    days: int = 730,
    window: int = 30,
) -> pd.Series:
    """
    Rolling *window*-day realized vol (annualised %) from spot log-returns.
    """
    spot    = fetch_spot_history(currency, days + window + 10)
    log_ret = np.log(spot["close"] / spot["close"].shift(1))
    rv      = log_ret.rolling(window).std() * np.sqrt(365) * 100
    return rv.rename(f"{currency}_rv{window}").dropna()
