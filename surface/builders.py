# surface/builders.py
"""
Implied vol surface builder.

Steps:
  1. Clean raw OptionQuote list (filter illiquid, zero-bid, extreme moneyness).
  2. Convert to per-expiry (k, iv) pairs in total-variance space.
  3. Dispatch to the chosen parametric model (SVI / SSVI / Heston).
  4. Return (surface_df, raw_df, params_df) for display.

surface_df : index = log-moneyness grid, columns = T (years), values = IV (decimal)
raw_df     : one row per clean market point for scatter overlay plots
params_df  : model calibration output (format depends on model)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from math import log, sqrt, exp

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List

from market_data.schema import OptionQuote
from surface.models import MODEL_REGISTRY

STORAGE_DIR = Path(__file__).resolve().parents[1] / "storage"

# Output grids
K_GRID = np.linspace(-0.50, 0.50, 100)   # log-moneyness [-50%, +50%]
N_T    = 30                                # number of expiry slices on output


# ── Cleaning ──────────────────────────────────────────────────────────────────

def _clean(
    quotes: List[OptionQuote],
    spot: float,
    rate: float,
    div: float,
    now: datetime,
) -> pd.DataFrame:
    """
    Convert quotes to a clean DataFrame with columns [expiry, T, k, iv].

    Filters applied:
    - Drop if both bid and ask are 0 or None (no market)
    - Drop if IV is NaN / <= 0 / > 3.0 (300%)
    - Drop if T < 3 days (avoid pinning near expiry)
    - Drop moneyness |k| > 0.50 (deep OTM/ITM, unreliable)
    - Per-expiry: keep only call or put at each strike (use put below forward, call above)
    - Per-expiry: IQR filter (remove IV outliers ±2×IQR around median)
    - Require ≥ 5 points per expiry straddling ATM
    """
    rows = []
    for q in quotes:
        bid = q.bid or 0.0
        ask = q.ask or 0.0
        iv  = q.implied_vol

        if bid <= 0 and ask <= 0:
            continue
        if iv is None or np.isnan(iv) or iv <= 0.001 or iv > 3.0:
            continue

        T = (q.expiry - now).total_seconds() / (365.25 * 86400)
        if T < 3 / 365:
            continue

        F = spot * exp((rate - div) * T)
        k = log(q.strike / F)
        if abs(k) > 0.55:
            continue

        rows.append({
            "expiry": q.expiry,
            "T":      T,
            "k":      k,
            "iv":     iv,
            "strike": q.strike,
            "F":      F,
            "type":   q.option_type,
            "bid":    bid,
            "ask":    ask,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Use OTM options only: put if k < 0, call if k > 0 (avoid put-call duality artifacts)
    df = df[((df["k"] <= 0) & (df["type"] == "put")) |
            ((df["k"] >= 0) & (df["type"] == "call"))]

    # Per-expiry IQR filter
    clean_rows = []
    for _, grp in df.groupby("expiry"):
        q25, q75 = grp["iv"].quantile(0.25), grp["iv"].quantile(0.75)
        iqr = q75 - q25
        lo, hi = q25 - 2.0 * iqr, q75 + 2.0 * iqr
        grp = grp[(grp["iv"] >= max(lo, 0.001)) & (grp["iv"] <= hi)]
        if len(grp) < 5:
            continue
        if not ((grp["k"] < -0.01).any() and (grp["k"] > 0.01).any()):
            continue
        clean_rows.append(grp)

    if not clean_rows:
        return pd.DataFrame()

    return pd.concat(clean_rows, ignore_index=True)


# ── Public API ────────────────────────────────────────────────────────────────

def build_vol_surface(
    quotes: List[OptionQuote],
    spot: float,
    rate: float = 0.04,
    dividend_yield: float = 0.0,
    model: str = "SVI",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build a smooth implied vol surface from raw option quotes.

    Parameters
    ----------
    quotes         : raw option chain from any provider
    spot           : underlying spot / index level
    rate           : risk-free rate (decimal)
    dividend_yield : continuous dividend yield
    model          : "SVI" | "SSVI" | "Heston"

    Returns
    -------
    surf_df   : DataFrame — index=log-moneyness, columns=T (years), values=IV
    params_df : model calibration params / diagnostics
    raw_df    : clean market points (for scatter overlay)
    """
    now = datetime.utcnow()
    raw_df = _clean(quotes, spot, rate, dividend_yield, now)

    if raw_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Build T_grid from actual expiry buckets
    T_vals = np.sort(raw_df["T"].unique())
    if len(T_vals) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    T_grid = np.linspace(T_vals.min(), T_vals.max(), N_T)

    # Dispatch to chosen model
    fitter = MODEL_REGISTRY.get(model)
    if fitter is None:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(MODEL_REGISTRY)}")

    surf_df, params_df, _ = fitter(raw_df, K_GRID, T_grid)

    # Persist debug files
    try:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(STORAGE_DIR / "last_option_quotes.csv", index=False)
        surf_df.to_csv(STORAGE_DIR / "last_vol_surface.csv")
    except OSError:
        pass

    return surf_df, params_df, raw_df
