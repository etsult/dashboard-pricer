# surface/builders.py

import os
from pathlib import Path
from datetime import datetime
from math import log, sqrt, exp
import pandas as pd
import numpy as np
from typing import List
from scipy.stats import norm
from scipy.interpolate import griddata
from market_data.schema import OptionQuote


STORAGE_DIR = Path(__file__).resolve().parents[1] / "storage"


def _bs_delta(spot: float, strike: float, t: float, vol: float, rate: float, div: float, option_type: str) -> float:
    """Black-Scholes delta for calls (+) and puts (-)."""
    if vol <= 0 or t <= 0 or spot <= 0 or strike <= 0:
        return np.nan
    try:
        d1 = (log(spot / strike) + (rate - div + 0.5 * vol * vol) * t) / (vol * sqrt(t))
    except Exception:
        return np.nan
    disc = exp(-div * t)
    if option_type.lower() == "call":
        return disc * norm.cdf(d1)
    else:
        return disc * (norm.cdf(d1) - 1.0)


def build_vol_surface(quotes: List[OptionQuote], spot: float, rate: float = 0.02, dividend_yield: float = 0.0):
    """
    Build a moneyness-based implied volatility surface:
    - compute forward F(T) from spot/rate/dividend
    - use log-moneyness ln(K/F) as x-axis (handles call/put symmetry)
    - per-expiry filtering in log-moneyness space
    - quadratic fit per expiry
    - bilinear interpolation onto a regular (log-moneyness, days) grid
    returns: DataFrame indexed by log-moneyness, columns as expiry datetimes, values IV
    """

    now = datetime.utcnow()

    # Convert to DataFrame (keep option type for stats/debug)
    df_raw = pd.DataFrame([{
        "expiry": q.expiry,
        "strike": q.strike,
        "iv": q.implied_vol
    } for q in quotes])

    # Persist raw pull (before any filtering) for debugging
    try:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(STORAGE_DIR / "last_yahoo_raw.csv", index=False)
    except OSError as exc:
        print(f"[vol-surface] ⚠️ could not write raw yahoo CSV: {exc}")

    df = df_raw.copy()

    # Drop missing IV
    df = df.dropna(subset=["iv"])

    # Remove absurd IV values (Yahoo has glitches)
    df = df[(df["iv"] > 0.001) & (df["iv"] < 2.0)]  # keep 0.1% to 200%

    # Compute time to expiry (years) and forward, then log-moneyness
    df["t"] = (df["expiry"] - now).dt.total_seconds() / (365.25 * 24 * 3600)
    df["t"] = df["t"].clip(lower=1 / 365)  # min 1 day to avoid zero maturity
    df["forward"] = spot * np.exp((rate - dividend_yield) * df["t"])
    df["log_m"] = np.log(df["strike"] / df["forward"])

    # Per-expiry filtering in log-moneyness space
    fit_rows = []
    per_expiry_stats = []

    log_m_grid = np.linspace(-0.5, 0.5, 81)  # approx strike from ~60% to ~165% of F

    for expiry, g in df.groupby("expiry"):
        # basic band on moneyness
        g = g[(g["log_m"] > log_m_grid.min() - 0.05) & (g["log_m"] < log_m_grid.max() + 0.05)]
        if len(g) < 4:
            continue

        mean_iv = g["iv"].mean()
        std_iv = g["iv"].std()
        if not np.isnan(std_iv) and std_iv > 0:
            g = g[(g["iv"] > mean_iv - 2 * std_iv) & (g["iv"] < mean_iv + 2 * std_iv)]

        if len(g) < 4:
            continue
        if not ((g["log_m"] < 0).any() and (g["log_m"] > 0).any()):
            continue  # need both wings around ATM

        try:
            coef = np.polyfit(g["log_m"], g["iv"], 2)
        except Exception:
            continue

        # sanity on curvature (avoid explosive fits)
        if abs(coef[0]) > 10:
            continue

        iv_fit = np.polyval(coef, log_m_grid)
        iv_fit = np.maximum(iv_fit, 0.0001)
        t_days = (expiry - df["expiry"].min()).days
        for m_val, iv_val in zip(log_m_grid, iv_fit):
            fit_rows.append({"expiry": expiry, "days": t_days, "log_m": m_val, "iv": iv_val})

        per_expiry_stats.append(
            {
                "expiry": expiry,
                "n_points": len(g),
                "coef_a": coef[0],
                "coef_b": coef[1],
                "coef_c": coef[2],
                "mean_iv": mean_iv,
                "std_iv": std_iv,
            }
        )

    if not fit_rows:
        print("[vol-surface] ❌ no expiries passed filtering; returning empty surface")
        return pd.DataFrame()

    fit_df = pd.DataFrame(fit_rows)

    # Regularize on a daily expiry grid using bilinear interpolation
    min_day = fit_df["days"].min()
    max_day = fit_df["days"].max()
    expiry_grid_days = np.linspace(min_day, max_day, int(max_day - min_day + 1)) if max_day > min_day else np.array([min_day])

    grid_x, grid_y = np.meshgrid(log_m_grid, expiry_grid_days, indexing="xy")
    points = fit_df[["log_m", "days"]].values
    values = fit_df["iv"].values

    grid_z = griddata(points, values, (grid_x, grid_y), method="linear")
    if np.isnan(grid_z).any():
        nearest = griddata(points, values, (grid_x, grid_y), method="nearest")
        grid_z = np.where(np.isnan(grid_z), nearest, grid_z)

    expiry_dates = [df["expiry"].min() + pd.Timedelta(days=float(d)) for d in expiry_grid_days]
    surface = pd.DataFrame(grid_z, columns=log_m_grid, index=expiry_dates).T  # index=log-m, columns=dates

    # -------- Debug + persistence ---------------------------------
    try:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(STORAGE_DIR / "last_option_quotes.csv", index=False)
        fit_df.to_csv(STORAGE_DIR / "last_vol_surface_fit_points.csv", index=False)
        surface.to_csv(STORAGE_DIR / "last_vol_surface.csv")
        pd.DataFrame(per_expiry_stats).to_csv(STORAGE_DIR / "last_vol_surface_fit_stats.csv", index=False)
    except OSError as exc:
        print(f"[vol-surface] ⚠️ could not write debug CSVs: {exc}")

    print(f"[vol-surface] raw rows={len(df_raw)}, post-filter rows={len(df)}")
    print(f"[vol-surface] expiries kept={len(per_expiry_stats)}, log-m grid points per expiry={len(log_m_grid)}")
    print(f"[vol-surface] surface shape={surface.shape} (log-m x expiry)")

    return surface
