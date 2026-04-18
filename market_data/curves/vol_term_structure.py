# market_data/curves/vol_term_structure.py
"""
ATM implied vol term structure and forward vol analytics.

Core concepts
─────────────
  Total variance :  w(T)        = σ_atm(T)² × T
  Forward var    :  w_fwd(T1,T2) = w(T2) - w(T1)      must be > 0 (no-arb)
  Forward vol    :  σ_fwd(T1,T2) = √[ w_fwd / (T2-T1) ]

Calendar arbitrage ← w(T) is not monotone increasing in T.
If w(T2) ≤ w(T1) for T2 > T1 the forward variance is negative (imaginary vol).
In practice this signals a mis-pricing to trade.

Usage
─────
  from market_data.curves.vol_term_structure import build_term_structure
  df = build_term_structure(quotes, spot=95000)
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from market_data.schema import OptionQuote


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _days_label(days: float) -> str:
    """Human-readable label for a tenor in days."""
    if days < 2:
        return f"{int(round(days))}D"
    if days < 14:
        return f"{int(round(days))}D"
    if days < 60:
        weeks = round(days / 7)
        return f"{weeks}W"
    if days < 365:
        months = round(days / 30.44)
        return f"{months}M"
    years = days / 365.25
    return f"{years:.1f}Y".replace(".0Y", "Y")


def _atm_iv_for_expiry(
    quotes: List[OptionQuote],
    forward: float,
) -> Optional[float]:
    """
    Extract ATM implied vol for a single expiry.
    Finds the strike closest to the forward and averages call & put IVs.
    """
    valid = [q for q in quotes if q.implied_vol and q.implied_vol > 0]
    if not valid:
        return None

    strikes = sorted(set(q.strike for q in valid))
    atm_strike = min(strikes, key=lambda k: abs(k - forward))

    atm_quotes = [q for q in valid if q.strike == atm_strike]
    ivs = [q.implied_vol for q in atm_quotes if q.implied_vol]
    if not ivs:
        return None

    return float(np.mean(ivs))


# ─────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────

def build_term_structure(
    quotes: List[OptionQuote],
    spot: float,
    rate: float = 0.0,
) -> pd.DataFrame:
    """
    Build ATM implied vol term structure and derive forward vols.

    Parameters
    ----------
    quotes : list of OptionQuote (from DeribitProvider.get_option_chain)
    spot   : current index price
    rate   : funding / lending rate for forward computation (default 0)

    Returns
    -------
    DataFrame, one row per expiry, sorted by tau, columns:
        expiry          datetime
        days            float   – calendar days to expiry
        tau             float   – years to expiry
        forward         float   – F = spot × exp(r × tau)
        atm_strike      float   – strike closest to forward
        atm_iv          float   – ATM implied vol (decimal)
        total_var       float   – w(T) = σ² × T
        fwd_vol         float | NaN  – forward vol from previous slice
        fwd_days        float | NaN  – length of forward period (days)
        fwd_label       str     – e.g. "1M → 3M"
        is_calendar_arb bool    – True when w(T) dips vs previous slice
    """
    now = datetime.utcnow()

    # Group by expiry ────────────────────────────────────────────────
    by_expiry: dict = {}
    for q in quotes:
        if q.implied_vol is None or q.implied_vol <= 0:
            continue
        by_expiry.setdefault(q.expiry, []).append(q)

    rows = []
    for expiry in sorted(by_expiry):
        tau = (expiry - now).total_seconds() / (365.25 * 24 * 3600)
        if tau < 1 / 365:          # skip same-day / expired
            continue
        days    = tau * 365.25
        forward = spot * np.exp(rate * tau)
        atm_iv  = _atm_iv_for_expiry(by_expiry[expiry], forward)
        if atm_iv is None:
            continue

        rows.append({
            "expiry":     expiry,
            "days":       days,
            "tau":        tau,
            "forward":    forward,
            "atm_strike": min(
                set(q.strike for q in by_expiry[expiry]),
                key=lambda k: abs(k - forward),
            ),
            "atm_iv":     atm_iv,
            "total_var":  atm_iv ** 2 * tau,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)
    df["tenor_label"] = df["days"].apply(_days_label)

    # Forward vol & calendar arb detection ───────────────────────────
    fwd_vol  = [np.nan]
    fwd_days = [np.nan]
    fwd_lbl  = ["–"]
    is_arb   = [False]

    for i in range(1, len(df)):
        w_prev   = df.loc[i - 1, "total_var"]
        w_curr   = df.loc[i,     "total_var"]
        tau_prev = df.loc[i - 1, "tau"]
        tau_curr = df.loc[i,     "tau"]
        lbl_prev = df.loc[i - 1, "tenor_label"]
        lbl_curr = df.loc[i,     "tenor_label"]
        dt       = tau_curr - tau_prev

        fwd_lbl.append(f"{lbl_prev} → {lbl_curr}")
        fwd_days.append((tau_curr - tau_prev) * 365.25)

        if w_curr <= w_prev + 1e-10:
            is_arb.append(True)
            fwd_vol.append(np.nan)          # imaginary – no-arb violated
        else:
            is_arb.append(False)
            fwd_vol.append(np.sqrt((w_curr - w_prev) / dt))

    df["fwd_vol"]         = fwd_vol
    df["fwd_days"]        = fwd_days
    df["fwd_label"]       = fwd_lbl
    df["is_calendar_arb"] = is_arb

    return df
