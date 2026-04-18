# market_data/providers/fred.py
"""
Fetches US Treasury par yield curve from the FRED REST API (free, no library needed).
Register for a free API key at https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
import requests
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# FRED series_id -> tenor in years
_SERIES = {
    "DGS1MO":  1 / 12,
    "DGS3MO":  3 / 12,
    "DGS6MO":  6 / 12,
    "DGS1":    1.0,
    "DGS2":    2.0,
    "DGS3":    3.0,
    "DGS5":    5.0,
    "DGS7":    7.0,
    "DGS10":  10.0,
    "DGS20":  20.0,
    "DGS30":  30.0,
}

_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Approximate fallback curve (US, early 2025) — used when FRED is unreachable
_FALLBACK_CURVE: Dict[float, float] = {
    1 / 12: 0.0433,
    3 / 12: 0.0427,
    6 / 12: 0.0420,
    1.0:    0.0410,
    2.0:    0.0405,
    3.0:    0.0408,
    5.0:    0.0420,
    7.0:    0.0430,
    10.0:   0.0445,
    20.0:   0.0480,
    30.0:   0.0470,
}


def _fetch_latest(series_id: str, api_key: str) -> Optional[float]:
    """Return most recent non-null observation for a FRED series (as decimal rate)."""
    try:
        resp = requests.get(
            _BASE,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,  # fetch a few in case latest is a holiday (null)
            },
            timeout=10,
        )
        resp.raise_for_status()
        for obs in resp.json()["observations"]:
            if obs["value"] != ".":
                return float(obs["value"]) / 100.0  # percent -> decimal
    except Exception as exc:
        print(f"[FRED] {series_id}: {exc}")
    return None


def fetch_usd_curve(api_key: Optional[str] = None) -> Dict[float, float]:
    """
    Fetch US Treasury par yield curve from FRED.

    Returns {tenor_years: rate} sorted by tenor (rates in decimal, e.g. 0.042).
    Falls back to a hardcoded approximate curve if FRED is unreachable or no key is set.

    api_key: FRED API key. If None, reads FRED_API_KEY env var.
    """
    key = api_key or os.environ.get("FRED_API_KEY", "")
    curve: Dict[float, float] = {}

    if key:
        for series_id, tenor in _SERIES.items():
            rate = _fetch_latest(series_id, key)
            if rate is not None:
                curve[tenor] = rate

    if len(curve) < 3:
        print("[FRED] insufficient live data — using hardcoded fallback curve.")
        print("[FRED] Tip: set FRED_API_KEY env var or pass api_key for live rates.")
        return dict(sorted(_FALLBACK_CURVE.items()))

    return dict(sorted(curve.items()))
