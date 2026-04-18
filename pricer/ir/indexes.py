"""
IR index catalog: conventions and multi-curve parameters.

Each index entry specifies:
  ccy        : ISO currency
  reset_freq : resets per year
  daycount   : ACT360 | ACT365 | 30_360
  basis_bps  : spread over OIS for index forward curve (basis swap market)
  label      : display name

CCY_CURVES: per-currency flat OIS rates (fallback when no live curve is loaded).
"""

from __future__ import annotations

INDEX_CATALOG: dict[str, dict] = {
    # ── USD ───────────────────────────────────────────────────────────────
    "SOFR": {
        "ccy": "USD", "label": "SOFR (O/N)",
        "reset_freq": 1, "daycount": "ACT360", "basis_bps": 0,
    },
    "SOFR_3M": {
        "ccy": "USD", "label": "SOFR Compounded 3M",
        "reset_freq": 4, "daycount": "ACT360", "basis_bps": 10,
    },
    "TERM_SOFR_1M": {
        "ccy": "USD", "label": "Term SOFR 1M",
        "reset_freq": 12, "daycount": "ACT360", "basis_bps": 8,
    },
    "TERM_SOFR_3M": {
        "ccy": "USD", "label": "Term SOFR 3M",
        "reset_freq": 4, "daycount": "ACT360", "basis_bps": 10,
    },
    "TERM_SOFR_6M": {
        "ccy": "USD", "label": "Term SOFR 6M",
        "reset_freq": 2, "daycount": "ACT360", "basis_bps": 12,
    },
    "TERM_SOFR_12M": {
        "ccy": "USD", "label": "Term SOFR 12M",
        "reset_freq": 1, "daycount": "ACT360", "basis_bps": 15,
    },
    "LIBOR_3M": {
        "ccy": "USD", "label": "USD LIBOR 3M (legacy)",
        "reset_freq": 4, "daycount": "ACT360", "basis_bps": 15,
    },
    # ── EUR ───────────────────────────────────────────────────────────────
    "EUR_1M": {
        "ccy": "EUR", "label": "EURIBOR 1M",
        "reset_freq": 12, "daycount": "ACT360", "basis_bps": 20,
    },
    "EUR_3M": {
        "ccy": "EUR", "label": "EURIBOR 3M",
        "reset_freq": 4, "daycount": "ACT360", "basis_bps": 25,
    },
    "EUR_6M": {
        "ccy": "EUR", "label": "EURIBOR 6M",
        "reset_freq": 2, "daycount": "ACT360", "basis_bps": 30,
    },
    "ESTR": {
        "ccy": "EUR", "label": "€STR (O/N)",
        "reset_freq": 1, "daycount": "ACT360", "basis_bps": 0,
    },
    # ── GBP ───────────────────────────────────────────────────────────────
    "GBP_SONIA": {
        "ccy": "GBP", "label": "SONIA (O/N)",
        "reset_freq": 2, "daycount": "ACT365", "basis_bps": 0,
    },
    "GBP_SONIA_3M": {
        "ccy": "GBP", "label": "SONIA Compounded 3M",
        "reset_freq": 4, "daycount": "ACT365", "basis_bps": 5,
    },
}

# Flat OIS rates per currency — updated from FRED/market in the page layer.
CCY_CURVES: dict[str, dict[str, float]] = {
    "USD": {"ois": 0.0480},
    "EUR": {"ois": 0.0220},
    "GBP": {"ois": 0.0450},
}


def daycount_fraction(tenor_y: float, dc: str) -> float:
    """Convert year-fraction to day-count adjusted fraction."""
    if dc == "ACT360":
        return tenor_y * 365 / 360
    return tenor_y  # ACT365 / 30_360 ≈ 1 for flat curves


def index_forward_rate(base_zero_rate: float, index_key: str) -> float:
    """Add basis spread to OIS zero rate to get index forward rate."""
    basis = INDEX_CATALOG[index_key]["basis_bps"] / 10_000.0
    return base_zero_rate + basis
