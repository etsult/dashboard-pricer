"""
Shared test fixtures for IR pricer consistency tests.

All fixture functions are pure (no side effects) and seeded for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pricer.ir.instruments import IRPosition, Book
from pricer.ir.indexes import INDEX_CATALOG
from pricer.ir.zabr import zabr_normal_vol

# ── Representative index subsets ──────────────────────────────────────────────

USD_INDEXES = [k for k, v in INDEX_CATALOG.items() if v["ccy"] == "USD"]
EUR_INDEXES = [k for k, v in INDEX_CATALOG.items() if v["ccy"] == "EUR"]
ALL_INDEXES = USD_INDEXES + EUR_INDEXES

# All (instrument, index) combinations for parametric tests
SWAPTION_CASES = [
    (instr, idx)
    for instr in ("payer_swaption", "receiver_swaption")
    for idx in ALL_INDEXES
]
CAPFLOOR_CASES = [
    (instr, idx)
    for instr in ("cap", "floor")
    for idx in ALL_INDEXES
]


# ── Formula-level samples ─────────────────────────────────────────────────────

def make_formula_samples(n: int = 500, seed: int = 0) -> dict[str, np.ndarray]:
    """
    Generate n single-option samples for Bachelier formula comparison.

    Returns dict of float32 arrays:
      F, K, sigma_n, tau, is_call, expected_moneyness
    """
    rng = np.random.default_rng(seed)

    F       = rng.uniform(0.01, 0.08, n).astype(np.float64)
    spreads = rng.uniform(-0.015, 0.015, n)
    K       = np.clip(F + spreads, 1e-4, None).astype(np.float64)

    # ZABR-derived sigma_n for realism
    alpha = rng.uniform(0.002, 0.012, n)
    nu    = rng.uniform(0.10, 0.60, n)
    rho   = rng.uniform(-0.50, 0.10, n)
    tau   = rng.uniform(0.25, 10.0, n).astype(np.float64)
    sigma_n = zabr_normal_vol(F, K, tau, alpha, nu, rho)

    is_call = rng.integers(0, 2, n).astype(bool)

    return {
        "F": F, "K": K, "sigma_n": sigma_n,
        "tau": tau, "is_call": is_call,
        "alpha": alpha, "nu": nu, "rho": rho,
    }


# ── Book-level fixtures ───────────────────────────────────────────────────────

def _make_position(
    rng: np.random.Generator,
    instrument: str,
    index_key: str,
) -> IRPosition:
    meta   = INDEX_CATALOG[index_key]
    ccy    = meta["ccy"]
    freq   = 1.0 / meta["reset_freq"]
    basis  = meta["basis_bps"] / 10_000.0

    # Realistic ATM forward by CCY
    F_atm  = {"USD": 0.0450, "EUR": 0.0235, "GBP": 0.0455}.get(ccy, 0.04)
    expiry = float(rng.choice([0.25, 0.5, 1, 2, 3, 5, 7, 10]))
    tenor  = float(rng.choice([1, 2, 3, 5, 10]) if "swaption" in instrument
                   else rng.choice([2, 3, 5, 7, 10]))

    K      = float(np.clip(F_atm + rng.uniform(-0.015, 0.015), 1e-4, None))
    alpha, nu, rho = _zabr_params(expiry, ccy)
    sigma_n = float(zabr_normal_vol(
        np.array([F_atm]), np.array([K]),
        np.array([expiry]), np.array([alpha]),
        np.array([nu]),     np.array([rho]),
    )[0])
    notional  = float(rng.choice([10e6, 25e6, 50e6, 100e6]))
    direction = int(rng.choice([1, -1]))

    return IRPosition(
        instrument=instrument,
        index_key=index_key,
        notional=notional,
        strike=K,
        expiry_y=expiry,
        tenor_y=tenor,
        sigma_n=sigma_n,
        direction=direction,
    )


def _zabr_params(T: float, ccy: str) -> tuple[float, float, float]:
    ccy_base = {"EUR": 0.0038, "GBP": 0.0055}.get(ccy, 0.0060)
    hump  = 1 + 0.6 * np.exp(-((T - 2.0) ** 2) / 4)
    alpha = ccy_base * hump
    nu    = 0.18 + 0.40 * np.exp(-0.30 * T)
    rho   = -0.08 - 0.28 * np.exp(-0.45 * T)
    return float(alpha), float(nu), float(rho)


def make_swaption_book(
    n: int = 100,
    seed: int = 0,
    instruments: list[str] | None = None,
    indexes: list[str] | None = None,
) -> Book:
    """Generate a Book of swaption positions."""
    rng = np.random.default_rng(seed)
    instrs = instruments or ["payer_swaption", "receiver_swaption"]
    idxs   = indexes or ALL_INDEXES
    positions = [
        _make_position(rng, rng.choice(instrs), rng.choice(idxs))
        for _ in range(n)
    ]
    return Book(positions=positions)


def make_capfloor_book(
    n: int = 100,
    seed: int = 1,
    instruments: list[str] | None = None,
    indexes: list[str] | None = None,
) -> Book:
    """Generate a Book of cap/floor positions."""
    rng = np.random.default_rng(seed)
    instrs = instruments or ["cap", "floor"]
    idxs   = indexes or ALL_INDEXES
    positions = [
        _make_position(rng, rng.choice(instrs), rng.choice(idxs))
        for _ in range(n)
    ]
    return Book(positions=positions)


def make_mixed_book(n: int = 200, seed: int = 2) -> Book:
    """Generate a full mixed book (all products, all indexes)."""
    rng = np.random.default_rng(seed)
    all_instrs = ["payer_swaption", "receiver_swaption", "cap", "floor"]
    positions = [
        _make_position(rng, rng.choice(all_instrs), rng.choice(ALL_INDEXES))
        for _ in range(n)
    ]
    return Book(positions=positions)


# ── NN-level samples ──────────────────────────────────────────────────────────

def make_nn_samples(
    n: int = 500,
    seed: int = 0,
    convention: str = "USD_SOFR",   # "USD_SOFR" | "EUR_EURIB" | "GBP_SONIA"
) -> dict[str, np.ndarray]:
    """
    Generate samples compatible with SwaptionNet v1 inference.

    Returns continuous features + target normalised price + raw price.
    Convention mapping: USD_SOFR→idx 0, EUR_EURIB→1, GBP_SONIA→2.
    """
    CONV_MAP = {"USD_SOFR": 0, "EUR_EURIB": 1, "GBP_SONIA": 2}
    CCY_MAP  = {"USD_SOFR": "USD", "EUR_EURIB": "EUR", "GBP_SONIA": "GBP"}
    conv_idx = CONV_MAP[convention]
    ccy      = CCY_MAP[convention]

    rng = np.random.default_rng(seed)

    T       = rng.uniform(0.25, 10.0, n).astype(np.float64)
    tenor   = rng.choice([1.0, 2.0, 3.0, 5.0, 7.0, 10.0], n).astype(np.float64)
    F_atm   = {"USD": 0.0450, "EUR": 0.0235, "GBP": 0.0455}[ccy]
    F       = np.full(n, F_atm) + rng.uniform(-0.01, 0.01, n)
    F       = np.clip(F, 1e-4, None).astype(np.float64)
    spreads = rng.uniform(-0.015, 0.015, n)
    K       = np.clip(F + spreads, 1e-4, None).astype(np.float64)

    alpha = rng.uniform(0.002, 0.012, n)
    nu    = rng.uniform(0.10, 0.60, n)
    rho   = rng.uniform(-0.50, 0.10, n)

    sigma_n  = zabr_normal_vol(F, K, T, alpha, nu, rho)
    sqT      = np.sqrt(np.clip(T, 1e-8, None))
    d        = (F - K) / np.clip(sigma_n * sqT, 1e-12, None)
    is_payer = rng.integers(0, 2, n).astype(bool)

    # Bachelier raw price (ground truth)
    from scipy.special import ndtr
    pdf = np.exp(-0.5 * d**2) / np.sqrt(2 * np.pi)
    raw_price = np.where(
        is_payer,
        (F - K) * ndtr(d)   + sigma_n * sqT * pdf,
        (K - F) * ndtr(-d)  + sigma_n * sqT * pdf,
    )
    norm_price = raw_price / np.clip(sigma_n * sqT, 1e-12, None)

    return {
        # Continuous features (v1 model order)
        "moneyness":  d.astype(np.float32),
        "log_mk":     np.log(K / F).astype(np.float32),
        "T":          T.astype(np.float32),
        "tenor":      tenor.astype(np.float32),
        "sigma_atm":  alpha.astype(np.float32),
        "nu":         nu.astype(np.float32),
        "rho":        rho.astype(np.float32),
        "is_payer":   is_payer.astype(np.float32),
        # Convention index
        "conv_idx":   np.full(n, conv_idx, dtype=np.int64),
        # Targets
        "sigma_n":    sigma_n.astype(np.float32),
        "norm_price": norm_price.astype(np.float32),
        "raw_price":  raw_price.astype(np.float32),
        # Metadata
        "F":          F.astype(np.float32),
        "K":          K.astype(np.float32),
        "is_payer_bool": is_payer,
    }
