"""
Vectorized ZABR normal vol (beta=0, Antonov-Piterbarg closed form).

sigma_N(F, K, T; alpha, nu, rho) — normal implied vol.

Works on scalars or numpy arrays of any shape.
"""

from __future__ import annotations
import numpy as np


def zabr_normal_vol(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    alpha: np.ndarray,
    nu: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """
    ZABR (beta=0) closed-form normal vol.

    Parameters
    ----------
    F, K   : forward rate, strike (decimal, e.g. 0.04)
    T      : option expiry in years
    alpha  : ATM normal vol (SABR vol-of-vol analogue)
    nu     : vol-of-vol
    rho    : spot-vol correlation

    Returns
    -------
    sigma_N : normal implied vol (same units as alpha)
    """
    atm = np.abs(F - K) < 1e-9
    z = np.where(atm, 0.0, nu / np.clip(alpha, 1e-9, None) * (F - K))
    sq = np.sqrt(np.clip(1 - 2 * rho * z + z**2, 1e-12, None))
    chi = np.where(
        atm,
        1.0,
        np.log((sq + z - rho) / np.clip(1 - rho, 1e-12, None)),
    )
    z_chi = np.where(atm, 1.0, z / np.where(np.abs(chi) < 1e-12, 1e-12, chi))
    sig = alpha * z_chi * (1 + T * (2 - 3 * rho**2) / 24 * nu**2)
    return np.clip(sig, 1e-6, None)


def zabr_atm_vol(T: float, tenor: float, ccy: str = "USD") -> tuple[float, float, float]:
    """
    Realistic ZABR parameters (alpha, nu, rho) for a given expiry/tenor/ccy.

    Returns (alpha, nu, rho) — calibrated surface mimicking market shape.
    """
    ccy_base = {"EUR": 0.0038, "GBP": 0.0055}.get(ccy, 0.0060)
    hump  = 1 + 0.6 * np.exp(-((T - 2.0) ** 2) / 4)
    alpha = ccy_base * hump * (0.7 + 0.3 * np.exp(-0.06 * tenor))
    nu    = 0.18 + 0.40 * np.exp(-0.30 * T)
    rho   = -0.08 - 0.28 * np.exp(-0.45 * T)
    return float(alpha), float(nu), float(rho)


def smile_vol_strip(
    F: float,
    strikes: np.ndarray,
    T: float,
    alpha: float,
    nu: float,
    rho: float,
) -> np.ndarray:
    """Vol smile across a strike strip for a single expiry."""
    return zabr_normal_vol(
        np.full_like(strikes, F),
        strikes,
        np.full_like(strikes, T),
        np.full_like(strikes, alpha),
        np.full_like(strikes, nu),
        np.full_like(strikes, rho),
    )
