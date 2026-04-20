"""
Heston (1993) stochastic volatility model.

Pricing via the classical two-probability CF representation (Heston 1993 §2).
Numerically stable: uses OTM options throughout (put for k<0, call for k≥0).
Calibration: L-BFGS-B on (v0, kappa, theta, sigma, rho).

Full surface calibration ~10–30 seconds.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize, brentq
from scipy.stats import norm as _norm

# Fixed 128-point GL quadrature on [0, U_MAX]
_N_GL  = 128
_U_MAX = 100.0
_gl_x, _gl_w = leggauss(_N_GL)
_U_NODES = _U_MAX * (_gl_x + 1) / 2
_U_WGTS  = _gl_w * _U_MAX / 2


# ── Heston characteristic function (two-prob form) ────────────────────────────

def _heston_pj(u: np.ndarray, T, F, K, v0, kappa, theta, sigma, rho, j: int) -> np.ndarray:
    """
    Returns the integrand Re[φ_j(u)] / u for probability P_j.
    j=1: underlying probability  (drift adjusted)
    j=2: risk-neutral probability (standard)
    """
    uj = 0.5 if j == 1 else -0.5
    bj = (kappa - rho * sigma) if j == 1 else kappa

    d = np.sqrt((rho * sigma * 1j * u - bj) ** 2
                - sigma ** 2 * (2j * uj * u - u ** 2))

    # Numerically stable "g" formulation
    r_m = bj - rho * sigma * 1j * u - d
    r_p = bj - rho * sigma * 1j * u + d
    g   = r_m / r_p

    exp_dt = np.exp(-d * T)
    D = r_m / sigma ** 2 * (1 - exp_dt) / (1 - g * exp_dt)
    C = kappa * theta / sigma ** 2 * (
        r_m * T - 2 * np.log((1 - g * exp_dt) / (1 - g))
    )

    phi = np.exp(C + D * v0 + 1j * u * np.log(F / K))
    return np.real(phi / (1j * u))


def _heston_price(F, K, T, v0, kappa, theta, sigma, rho) -> float:
    """Undiscounted Heston call price via two-CF GL integration."""
    u = _U_NODES + 1e-8  # avoid u=0 singularity
    P1 = 0.5 + 1 / np.pi * float((_heston_pj(u, T, F, K, v0, kappa, theta, sigma, rho, 1) * _U_WGTS).sum())
    P2 = 0.5 + 1 / np.pi * float((_heston_pj(u, T, F, K, v0, kappa, theta, sigma, rho, 2) * _U_WGTS).sum())
    return F * P1 - K * P2


# ── OTM-aware IV grid ─────────────────────────────────────────────────────────

def heston_iv_grid(
    k_arr: np.ndarray,
    T: float,
    v0, kappa, theta, sigma, rho,
) -> np.ndarray:
    """
    Implied vols from Heston for each log-moneyness k = log(K/F).
    Uses OTM call for k≥0, OTM put for k<0 (numerically stable).
    """
    ivs = np.zeros_like(k_arr, dtype=float)
    sqrt_v0 = np.sqrt(max(v0, 1e-6))

    for i, k in enumerate(k_arr):
        Kval = np.exp(k)  # F=1 normalised
        call_price = _heston_price(1.0, Kval, T, v0, kappa, theta, sigma, rho)
        call_price = float(np.real(call_price))

        if k < 0:
            # OTM put via put-call parity: P = C - (F - K) = C - (1 - Kval)
            otm_price = call_price - (1.0 - Kval)
            otm_price = max(otm_price, 1e-8)

            def _bs_put(iv):
                vol_t = iv * np.sqrt(T)
                if vol_t < 1e-10:
                    return max(Kval - 1.0, 0.0) - otm_price
                d1 = (-k + vol_t ** 2 / 2) / vol_t
                d2 = d1 - vol_t
                return Kval * _norm.cdf(-d2) - _norm.cdf(-d1) - otm_price

            try:
                lo, hi = 0.001, 5.0
                if _bs_put(lo) * _bs_put(hi) > 0:
                    ivs[i] = sqrt_v0
                else:
                    ivs[i] = brentq(_bs_put, lo, hi, xtol=1e-6)
            except Exception:
                ivs[i] = sqrt_v0

        else:
            # OTM call
            otm_price = max(call_price, 1e-8)
            otm_price = min(otm_price, 1.0 - 1e-8)

            def _bs_call(iv):
                vol_t = iv * np.sqrt(T)
                if vol_t < 1e-10:
                    return max(1.0 - Kval, 0.0) - otm_price
                d1 = (-k + vol_t ** 2 / 2) / vol_t
                d2 = d1 - vol_t
                return _norm.cdf(d1) - Kval * _norm.cdf(d2) - otm_price

            try:
                lo, hi = 0.001, 5.0
                if _bs_call(lo) * _bs_call(hi) > 0:
                    ivs[i] = sqrt_v0
                else:
                    ivs[i] = brentq(_bs_call, lo, hi, xtol=1e-6)
            except Exception:
                ivs[i] = sqrt_v0

    return np.clip(ivs, 0.001, 5.0)


# ── Calibration ───────────────────────────────────────────────────────────────

def _calibrate(raw: pd.DataFrame) -> tuple:
    groups = [(T, grp["k"].values, grp["iv"].values) for T, grp in raw.groupby("T")]
    # Sub-sample to ≤10 expiries for speed
    if len(groups) > 10:
        step = max(1, len(groups) // 10)
        groups = groups[::step]

    all_ivs = np.concatenate([iv for _, _, iv in groups])
    iv0 = float(np.median(all_ivs))

    def obj(x):
        v0, kappa, theta, sigma, rho = x
        if any([v0 <= 0, kappa <= 0, theta <= 0, sigma <= 0, abs(rho) >= 1]):
            return 1e8
        total = 0.0
        for T, k_exp, iv_exp in groups:
            mask = np.abs(k_exp) < 0.35
            if mask.sum() < 3:
                mask = np.ones(len(k_exp), dtype=bool)
            k_s, iv_s = k_exp[mask], iv_exp[mask]
            try:
                pred = heston_iv_grid(k_s, T, v0, kappa, theta, sigma, rho)
                total += float(np.mean((pred - iv_s) ** 2))
            except Exception:
                total += 1.0
        return total

    bounds = [
        (0.0001, 1.0),  # v0
        (0.10, 10.0),   # kappa
        (0.0001, 1.0),  # theta
        (0.01, 2.0),    # sigma
        (-0.99, 0.99),  # rho
    ]
    x0 = [iv0 ** 2, 1.5, iv0 ** 2, 0.4, -0.7]

    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 400, "ftol": 1e-10})
    return tuple(res.x), float(res.fun)


# ── Public API ────────────────────────────────────────────────────────────────

def fit_heston_surface(
    raw: pd.DataFrame,
    k_grid: np.ndarray,
    T_grid: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params, loss = _calibrate(raw)
    v0, kappa, theta, sigma, rho = params

    z = np.full((len(k_grid), len(T_grid)), np.nan)
    for i_T, T_eval in enumerate(T_grid):
        z[:, i_T] = heston_iv_grid(k_grid, T_eval, v0, kappa, theta, sigma, rho)

    surf_df = pd.DataFrame(z, index=k_grid, columns=T_grid)

    rmse_rows = []
    for T, grp in raw.groupby("T"):
        mask = np.abs(grp["k"].values) < 0.35
        k_s = grp["k"].values[mask] if mask.any() else grp["k"].values
        iv_s = grp["iv"].values[mask] if mask.any() else grp["iv"].values
        pred = heston_iv_grid(k_s, T, v0, kappa, theta, sigma, rho)
        rmse = float(np.sqrt(np.mean((pred - iv_s) ** 2))) * 100
        rmse_rows.append({"T": round(T, 4), "rmse_iv_pct": round(rmse, 3)})

    feller_ok = 2 * kappa * theta > sigma ** 2
    params_df = pd.DataFrame([{
        "v0": round(v0, 6), "kappa": round(kappa, 4), "theta": round(theta, 6),
        "sigma": round(sigma, 4), "rho": round(rho, 4),
        "feller_ok": feller_ok,
        "mean_rmse_pct": round(float(np.mean([r["rmse_iv_pct"] for r in rmse_rows])), 3),
    }])

    return surf_df, params_df, raw
