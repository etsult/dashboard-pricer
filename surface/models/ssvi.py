"""
SSVI — Surface SVI (Gatheral & Jacquier 2014).

Power-law parameterization:
  w(k, θ) = θ/2 * (1 + ρ·φ·k + sqrt((φ·k + ρ)² + 1 − ρ²))
  φ(θ)    = η / (θ^γ · (1+θ)^(1-γ))

Global parameters: (ρ, η, γ)
Per-expiry ATM total variance θ_T = σ_ATM(T)² · T (estimated from data).

No-butterfly-arb sufficient condition: 0 < φ·(1+|ρ|) < 4 per expiry.
Inherently smooth across expiries and strikes.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ── SSVI formula ──────────────────────────────────────────────────────────────

def _phi(theta: float | np.ndarray, eta: float, gamma: float) -> float | np.ndarray:
    return eta / (theta ** gamma * (1 + theta) ** (1 - gamma))


def _ssvi_w(k: np.ndarray, theta: float, rho: float, eta: float, gamma: float) -> np.ndarray:
    phi = _phi(theta, eta, gamma)
    return theta / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2))


def _ssvi_iv(k: np.ndarray, T: float, theta: float, rho: float, eta: float, gamma: float) -> np.ndarray:
    w = _ssvi_w(k, theta, rho, eta, gamma)
    return np.sqrt(np.maximum(w / T, 1e-8))


# ── ATM total variance per expiry ─────────────────────────────────────────────

def _estimate_theta(T: float, k: np.ndarray, iv: np.ndarray) -> float:
    """Interpolate or extrapolate ATM total variance."""
    order = np.argsort(np.abs(k))
    k_sorted = k[order][:10]
    iv_sorted = iv[order][:10]
    iv_atm = float(np.interp(0.0, k_sorted, iv_sorted)) if k_sorted.min() < 0.05 else iv_sorted[0]
    return iv_atm ** 2 * T


# ── Global fit ────────────────────────────────────────────────────────────────

def _fit_global(raw: pd.DataFrame) -> tuple[float, float, float]:
    """Fit (rho, eta, gamma) to all expiries jointly."""
    groups = [(T, grp["k"].values, grp["iv"].values) for T, grp in raw.groupby("T")]
    thetas = {T: _estimate_theta(T, k, iv) for T, k, iv in groups}

    def obj(x):
        rho, eta, gamma = x
        if abs(rho) >= 0.999 or eta <= 0 or gamma <= 0 or gamma >= 1:
            return 1e8
        total = 0.0
        for T, k, iv in groups:
            theta = thetas[T]
            phi = _phi(theta, eta, gamma)
            if phi * (1 + abs(rho)) >= 4:
                return 1e8
            pred_iv = _ssvi_iv(k, T, theta, rho, eta, gamma)
            total += np.mean((pred_iv - iv) ** 2)
        return total

    x0 = np.array([-0.4, 1.0, 0.5])
    bounds = [(-0.999, 0.999), (0.01, 5.0), (0.01, 0.99)]

    best_val, best_x = np.inf, x0
    for rho0 in [-0.6, -0.3, 0.0]:
        for eta0 in [0.5, 1.5]:
            x0_ = np.array([rho0, eta0, 0.5])
            res = minimize(obj, x0_, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 1000, "ftol": 1e-14})
            if res.fun < best_val:
                best_val, best_x = res.fun, res.x

    return tuple(best_x)


# ── Public API ────────────────────────────────────────────────────────────────

def fit_ssvi_surface(
    raw: pd.DataFrame,
    k_grid: np.ndarray,
    T_grid: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit SSVI globally; evaluate on dense (k_grid, T_grid).

    Returns
    -------
    surf_df   : DataFrame(index=k_grid, columns=T_grid)
    params_df : single-row DataFrame with global params + per-expiry theta/rmse
    raw       : unchanged input
    """
    rho, eta, gamma = _fit_global(raw)

    # Evaluate on output grid
    thetas_out = np.array([
        _estimate_theta(T, raw[raw["T"] == T]["k"].values, raw[raw["T"] == T]["iv"].values)
        if T in raw["T"].values else (raw.groupby("T")["iv"].mean().mean()) ** 2 * T
        for T in T_grid
    ])

    z = np.full((len(k_grid), len(T_grid)), np.nan)
    for i_T, (T_eval, theta) in enumerate(zip(T_grid, thetas_out)):
        z[:, i_T] = _ssvi_iv(k_grid, T_eval, theta, rho, eta, gamma)

    surf_df = pd.DataFrame(z, index=k_grid, columns=T_grid)

    # Per-expiry RMSE
    rmse_rows = []
    for T, grp in raw.groupby("T"):
        theta = _estimate_theta(T, grp["k"].values, grp["iv"].values)
        pred = _ssvi_iv(grp["k"].values, T, theta, rho, eta, gamma)
        rmse = float(np.sqrt(np.mean((pred - grp["iv"].values) ** 2))) * 100
        rmse_rows.append({"T": T, "theta": theta, "rmse_iv_pct": rmse})

    params_df = pd.DataFrame([{
        "rho": rho, "eta": eta, "gamma": gamma,
        "mean_rmse_pct": np.mean([r["rmse_iv_pct"] for r in rmse_rows]),
    }])

    return surf_df, params_df, raw
