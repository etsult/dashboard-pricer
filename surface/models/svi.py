"""
SVI (Stochastic Volatility Inspired) — Gatheral 2004.

Total variance: w(k) = a + b*(ρ*(k-m) + sqrt((k-m)² + σ²))
  k = log(K/F),  w = σ_BS² * T

Fit independently per expiry. Fast (~30ms each), no-arb constraints.

No-arb conditions (sufficient):
  b ≥ 0, |ρ| < 1, σ > 0, a + b*σ*sqrt(1-ρ²) ≥ 0  (minimum w ≥ 0)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ── SVI formula ───────────────────────────────────────────────────────────────

def _svi_w(k: np.ndarray, a: float, b: float, rho: float, m: float, sig: float) -> np.ndarray:
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2))


def _svi_iv(k: np.ndarray, T: float, a, b, rho, m, sig) -> np.ndarray:
    w = _svi_w(k, a, b, rho, m, sig)
    return np.sqrt(np.maximum(w / T, 1e-8))


# ── Per-expiry fit ────────────────────────────────────────────────────────────

def _fit_one(k: np.ndarray, w: np.ndarray, T: float) -> tuple[np.ndarray, float]:
    """Fit SVI to (k, w) data for one expiry. Returns (params, rmse_iv)."""
    w_atm = float(np.interp(0.0, k, w)) if k.min() < 0 < k.max() else w.mean()
    slope = np.polyfit(k, w, 1)[0] if len(k) > 2 else 0.0

    x0 = np.array([w_atm * 0.8, max(abs(slope) * 2, 0.01), -0.3, 0.0, 0.1])

    def obj(x):
        a, b, rho, m, sig = x
        if b < 0 or sig < 1e-6 or abs(rho) >= 1:
            return 1e6
        if a + b * sig * np.sqrt(1 - rho ** 2) < 0:
            return 1e6
        pred = _svi_w(k, a, b, rho, m, sig)
        return float(np.mean((pred - w) ** 2))

    bounds = [
        (-0.5, 2.0),   # a
        (1e-4, 2.0),   # b
        (-0.999, 0.999),  # rho
        (-1.0, 1.0),   # m
        (1e-4, 1.0),   # sig
    ]

    best_res, best_val = None, np.inf
    for rho0 in [-0.5, -0.2, 0.0]:
        x0[2] = rho0
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "ftol": 1e-14})
        if res.fun < best_val:
            best_val, best_res = res.fun, res

    if best_res is None or best_res.fun > 0.1:
        return None, np.nan

    params = best_res.x
    pred_iv = _svi_iv(k, T, *params)
    rmse = float(np.sqrt(np.mean((pred_iv - np.sqrt(w / T)) ** 2))) * 100
    return params, rmse


# ── Public API ────────────────────────────────────────────────────────────────

def fit_svi_surface(
    raw: pd.DataFrame,
    k_grid: np.ndarray,
    T_grid: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit SVI per expiry; interpolate params linearly across T; evaluate on grid.

    Parameters
    ----------
    raw    : DataFrame with columns [expiry, T, k, iv]
    k_grid : 1-D log-moneyness grid for output
    T_grid : 1-D expiry grid in years for output

    Returns
    -------
    surf_df   : DataFrame(index=k_grid, columns=T_grid, values=iv) — the fitted surface
    params_df : per-expiry SVI params + RMSE
    raw       : input DataFrame (unchanged, for scatter plots)
    """
    param_rows = []

    for T_exp, grp in raw.groupby("T"):
        k_exp = grp["k"].values
        iv_exp = grp["iv"].values
        w_exp = iv_exp ** 2 * T_exp

        order = np.argsort(k_exp)
        k_exp, w_exp, iv_exp = k_exp[order], w_exp[order], iv_exp[order]

        params, rmse = _fit_one(k_exp, w_exp, T_exp)
        if params is None:
            continue

        param_rows.append({
            "T": T_exp,
            "a": params[0], "b": params[1], "rho": params[2],
            "m": params[3], "sig": params[4],
            "rmse_iv_pct": rmse,
        })

    if not param_rows:
        return pd.DataFrame(), pd.DataFrame(), raw

    params_df = pd.DataFrame(param_rows).sort_values("T").reset_index(drop=True)

    # Evaluate on (k_grid, T_grid) by linearly interpolating SVI params across T
    T_knots = params_df["T"].values
    z = np.full((len(k_grid), len(T_grid)), np.nan)

    for i_T, T_eval in enumerate(T_grid):
        # Interpolate each SVI parameter at T_eval
        a   = float(np.interp(T_eval, T_knots, params_df["a"]))
        b   = float(np.interp(T_eval, T_knots, params_df["b"]))
        rho = float(np.interp(T_eval, T_knots, params_df["rho"]))
        m   = float(np.interp(T_eval, T_knots, params_df["m"]))
        sig = float(np.interp(T_eval, T_knots, params_df["sig"]))

        iv_col = _svi_iv(k_grid, T_eval, a, b, rho, m, sig)
        z[:, i_T] = iv_col

    surf_df = pd.DataFrame(z, index=k_grid, columns=T_grid)
    return surf_df, params_df, raw
