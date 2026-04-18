"""
Synthetic IR normal volatility cube.

Generates ATM normal vol surfaces for:
  • Swaptions:   2-D grid (expiry × swap tenor), one surface per CCY.
  • Caps/floors: 1-D grid (cap maturity), one curve per index.
  • Smile:       ZABR (nu, rho) per (index, expiry, product type).

Index differentiation (realistic 2025):
  USD swaptions (SOFR-based)   ~ 80–120 bps (hump at 2–3Y expiry)
  EUR swaptions (EURIBOR/€STR) ~ 55–85 bps
  GBP swaptions (SONIA-based)  ~ 70–100 bps
  Cap/floor vols > swaption vol by a convexity premium that depends on
  reset frequency: TERM_SOFR_1M caps are ~15 % above SOFR swaptions;
  EUR_3M caps ~8 % above EUR swaptions.

All public vol values are in decimal (0.0100 = 100 bps normal vol).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .indexes import INDEX_CATALOG

# ── Grid axes ──────────────────────────────────────────────────────────────────

EXPIRY_GRID = np.array([1/12, 3/12, 6/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0])
TENOR_GRID  = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
CAPMAT_GRID = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0])

# ── CCY base levels (ATM normal vol in bps at the "belly" of the surface) ──────

_SW_BASE_BPS: dict[str, float] = {
    "USD": 100.0,
    "EUR":  65.0,
    "GBP":  88.0,
}

# Convexity premium of cap/floor vol relative to same-CCY swaption level.
# Higher reset frequency → more caplet optionality → higher vol.
_CF_MULT: dict[str, float] = {
    "TERM_SOFR_1M":  1.18,   # monthly resets, maximum convexity
    "TERM_SOFR_3M":  1.10,   # benchmark USD cap market
    "TERM_SOFR_6M":  1.04,
    "SOFR_3M":       1.06,
    "SOFR":          0.92,   # O/N compounding smooths out vol
    "LIBOR_3M":      1.12,   # legacy — kept slightly elevated
    "EUR_1M":        1.14,
    "EUR_3M":        1.08,   # benchmark EUR cap market
    "EUR_6M":        1.02,
    "ESTR":          0.88,   # OIS benchmark, lowest vol
    "GBP_SONIA":     0.95,
    "GBP_SONIA_3M":  1.05,
}
_CF_MULT_DEFAULT = 1.05


# ── Surface shape functions ────────────────────────────────────────────────────

def _swaption_surface(
    base_bps: float,
    T_peak: float = 2.0,
    hump_amp: float = 0.25,
    tenor_decay: float = 0.015,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a (len(EXPIRY_GRID), len(TENOR_GRID)) ATM normal vol surface in bps.

    Shape:
      - Expiry axis: hump centred at T_peak (short-end suppressed).
      - Tenor axis:  mild exponential decay (longer swap tenor → lower vol).
    """
    # Expiry shape
    T = EXPIRY_GRID
    exp_shape = 1.0 + hump_amp * np.exp(-((T - T_peak) ** 2) / (2 * 1.5 ** 2))
    exp_shape *= 1.0 - 0.15 * np.exp(-T * 3)          # suppress very short end

    # Tenor shape
    ten_shape = np.exp(-tenor_decay * (TENOR_GRID - 1.0))

    surface = base_bps * np.outer(exp_shape, ten_shape)  # [nExp, nTen]

    if noise is not None:
        surface *= 1.0 + noise

    return np.clip(surface, 8.0, None) / 10_000.0       # bps → decimal


def _capfloor_curve(
    base_bps: float,
    mult: float = 1.0,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a (len(CAPMAT_GRID),) ATM normal vol curve in bps for a cap index.

    Shape: humped around 2–3Y maturity, flatter than swaption surface.
    """
    M = CAPMAT_GRID
    hump = 1.0 + 0.18 * np.exp(-((M - 2.5) ** 2) / (2 * 2.0 ** 2))
    curve = base_bps * mult * hump

    if noise is not None:
        curve *= 1.0 + noise

    return np.clip(curve, 8.0, None) / 10_000.0         # bps → decimal


# ── VolCube ────────────────────────────────────────────────────────────────────

class VolCube:
    """
    Synthetic IR normal volatility cube (ATM + ZABR smile params).

    Parameters
    ----------
    seed : int
        Random seed for synthetic noise (different seeds → different market scenarios).

    Attributes
    ----------
    swaption_surfaces : dict[ccy, ndarray]
        Shape (len(EXPIRY_GRID), len(TENOR_GRID)).  Values in decimal.
    capfloor_curves : dict[index_key, ndarray]
        Shape (len(CAPMAT_GRID),).  Values in decimal.

    Examples
    --------
    >>> cube = VolCube(seed=42)
    >>> cube.swaption_atm("USD", 2.0, 10.0)    # → ~0.0095 (95 bps)
    >>> cube.capfloor_atm("TERM_SOFR_3M", 5.0) # → ~0.0105 (105 bps)
    >>> cube.smile_params("EUR_3M", 1.0, is_capfloor=True)
    """

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        nE, nT, nM = len(EXPIRY_GRID), len(TENOR_GRID), len(CAPMAT_GRID)

        # ── Swaption surfaces ─────────────────────────────────────────────
        self.swaption_surfaces: dict[str, np.ndarray] = {}
        self._sw_interp: dict[str, RegularGridInterpolator] = {}

        for ccy, base in _SW_BASE_BPS.items():
            noise = rng.normal(0, 0.018, (nE, nT))
            surf  = _swaption_surface(base, noise=noise)
            self.swaption_surfaces[ccy] = surf
            self._sw_interp[ccy] = RegularGridInterpolator(
                (EXPIRY_GRID, TENOR_GRID), surf,
                method="linear", bounds_error=False, fill_value=None,
            )

        # ── Cap/floor curves (one per index) ─────────────────────────────
        self.capfloor_curves: dict[str, np.ndarray] = {}
        self._cf_x: dict[str, np.ndarray] = {}  # shared x-axis per index

        for idx, meta in INDEX_CATALOG.items():
            ccy  = meta["ccy"]
            base = _SW_BASE_BPS.get(ccy, 80.0)
            mult = _CF_MULT.get(idx, _CF_MULT_DEFAULT)
            noise = rng.normal(0, 0.018, nM)
            self.capfloor_curves[idx] = _capfloor_curve(base, mult=mult, noise=noise)
            self._cf_x[idx] = CAPMAT_GRID.copy()

    # ── Scalar accessors ──────────────────────────────────────────────────────

    def swaption_atm(self, ccy: str, expiry_y: float, tenor_y: float) -> float:
        """ATM normal vol (decimal) for a swaption."""
        interp = self._sw_interp.get(ccy, self._sw_interp["USD"])
        T   = np.clip(expiry_y, EXPIRY_GRID[0], EXPIRY_GRID[-1])
        ten = np.clip(tenor_y,  TENOR_GRID[0],  TENOR_GRID[-1])
        return float(interp([[T, ten]])[0])

    def capfloor_atm(self, index_key: str, mat_y: float) -> float:
        """ATM normal vol (decimal) for a cap/floor of given maturity."""
        fallback = "TERM_SOFR_3M" if "USD" in index_key else "EUR_3M"
        x = self._cf_x.get(index_key, self._cf_x.get(fallback, CAPMAT_GRID))
        y = self.capfloor_curves.get(
            index_key, self.capfloor_curves.get(fallback)
        )
        return float(np.interp(np.clip(mat_y, x[0], x[-1]), x, y))

    def smile_params(
        self,
        index_key: str,
        expiry_y: float,
        tenor_y: float = 5.0,
        is_capfloor: bool = False,
    ) -> tuple[float, float, float]:
        """
        Return (alpha, nu, rho) for ZABR smile.
        alpha = ATM vol for this instrument / index / expiry.
        """
        ccy = INDEX_CATALOG[index_key]["ccy"]
        nu  = 0.20 + 0.35 * np.exp(-0.30 * expiry_y)
        rho = -0.10 - 0.25 * np.exp(-0.45 * expiry_y)
        if is_capfloor:
            rho -= 0.05           # cap/floor: more negative skew
        if ccy == "EUR":
            rho -= 0.03

        if is_capfloor:
            alpha = self.capfloor_atm(index_key, expiry_y)
        else:
            alpha = self.swaption_atm(ccy, expiry_y, tenor_y)

        return alpha, float(nu), float(np.clip(rho, -0.95, 0.95))

    # ── Vectorized batch lookups (used by book generator) ─────────────────────

    def batch_swaption_vol(
        self,
        ccys: np.ndarray,
        expiries: np.ndarray,
        tenors: np.ndarray,
    ) -> np.ndarray:
        """Return ATM vol array for N swaption positions."""
        out = np.zeros(len(ccys))
        for ccy in np.unique(ccys):
            mask = ccys == ccy
            interp = self._sw_interp.get(ccy, self._sw_interp["USD"])
            T   = np.clip(expiries[mask], EXPIRY_GRID[0], EXPIRY_GRID[-1])
            ten = np.clip(tenors[mask],   TENOR_GRID[0],  TENOR_GRID[-1])
            out[mask] = interp(np.stack([T, ten], axis=1))
        return out

    def batch_capfloor_vol(
        self,
        index_keys: np.ndarray,
        maturities: np.ndarray,
    ) -> np.ndarray:
        """Return ATM vol array for N cap/floor positions."""
        out = np.zeros(len(index_keys))
        for idx in np.unique(index_keys):
            mask = index_keys == idx
            x = self._cf_x.get(idx, CAPMAT_GRID)
            y = self.capfloor_curves.get(idx)
            if y is None:
                fallback = "TERM_SOFR_3M" if "USD" in idx else "EUR_3M"
                x, y = self._cf_x[fallback], self.capfloor_curves[fallback]
            m = np.clip(maturities[mask], x[0], x[-1])
            out[mask] = np.interp(m, x, y)
        return out

    # ── DataFrame views (for display / export) ───────────────────────────────

    def swaption_surface_df(self, ccy: str = "USD"):
        """Swaption ATM surface in bps as a labelled DataFrame."""
        import pandas as pd
        surf = self.swaption_surfaces.get(ccy, self.swaption_surfaces["USD"])
        return pd.DataFrame(
            surf * 10_000,
            index=[f"{t*12:.0f}M" if t < 1 else f"{t:.0f}Y" for t in EXPIRY_GRID],
            columns=[f"{t:.0f}Y" for t in TENOR_GRID],
        )

    def capfloor_surface_df(self):
        """All cap/floor curves (bps) in one DataFrame, indexes as columns."""
        import pandas as pd
        data = {idx: self.capfloor_curves[idx] * 10_000
                for idx in sorted(self.capfloor_curves)}
        return pd.DataFrame(
            data,
            index=[f"{m:.0f}Y" for m in CAPMAT_GRID],
        )
