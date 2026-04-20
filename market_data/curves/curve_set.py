"""
CurveSet: multi-curve market data container for IRD pricing.

Post-2008 multi-curve framework separates:
  OIS curve        — discounts all cash flows (SOFR, €STR, SONIA, …)
  Projection curve — projects index forward rates (one per floating index)

Backward compatibility
──────────────────────
CurveSet.from_single_curve(curve) wraps a single RateCurve using the flat
basis_bps entries in INDEX_CATALOG, replicating the old single-curve
approximation exactly. All existing callers (engines, tests, pages) that pass
a RateCurve can continue unchanged — the engines auto-wrap internally.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from market_data.curves.rate_curve import RateCurve


@dataclass
class CurveSet:
    """
    Multi-curve market data set passed to all IRD pricers.

    Parameters
    ----------
    ois         OIS (risk-free) discount curve.
    projections {index_key: RateCurve} for each floating index.
                Absent keys fall back to OIS + flat basis from INDEX_CATALOG.
    """
    ois: RateCurve
    projections: dict[str, RateCurve] = field(default_factory=dict)

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_single_curve(cls, curve: RateCurve) -> "CurveSet":
        """Wrap a single RateCurve: OIS = curve, no projection overrides."""
        return cls(ois=curve, projections={})

    def with_projection(self, index_key: str, curve: RateCurve) -> "CurveSet":
        """Return a new CurveSet with one additional projection curve."""
        return CurveSet(
            ois=self.ois,
            projections={**self.projections, index_key: curve},
        )

    # ── Convenience accessors (scalar, for non-hot-path use) ──────────────────

    def disc_df(self, T: float) -> float:
        """OIS discount factor at T (scalar)."""
        return float(np.exp(-np.interp(T, self.ois._tenors, self.ois._zero_rates) * T))

    def proj_df(self, index_key: str, T: float) -> float:
        """Projection curve discount factor at T for index_key (scalar)."""
        if index_key in self.projections:
            c = self.projections[index_key]
            z = float(np.interp(T, c._tenors, c._zero_rates))
            return float(np.exp(-z * T))
        from pricer.ir.indexes import INDEX_CATALOG
        basis = INDEX_CATALOG[index_key]["basis_bps"] / 10_000.0
        z_ois = float(np.interp(T, self.ois._tenors, self.ois._zero_rates))
        return float(np.exp(-(z_ois + basis) * T))
