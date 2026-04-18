"""
Pydantic schemas for the /market endpoints.
"""

from __future__ import annotations
from typing import List, Optional

from pydantic import BaseModel, Field


class VolTermStructurePoint(BaseModel):
    tenor_label: str
    days: float
    atm_iv_pct: float          # ATM implied vol in %
    fwd_vol_pct: Optional[float]  # Forward vol in % (None for the first expiry)
    fwd_label: Optional[str]
    total_var: float           # σ² × T — must be non-decreasing for no-arb
    is_calendar_arb: bool


class VolTermStructureResponse(BaseModel):
    currency: str
    spot: float
    fetched_at: str
    n_quotes: int
    term_structure: List[VolTermStructurePoint]
    n_arb_violations: int
