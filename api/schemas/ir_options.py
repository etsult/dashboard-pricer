"""
Pydantic schemas for the /ir (Interest Rate Options) endpoints.

Concepts introduced here:
  - Discriminated unions: CurveSource can be "fred" or "manual"
  - Nested models: CurveSource is embedded inside request bodies
  - Response models with optional fields (fwd_vol may be None for the first expiry)
"""

from __future__ import annotations
from typing import Literal, List, Optional

from pydantic import BaseModel, Field


# ─── Curve source (shared across cap/floor and swaption) ──────────────────────

class FredCurveSource(BaseModel):
    type: Literal["fred"] = "fred"
    api_key: str = Field(..., description="FRED API key (free at fred.stlouisfed.org)")


class ManualCurvePoint(BaseModel):
    tenor: float = Field(..., gt=0, description="Tenor in years (e.g. 0.25 for 3M)")
    rate: float = Field(..., description="Par yield as decimal (e.g. 0.042 for 4.2%)")


class ManualCurveSource(BaseModel):
    type: Literal["manual"] = "manual"
    points: List[ManualCurvePoint] = Field(..., min_length=2)


# ─── Curve response ───────────────────────────────────────────────────────────

class ZeroCurvePoint(BaseModel):
    tenor: float
    tenor_label: str
    zero_rate_pct: float
    discount_factor: float


class CurveResponse(BaseModel):
    """Response from GET /ir/curve."""
    points: List[ZeroCurvePoint]


# ─── Cap / Floor ──────────────────────────────────────────────────────────────

class CapFloorRequest(BaseModel):
    curve: FredCurveSource | ManualCurveSource = Field(
        ..., discriminator="type",
        description="Yield curve source: 'fred' (live) or 'manual' (user-supplied points)"
    )
    instrument_type: Literal["cap", "floor"] = "cap"
    notional: float = Field(10_000_000, gt=0)
    maturity: float = Field(5.0, gt=0, le=30.0, description="Maturity in years")
    freq: float = Field(0.25, description="Reset frequency in years (0.25=quarterly, 0.5=semi)")
    vol_type: Literal["normal", "lognormal"] = "normal"
    sigma: float = Field(..., gt=0, description="Vol in decimal (normal: bps/10000, lognormal: %/100)")
    strike: float = Field(..., gt=0, description="Strike rate as decimal (e.g. 0.042)")


class CapletDetail(BaseModel):
    reset_years: float
    pay_years: float
    fwd_rate_pct: float
    discount_factor: float
    pv: float


class SensitivityPoint(BaseModel):
    x: float  # strike % or vol bps/%
    price: float


class CapFloorResponse(BaseModel):
    price: float
    price_bps: float
    strike_pct: float
    caplet_details: List[CapletDetail]
    sensitivity_strike: List[SensitivityPoint]
    sensitivity_vol: List[SensitivityPoint]


# ─── Swaption ─────────────────────────────────────────────────────────────────

class SwaptionRequest(BaseModel):
    curve: FredCurveSource | ManualCurveSource = Field(..., discriminator="type")
    swaption_type: Literal["payer", "receiver"] = "payer"
    notional: float = Field(10_000_000, gt=0)
    expiry: float = Field(..., gt=0, description="Option expiry in years")
    swap_tenor: float = Field(..., gt=0, description="Underlying swap tenor in years")
    freq: float = Field(0.5, description="Fixed leg payment frequency in years")
    vol_type: Literal["normal", "lognormal"] = "normal"
    sigma: float = Field(..., gt=0)
    strike: float = Field(..., gt=0, description="Fixed rate as decimal")


class SwaptionResponse(BaseModel):
    price: float
    price_bps: float
    par_swap_rate_pct: float
    annuity: float
    moneyness_bps: float
    moneyness_label: str  # "ITM", "ATM", "OTM"
    sensitivity_strike: List[SensitivityPoint]
    sensitivity_vol: List[SensitivityPoint]
