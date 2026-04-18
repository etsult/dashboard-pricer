"""
Pydantic schemas for book management, fast pricing, and risk endpoints.
"""

from __future__ import annotations

from typing import Literal, List, Optional
from pydantic import BaseModel, Field


# ── Position ──────────────────────────────────────────────────────────────────

class IRPositionSchema(BaseModel):
    instrument:  Literal["payer_swaption", "receiver_swaption", "cap", "floor"]
    index_key:   str
    notional:    float = Field(..., gt=0)
    strike:      float = Field(..., gt=0, description="Fixed rate, decimal")
    expiry_y:    float = Field(..., gt=0)
    tenor_y:     float = Field(..., gt=0)
    sigma_n:     float = Field(..., gt=0, description="Normal vol, decimal")
    direction:   Literal[-1, 1] = 1
    label:       str   = ""

    model_config = {"from_attributes": True}


# ── Book ──────────────────────────────────────────────────────────────────────

class BookSchema(BaseModel):
    book_id:     str
    n_positions: int
    positions:   List[IRPositionSchema]


class GenerateBookRequest(BaseModel):
    n:           int   = Field(10_000, ge=100, le=500_000)
    seed:        int   = 42
    usd_weight:  float = Field(0.60, ge=0.0, le=1.0)
    add_hedges:  bool  = True


# ── Pricing ───────────────────────────────────────────────────────────────────

class CurveSpec(BaseModel):
    """Inline zero-curve: list of (tenor_y, zero_rate_decimal) pairs."""
    points: List[tuple[float, float]] = Field(
        ...,
        description="[(tenor_y, zero_rate)] e.g. [(1.0, 0.042), (5.0, 0.044)]",
    )
    shift_bp: float = Field(0.0, description="Optional parallel shift in bps applied on top")


class PriceBookRequest(BaseModel):
    positions: List[IRPositionSchema]
    curve:     CurveSpec


class PriceRow(BaseModel):
    label:       str
    instrument:  str
    index_key:   str
    ccy:         str
    expiry_y:    float
    tenor_y:     float
    strike_pct:  float
    atm_pct:     float
    sigma_bps:   float
    notional:    float
    direction:   int
    pv:          float


class PriceBookResponse(BaseModel):
    total_pv:  float
    positions: List[PriceRow]


# ── Risk ──────────────────────────────────────────────────────────────────────

class RiskBookRequest(BaseModel):
    positions: List[IRPositionSchema]
    curve:     CurveSpec
    dv01_bp:   float = Field(1.0,  ge=0.1, le=100.0)
    gamma_bp:  float = Field(5.0,  ge=0.1, le=200.0)


class RiskRow(BaseModel):
    label:      str
    instrument: str
    index_key:  str
    ccy:        str
    expiry_y:   float
    tenor_y:    float
    notional:   float
    direction:  int
    pv:         float
    dv01:       float
    gamma_up:   float
    gamma_dn:   float


class AggregateRisk(BaseModel):
    total_pv:       float
    total_dv01:     float
    total_gamma_up: float
    total_gamma_dn: float
    by_index:       dict[str, dict[str, float]]   # index_key → {pv, dv01, ...}
    by_expiry:      dict[str, dict[str, float]]   # bucket   → {pv, dv01, ...}


class RiskBookResponse(BaseModel):
    aggregate:  AggregateRisk
    positions:  List[RiskRow]


# ── Vol Cube ──────────────────────────────────────────────────────────────────

class VolCubeRequest(BaseModel):
    seed: int = 0


class SwaptionSurfaceResponse(BaseModel):
    ccy:         str
    expiry_grid: List[str]
    tenor_grid:  List[str]
    vols_bps:    List[List[float]]   # [expiry, tenor] in bps


class CapFloorCurveResponse(BaseModel):
    index_key:    str
    index_label:  str
    mat_grid:     List[str]
    vols_bps:     List[float]


class SmileRequest(BaseModel):
    index_key:   str
    expiry_y:    float
    tenor_y:     float = 5.0
    is_capfloor: bool  = False
    n_strikes:   int   = Field(100, ge=20, le=500)


class SmilePoint(BaseModel):
    strike_pct: float
    vol_bps:    float


class SmileResponse(BaseModel):
    index_key:   str
    expiry_y:    float
    atm_pct:     float
    alpha_bps:   float
    nu:          float
    rho:         float
    smile:       List[SmilePoint]
