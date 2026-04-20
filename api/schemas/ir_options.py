"""Pydantic schemas for the /ir (Interest Rate Options) endpoints."""

from __future__ import annotations
from typing import Literal, List, Optional

from pydantic import BaseModel, Field


# ─── Curve sources ────────────────────────────────────────────────────────────

class FredCurveSource(BaseModel):
    type: Literal["fred"] = "fred"
    api_key: str = Field(..., description="FRED API key (free at fred.stlouisfed.org)")


class ManualCurvePoint(BaseModel):
    tenor: float = Field(..., gt=0, description="Tenor in years")
    rate: float = Field(..., description="Par yield as decimal (0.042 = 4.2%)")


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
    points: List[ZeroCurvePoint]


# ─── Sensitivity ──────────────────────────────────────────────────────────────

class SensitivityPoint(BaseModel):
    x: float
    price: float


# ─── Cap / Floor ──────────────────────────────────────────────────────────────

class CapFloorRequest(BaseModel):
    curve: FredCurveSource | ManualCurveSource = Field(..., discriminator="type")
    instrument_type: Literal["cap", "floor"] = "cap"
    notional: float = Field(10_000_000, gt=0)
    maturity: float = Field(5.0, gt=0, le=30.0, description="Maturity in years")
    freq: float = Field(0.25, description="Reset frequency in years")
    vol_type: Literal["normal", "lognormal"] = "normal"
    sigma: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    # Exotic params
    pricer_model: Literal["fast", "quantlib", "nn"] = "fast"
    start_shift_y: float = Field(0.0, ge=0, description="Forward start offset in years")
    day_count: Literal["ACT/360", "ACT/365", "30/360"] = "ACT/360"
    settlement_delay_y: float = Field(0.0, ge=0, description="Settlement delay in years")
    index_key: Optional[str] = None


class CapletDetail(BaseModel):
    reset_years: float
    pay_years: float
    fwd_rate_pct: float
    discount_factor: float
    pv: float


class CapFloorResponse(BaseModel):
    price: float
    price_bps: float
    strike_pct: float
    caplet_details: List[CapletDetail]
    sensitivity_strike: List[SensitivityPoint]
    sensitivity_vol: List[SensitivityPoint]
    pricer_model: str = "fast"


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
    # Exotic params
    pricer_model: Literal["fast", "quantlib", "nn"] = "fast"
    start_shift_y: float = Field(0.0, ge=0, description="Forward start offset in years (0 = spot)")
    day_count: Literal["ACT/360", "ACT/365", "30/360"] = "ACT/360"
    settlement_delay_y: float = Field(0.0, ge=0, description="Settlement delay in years")
    index_key: Optional[str] = None


class SwaptionResponse(BaseModel):
    price: float
    price_bps: float
    par_swap_rate_pct: float
    annuity: float
    moneyness_bps: float
    moneyness_label: str
    sensitivity_strike: List[SensitivityPoint]
    sensitivity_vol: List[SensitivityPoint]
    pricer_model: str = "fast"


# ─── Vanilla IRS ──────────────────────────────────────────────────────────────

class IRSRequest(BaseModel):
    curve: FredCurveSource | ManualCurveSource = Field(..., discriminator="type")
    irs_type: Literal["payer", "receiver"] = "payer"  # payer = pay fixed, receive float
    notional: float = Field(10_000_000, gt=0)
    start_shift_y: float = Field(0.0, ge=0, description="Forward start in years (0 = spot)")
    tenor_y: float = Field(..., gt=0, description="Swap tenor in years")
    fixed_rate: float = Field(..., gt=0, description="Fixed leg rate as decimal")
    fixed_freq: float = Field(0.5, description="Fixed leg payment frequency in years")
    float_freq: float = Field(0.25, description="Float leg reset frequency in years")
    day_count: Literal["ACT/360", "ACT/365", "30/360"] = "ACT/360"
    index_key: Optional[str] = None
    pricer_model: Literal["fast", "quantlib"] = "fast"
    # XCcy fields
    xccy: bool = False
    domestic_ccy: str = "USD"
    foreign_ccy: str = "EUR"
    fx_rate: float = Field(1.0, description="Units of domestic per 1 foreign")
    basis_spread_bps: float = Field(0.0, description="Cross-currency basis spread in bps")


class IRSLegDetail(BaseModel):
    pay_years: float
    fixed_cashflow: float
    float_cashflow: float
    discount_factor: float
    net_pv: float


class IRSResponse(BaseModel):
    price: float
    price_bps: float
    par_swap_rate_pct: float
    annuity: float
    fixed_leg_pv: float
    float_leg_pv: float
    dv01: float
    leg_details: List[IRSLegDetail]
    sensitivity_rate: List[SensitivityPoint]
    pricer_model: str = "fast"


# ─── Benchmark ────────────────────────────────────────────────────────────────

class BenchmarkEngineResult(BaseModel):
    engine: str                     # "fast_bachelier" | "quantlib_single" | "quantlib_multi" | "ore"
    curve_mode: str                 # "single-curve (OIS)" | "multi-curve (OIS+basis)"
    available: bool
    price: Optional[float] = None
    price_bps: Optional[float] = None
    atm_rate_pct: Optional[float] = None
    diff_vs_fast_bps: Optional[float] = None   # price diff in bps vs fast_bachelier
    note: Optional[str] = None


class BenchmarkDiagnostics(BaseModel):
    index_key: str
    basis_spread_bps: float
    formula_diff_bps: float         # fast vs QL same-curve (should be ~0)
    multicurve_effect_bps: float    # QL single vs QL multi (= basis impact on price)


class BenchmarkRequest(BaseModel):
    instrument_class: Literal["swaption", "cap_floor", "irs"]
    swaption: Optional[SwaptionRequest] = None
    cap_floor: Optional[CapFloorRequest] = None
    irs: Optional[IRSRequest] = None


class BenchmarkResponse(BaseModel):
    instrument_class: str
    notional: float
    results: List[BenchmarkEngineResult]
    diagnostics: BenchmarkDiagnostics
