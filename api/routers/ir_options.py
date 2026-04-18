"""
Router: /ir

Endpoints:
  GET  /ir/curve          — build and return a zero-rate curve
  POST /ir/cap-floor      — price a cap or floor
  POST /ir/swaption       — price a swaption

FastAPI concepts demonstrated here:
  - GET with query parameters
  - POST with discriminated union body (fred vs manual curve)
  - Async function (async def) — non-blocking for I/O like the FRED HTTP call
  - Background-friendly: the FRED fetch is a network call, so async is the right choice
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas.ir_options import (
    CapFloorRequest, CapFloorResponse, CapletDetail, SensitivityPoint,
    SwaptionRequest, SwaptionResponse,
    CurveResponse, ZeroCurvePoint,
    FredCurveSource, ManualCurveSource,
)
from api.services.ir_pricer import price_cap_floor, price_swaption
from market_data.curves.rate_curve import RateCurve
from market_data.providers.fred import fetch_usd_curve

router = APIRouter(prefix="/ir", tags=["Interest Rate Options"])

# ─── Helpers ─────────────────────────────────────────────────────────────────

_TENOR_LABELS = {
    1/12: "1M", 2/12: "2M", 3/12: "3M", 6/12: "6M",
    1.0: "1Y", 2.0: "2Y", 3.0: "3Y", 5.0: "5Y",
    7.0: "7Y", 10.0: "10Y", 15.0: "15Y", 20.0: "20Y", 30.0: "30Y",
}


async def _build_curve(source: FredCurveSource | ManualCurveSource) -> RateCurve:
    """Fetch or assemble par yields and bootstrap the zero curve."""
    if source.type == "fred":
        try:
            # run_in_executor would make this truly async; fine as-is for now
            par_yields = fetch_usd_curve(api_key=source.api_key)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"FRED API error: {exc}")
    else:
        par_yields = {pt.tenor: pt.rate for pt in source.points}

    return RateCurve(par_yields)


# ─── GET /ir/curve ────────────────────────────────────────────────────────────

@router.post("/curve", response_model=CurveResponse)
async def get_curve(source: FredCurveSource | ManualCurveSource) -> CurveResponse:
    """
    Bootstrap a zero-rate curve from par yields.
    Accepts either live FRED data or manually supplied points.

    Note the `async def` — FastAPI runs this in an async event loop,
    which keeps the server responsive while waiting for the FRED HTTP call.
    """
    curve = await _build_curve(source)
    zdf   = curve.zero_curve_df()

    points = [
        ZeroCurvePoint(
            tenor=row["Tenor"],
            tenor_label=_TENOR_LABELS.get(row["Tenor"], f"{row['Tenor']}Y"),
            zero_rate_pct=round(row["Zero Rate (%)"], 4),
            discount_factor=round(
                float(np.exp(-row["Zero Rate (%)"] / 100 * row["Tenor"])), 6
            ),
        )
        for _, row in zdf.iterrows()
    ]
    return CurveResponse(points=points)


# ─── POST /ir/cap-floor ───────────────────────────────────────────────────────

@router.post("/cap-floor", response_model=CapFloorResponse)
async def price_cap_floor_endpoint(req: CapFloorRequest) -> CapFloorResponse:
    """Price a cap or floor and return caplet breakdown + sensitivities."""
    curve = await _build_curve(req.curve)

    price, details = price_cap_floor(
        curve=curve,
        K=req.strike,
        maturity=req.maturity,
        freq=req.freq,
        sigma=req.sigma,
        notional=req.notional,
        vol_type=req.vol_type,
        instrument_type=req.instrument_type,
    )
    price_bps = price / req.notional * 10_000.0

    # Strike sensitivity (80 points)
    k_range = np.linspace(max(req.strike * 0.5, 0.001), req.strike * 1.5, 80)
    sens_strike = [
        SensitivityPoint(
            x=round(k * 100, 4),
            price=round(price_cap_floor(curve, k, req.maturity, req.freq, req.sigma,
                                        req.notional, req.vol_type, req.instrument_type)[0], 2)
        )
        for k in k_range
    ]

    # Vol sensitivity (60 points)
    if req.vol_type == "normal":
        vol_range = np.linspace(5, 250, 60) / 10_000.0
        vol_axis  = vol_range * 10_000
    else:
        vol_range = np.linspace(0.05, 1.0, 60)
        vol_axis  = vol_range * 100

    sens_vol = [
        SensitivityPoint(
            x=round(float(vax), 4),
            price=round(price_cap_floor(curve, req.strike, req.maturity, req.freq, v,
                                        req.notional, req.vol_type, req.instrument_type)[0], 2)
        )
        for v, vax in zip(vol_range, vol_axis)
    ]

    return CapFloorResponse(
        price=round(price, 2),
        price_bps=round(price_bps, 4),
        strike_pct=round(req.strike * 100, 4),
        caplet_details=[CapletDetail(**d) for d in details],
        sensitivity_strike=sens_strike,
        sensitivity_vol=sens_vol,
    )


# ─── POST /ir/swaption ───────────────────────────────────────────────────────

@router.post("/swaption", response_model=SwaptionResponse)
async def price_swaption_endpoint(req: SwaptionRequest) -> SwaptionResponse:
    """Price a payer or receiver swaption and return key metrics + sensitivities."""
    curve = await _build_curve(req.curve)

    T_end = req.expiry + req.swap_tenor
    price, S0, annuity = price_swaption(
        curve=curve,
        T_expiry=req.expiry,
        T_end=T_end,
        K=req.strike,
        sigma=req.sigma,
        notional=req.notional,
        vol_type=req.vol_type,
        swaption_type=req.swaption_type,
        freq=req.freq,
    )
    price_bps    = price / req.notional * 10_000.0
    moneyness    = (S0 - req.strike) * 10_000 if req.swaption_type == "payer" else (req.strike - S0) * 10_000
    if moneyness > 0.5:
        moneyness_label = "ITM"
    elif abs(moneyness) <= 0.5:
        moneyness_label = "ATM"
    else:
        moneyness_label = "OTM"

    # Strike sensitivity
    k_range = np.linspace(max(S0 * 0.5, 0.0001), S0 * 1.5, 80)
    sens_strike = [
        SensitivityPoint(
            x=round(k * 100, 4),
            price=round(price_swaption(curve, req.expiry, T_end, k, req.sigma,
                                       req.notional, req.vol_type, req.swaption_type, req.freq)[0], 2)
        )
        for k in k_range
    ]

    # Vol sensitivity
    if req.vol_type == "normal":
        vol_range = np.linspace(5, 200, 60) / 10_000.0
        vol_axis  = vol_range * 10_000
    else:
        vol_range = np.linspace(0.03, 0.60, 60)
        vol_axis  = vol_range * 100

    sens_vol = [
        SensitivityPoint(
            x=round(float(vax), 4),
            price=round(price_swaption(curve, req.expiry, T_end, req.strike, v,
                                       req.notional, req.vol_type, req.swaption_type, req.freq)[0], 2)
        )
        for v, vax in zip(vol_range, vol_axis)
    ]

    return SwaptionResponse(
        price=round(price, 2),
        price_bps=round(price_bps, 4),
        par_swap_rate_pct=round(S0 * 100, 4),
        annuity=round(annuity, 6),
        moneyness_bps=round(moneyness, 2),
        moneyness_label=moneyness_label,
        sensitivity_strike=sens_strike,
        sensitivity_vol=sens_vol,
    )
