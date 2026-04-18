"""
Router: /vol-cube

Endpoints
---------
GET  /vol-cube/swaption/{ccy}    ATM swaption surface (expiry × tenor) in bps
GET  /vol-cube/capfloor          All cap/floor ATM curves by index
POST /vol-cube/smile             ZABR smile for (index, expiry, product type)
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas.books import (
    VolCubeRequest,
    SwaptionSurfaceResponse,
    CapFloorCurveResponse,
    SmileRequest, SmileResponse, SmilePoint,
)
from pricer.ir.vol_cube import VolCube, EXPIRY_GRID, TENOR_GRID, CAPMAT_GRID
from pricer.ir.indexes import INDEX_CATALOG
from pricer.ir.zabr import smile_vol_strip

router = APIRouter(prefix="/vol-cube", tags=["Vol Cube"])

_CCY_ATM = {"USD": 0.0450, "EUR": 0.0235, "GBP": 0.0455}


def _get_cube(seed: int) -> VolCube:
    return VolCube(seed=seed)


# ── GET /vol-cube/swaption/{ccy} ──────────────────────────────────────────────

@router.get("/swaption/{ccy}", response_model=SwaptionSurfaceResponse)
def swaption_surface(ccy: str, seed: int = 0) -> SwaptionSurfaceResponse:
    """ATM swaption normal vol surface in bps."""
    ccy = ccy.upper()
    if ccy not in ("USD", "EUR", "GBP"):
        raise HTTPException(400, f"Unknown CCY: {ccy}. Use USD, EUR or GBP.")
    cube = _get_cube(seed)
    surf = cube.swaption_surfaces[ccy] * 10_000   # decimal → bps
    return SwaptionSurfaceResponse(
        ccy=ccy,
        expiry_grid=[f"{t*12:.0f}M" if t < 1 else f"{t:.0f}Y" for t in EXPIRY_GRID],
        tenor_grid= [f"{t:.0f}Y" for t in TENOR_GRID],
        vols_bps=  [[round(float(v), 2) for v in row] for row in surf],
    )


# ── GET /vol-cube/capfloor ────────────────────────────────────────────────────

@router.get("/capfloor", response_model=list[CapFloorCurveResponse])
def capfloor_curves(seed: int = 0) -> list[CapFloorCurveResponse]:
    """ATM cap/floor normal vol curves (bps) for every index."""
    cube = _get_cube(seed)
    out  = []
    mat_labels = [f"{m:.0f}Y" for m in CAPMAT_GRID]
    for idx, meta in INDEX_CATALOG.items():
        curve_bps = cube.capfloor_curves.get(idx)
        if curve_bps is None:
            continue
        out.append(CapFloorCurveResponse(
            index_key=   idx,
            index_label= meta["label"],
            mat_grid=    mat_labels,
            vols_bps=    [round(float(v) * 10_000, 2) for v in curve_bps],
        ))
    return out


# ── POST /vol-cube/smile ──────────────────────────────────────────────────────

@router.post("/smile", response_model=SmileResponse)
def smile_slice(req: SmileRequest) -> SmileResponse:
    """
    Return a ZABR smile across strikes for a given (index, expiry).
    Strikes are spaced ±3σ around the ATM forward.
    """
    if req.index_key not in INDEX_CATALOG:
        raise HTTPException(400, f"Unknown index: {req.index_key}")

    cube = VolCube(seed=0)
    ccy  = INDEX_CATALOG[req.index_key]["ccy"]
    alpha, nu, rho = cube.smile_params(
        req.index_key, req.expiry_y,
        tenor_y=req.tenor_y, is_capfloor=req.is_capfloor,
    )
    F_atm = _CCY_ATM.get(ccy, 0.04)
    sigma_move = alpha * np.sqrt(req.expiry_y)
    K_lo = max(F_atm - 3 * sigma_move, 0.001)
    K_hi = F_atm + 3 * sigma_move
    strikes = np.linspace(K_lo, K_hi, req.n_strikes)
    vols    = smile_vol_strip(F_atm, strikes, req.expiry_y, alpha, nu, rho)

    return SmileResponse(
        index_key=  req.index_key,
        expiry_y=   req.expiry_y,
        atm_pct=    round(F_atm * 100, 4),
        alpha_bps=  round(alpha * 10_000, 2),
        nu=         round(nu, 4),
        rho=        round(rho, 4),
        smile=[
            SmilePoint(strike_pct=round(float(k) * 100, 4),
                       vol_bps=round(float(v) * 10_000, 2))
            for k, v in zip(strikes, vols)
        ],
    )
