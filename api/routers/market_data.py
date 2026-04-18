"""
Router: /market

Endpoints:
  GET /market/vol-term-structure?currency=BTC&rate=0.05

FastAPI concepts demonstrated here:
  - Query parameters: declared as function arguments with type hints
  - Optional query params with defaults
  - Literal type for constrained string choices
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException

from api.schemas.market_data import VolTermStructureResponse, VolTermStructurePoint
from market_data.providers.deribit import DeribitProvider
from market_data.curves.vol_term_structure import build_term_structure

router = APIRouter(prefix="/market", tags=["Market Data"])


@router.get("/vol-term-structure", response_model=VolTermStructureResponse)
async def get_vol_term_structure(
    currency: Literal["BTC", "ETH", "SOL"] = "BTC",
    rate: float = 0.05,
) -> VolTermStructureResponse:
    """
    Fetch live crypto option chain from Deribit and return the vol term structure.

    Query parameters are declared as plain function arguments — FastAPI reads them
    from the URL automatically. Example:
      GET /market/vol-term-structure?currency=ETH&rate=0.04

    No API key required (Deribit public endpoints).
    """
    try:
        provider   = DeribitProvider()
        spot       = provider.get_forward(currency)
        quotes     = provider.get_option_chain(currency)
        fetched_at = __import__("datetime").datetime.utcnow().strftime("%H:%M:%S UTC")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Deribit API error: {exc}")

    ts = build_term_structure(quotes, spot=spot, rate=rate)

    if ts.empty:
        raise HTTPException(status_code=204, detail="Not enough data to build term structure.")

    points = [
        VolTermStructurePoint(
            tenor_label=row["tenor_label"],
            days=round(row["days"], 1),
            atm_iv_pct=round(row["atm_iv"] * 100, 2),
            fwd_vol_pct=round(row["fwd_vol"] * 100, 2) if row["fwd_vol"] is not None and str(row["fwd_vol"]) != "nan" else None,
            fwd_label=row.get("fwd_label"),
            total_var=round(row["total_var"], 6),
            is_calendar_arb=bool(row["is_calendar_arb"]),
        )
        for _, row in ts.iterrows()
    ]

    return VolTermStructureResponse(
        currency=currency,
        spot=round(spot, 2),
        fetched_at=fetched_at,
        n_quotes=len(quotes),
        term_structure=points,
        n_arb_violations=int(ts["is_calendar_arb"].sum()),
    )
