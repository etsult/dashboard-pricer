"""
Router: /strategies

FastAPI concepts demonstrated here:
  - APIRouter: a mini-app you mount onto the main app
  - @router.post: HTTP POST route with automatic JSON body parsing
  - response_model: FastAPI validates and serializes the return value
  - HTTPException: standard way to return 4xx/5xx errors
  - Depends(): dependency injection (not used here yet, but the pattern is ready)
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas.strategies import (
    PriceStrategyRequest,
    PriceStrategyResponse,
    Greeks,
    GreeksVsForward,
)
from app.ui.helpers import build_strategy_from_df
import pandas as pd

router = APIRouter(prefix="/strategies", tags=["Strategies"])


def _request_to_dataframe(req: PriceStrategyRequest) -> pd.DataFrame:
    """Convert the list of LegInput objects into the DataFrame that build_strategy_from_df expects."""
    rows = []
    for leg in req.legs:
        rows.append({
            "Type":   leg.option_type.capitalize(),
            "Strike": leg.strike,
            "Qty":    leg.qty,
            "σ":      leg.sigma,
            "Expiry": datetime.combine(leg.expiry, datetime.min.time()),
        })
    return pd.DataFrame(rows)


@router.post("/price", response_model=PriceStrategyResponse)
def price_strategy(req: PriceStrategyRequest) -> PriceStrategyResponse:
    """
    Price a multi-leg option strategy and return greeks + chart data.

    FastAPI automatically:
    - Parses the JSON body into PriceStrategyRequest
    - Validates all field constraints (gt=0, min_length=1, etc.)
    - Returns 422 Unprocessable Entity if validation fails
    - Serializes the return value using PriceStrategyResponse
    """
    valuation_date = datetime.combine(req.valuation_date, datetime.min.time())

    try:
        legs_df = _request_to_dataframe(req)
        strategy = build_strategy_from_df(
            df=legs_df,
            model=req.model,
            F=req.forward,
            r=req.rate,
            q=req.dividend_yield,
            sigma_default=0.20,
            valuation_date=valuation_date,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Scalar greeks at current forward
    greeks = Greeks(
        price=strategy.price(),
        delta=strategy.delta(),
        gamma=strategy.gamma(),
        vega=strategy.vega(),
        theta=strategy.theta(),
        rho=strategy.rho(),
    )

    # Arrays over a forward sweep (for chart data)
    F_range = np.linspace(0.5 * req.forward, 1.5 * req.forward, req.forward_range_points)
    payoff_today  = strategy.price_vs_forward(F_range)
    payoff_expiry = strategy.payoff_at_expiry_vs_forward(F_range)
    gvf           = strategy.greeks_vs_forward(F_range)

    return PriceStrategyResponse(
        greeks=greeks,
        payoff={
            "forward_range":  F_range.tolist(),
            "payoff_today":   payoff_today.tolist(),
            "payoff_expiry":  payoff_expiry.tolist(),
        },
        greeks_vs_forward=GreeksVsForward(
            forward_range=F_range.tolist(),
            price=gvf["price"].tolist(),
            delta=gvf["delta"].tolist(),
            gamma=gvf["gamma"].tolist(),
            vega=gvf["vega"].tolist(),
            theta=gvf["theta"].tolist(),
            rho=gvf["rho"].tolist(),
        ),
    )
