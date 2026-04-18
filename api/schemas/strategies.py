"""
Pydantic schemas for the /strategies endpoint.

FastAPI uses these to:
  1. Validate and parse incoming JSON automatically
  2. Generate the OpenAPI (Swagger) documentation at /docs
  3. Serialize the response back to JSON

Key concepts here:
  - BaseModel: the base class for all schemas
  - Field(...): required field (... = no default)
  - Field(default): optional field with a default value
  - Literal: restricts a string to a fixed set of values
"""

from __future__ import annotations
from typing import Literal, List
from datetime import date

from pydantic import BaseModel, Field


# ─── Sub-schemas ──────────────────────────────────────────────────────────────

class LegInput(BaseModel):
    """A single option leg in a multi-leg strategy."""
    option_type: Literal["call", "put"]
    strike: float = Field(..., gt=0, description="Strike price")
    qty: float = Field(..., description="Signed quantity: positive = long, negative = short")
    sigma: float = Field(..., gt=0, description="Implied volatility as a decimal (e.g. 0.20 for 20%)")
    expiry: date = Field(..., description="Option expiry date (YYYY-MM-DD)")


# ─── Request ──────────────────────────────────────────────────────────────────

class PriceStrategyRequest(BaseModel):
    """
    Full request body for POST /strategies/price.
    Carries market context and the list of legs to price.
    """
    model: Literal["Black-76", "Black-Scholes", "Bachelier"] = Field(
        "Black-76", description="Pricing model to use"
    )
    forward: float = Field(..., gt=0, description="Forward / spot price")
    rate: float = Field(0.05, description="Risk-free rate as decimal")
    dividend_yield: float = Field(0.0, description="Dividend yield (Black-Scholes only)")
    valuation_date: date = Field(..., description="Pricing date (YYYY-MM-DD)")
    legs: List[LegInput] = Field(..., min_length=1, description="At least one leg required")
    forward_range_points: int = Field(
        400, ge=50, le=1000,
        description="Number of points in the forward sweep for charts"
    )


# ─── Response ─────────────────────────────────────────────────────────────────

class Greeks(BaseModel):
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class GreeksVsForward(BaseModel):
    """Arrays of greek values across the forward range (for charting)."""
    forward_range: List[float]
    price: List[float]
    delta: List[float]
    gamma: List[float]
    vega: List[float]
    theta: List[float]
    rho: List[float]


class PriceStrategyResponse(BaseModel):
    """Full response from POST /strategies/price."""
    greeks: Greeks
    payoff: dict = Field(
        description="forward_range, payoff_today, payoff_expiry arrays for the payoff chart"
    )
    greeks_vs_forward: GreeksVsForward
