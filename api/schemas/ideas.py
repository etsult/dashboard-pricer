from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ─── Trade Idea ───────────────────────────────────────────────────────────────

class IdeaCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10, description="The hypothesis in plain English")
    asset_class: Literal["crypto", "eqd", "ir", "fx"] = "crypto"
    underlying: str = Field(..., description="e.g. BTC, ETH, SPX, SOFR")
    strategy_type: str = Field(..., description="e.g. short_straddle, risk_reversal, cap_floor")
    direction: Literal["long", "short", "neutral"] = "neutral"
    conviction: int = Field(3, ge=1, le=5, description="1=speculative, 5=high conviction")
    parameters: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class IdeaUpdate(BaseModel):
    description: Optional[str] = None
    status: Optional[Literal["idea", "backtested", "paper", "live", "archived"]] = None
    conviction: Optional[int] = Field(None, ge=1, le=5)
    parameters: Optional[dict[str, Any]] = None
    tags: Optional[list[str]] = None
    direction: Optional[Literal["long", "short", "neutral"]] = None


class BacktestSummary(BaseModel):
    id: int
    run_at: datetime
    sharpe: Optional[float]
    sortino: Optional[float]
    total_pnl: Optional[float]
    max_drawdown: Optional[float]
    win_rate: Optional[float]
    annotation: Optional[str]


class NoteOut(BaseModel):
    id: int
    created_at: datetime
    body: str


class IdeaOut(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime
    name: str
    description: str
    asset_class: str
    underlying: str
    strategy_type: str
    direction: str
    status: str
    conviction: int
    parameters: dict[str, Any]
    tags: list[str]
    backtest_results: list[BacktestSummary] = []
    notes: list[NoteOut] = []

    model_config = {"from_attributes": True}  # allows building from ORM objects


# ─── Notes ────────────────────────────────────────────────────────────────────

class NoteCreate(BaseModel):
    body: str = Field(..., min_length=5)


# ─── Portfolio ────────────────────────────────────────────────────────────────

class PortfolioCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=200)
    description: Optional[str] = None
    construction_method: Literal["equal_weight", "inv_vol", "mean_variance"] = "equal_weight"


class PortfolioAddIdea(BaseModel):
    idea_id: int
    notional_usd: Optional[float] = Field(None, gt=0)


class PortfolioIdeaOut(BaseModel):
    idea_id: int
    idea_name: str
    idea_status: str
    notional_usd: Optional[float]
    weight: Optional[float]
    best_sharpe: Optional[float]

    model_config = {"from_attributes": True}


class PortfolioOut(BaseModel):
    id: int
    name: str
    description: Optional[str]
    construction_method: str
    created_at: datetime
    ideas: list[PortfolioIdeaOut] = []

    model_config = {"from_attributes": True}
