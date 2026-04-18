from __future__ import annotations
from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel, Field


class CostModelInput(BaseModel):
    spread_pct: float = Field(0.05 / 100, ge=0, description="Bid-ask spread as decimal (0.0005 = 5bps)")
    commission_pct: float = Field(0.03 / 100, ge=0, description="Exchange commission per leg")
    slippage_pct: float = Field(0.01 / 100, ge=0, description="Market impact")
    funding_rate_daily: float = Field(0.0003, ge=0, description="Daily funding on delta hedge")


class DHStraddleRequest(BaseModel):
    currency: Literal["BTC", "ETH"] = "BTC"
    history_days: int = Field(365, ge=90, le=1000)
    T_days: int = Field(30, ge=7, le=90, description="Straddle maturity in calendar days")
    rebalance_freq: int = Field(1, ge=1, le=30, description="Delta rebalance every N days")
    notional_usd: float = Field(100_000, gt=0)
    costs: CostModelInput = Field(default_factory=CostModelInput)


class PerformanceOut(BaseModel):
    total_pnl: float
    avg_daily_pnl: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_drawdown: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    win_rate: Optional[float]
    n_trades: Optional[int]
    avg_win: Optional[float]
    avg_loss: Optional[float]
    profit_factor: Optional[float]


class CostSummary(BaseModel):
    total_costs: float
    entry_exit_costs: float
    hedge_rebal_costs: float
    funding_costs: float
    cost_as_pct_gross: Optional[float]


class DHStraddleResponse(BaseModel):
    gross_performance: PerformanceOut
    net_performance: PerformanceOut
    cost_summary: CostSummary
    daily: list[dict]
    trades: list[dict]


# ─── Equity strategies ────────────────────────────────────────────────────────

class LumpSumRequest(BaseModel):
    ticker: str = Field("SPY", description="ETF or stock ticker (e.g. SPY, QQQ, IVV)")
    start: date = Field(..., description="Start date YYYY-MM-DD")
    end: date = Field(..., description="End date YYYY-MM-DD")
    amount: float = Field(..., gt=0, description="Total investment in USD")
    transaction_cost_pct: float = Field(0.0, ge=0, le=0.02, description="Round-trip cost as decimal")


class DCARequest(BaseModel):
    ticker: str = Field("SPY")
    start: date = Field(...)
    end: date = Field(...)
    periodic_amount: float = Field(..., gt=0, description="USD invested per period")
    frequency: Literal["weekly", "biweekly", "monthly", "quarterly"] = "monthly"
    transaction_cost_pct: float = Field(0.0, ge=0, le=0.02)


class EquityPerformance(BaseModel):
    final_value: float
    total_invested: float
    total_return_pct: float
    cagr_pct: float
    volatility_ann_pct: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    best_year_pct: Optional[float]
    worst_year_pct: Optional[float]
    n_trades: int
    n_years: float
    avg_cost_basis: Optional[float] = None  # DCA only


class EquityBacktestResponse(BaseModel):
    performance: EquityPerformance
    daily: list[dict]
    trades: list[dict]


# ─── Comparison ───────────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    """Run lump sum and DCA on the same ticker/period and return both side by side."""
    ticker: str = Field("SPY")
    start: date = Field(...)
    end: date = Field(...)
    total_amount: float = Field(..., gt=0, description="Total USD available to invest")
    dca_frequency: Literal["weekly", "biweekly", "monthly", "quarterly"] = "monthly"
    transaction_cost_pct: float = Field(0.0, ge=0, le=0.02)


class CompareResponse(BaseModel):
    lump_sum: EquityBacktestResponse
    dca: EquityBacktestResponse


# ─── All Weather ──────────────────────────────────────────────────────────────

class AllWeatherRequest(BaseModel):
    start: date = Field(...)
    end: date = Field(...)
    initial_amount: float = Field(..., gt=0)
    rebalance_frequency: Literal["daily", "monthly", "quarterly", "annually"] = "quarterly"
    transaction_cost_pct: float = Field(0.0, ge=0, le=0.02)
    weights: dict[str, float] | None = Field(
        None,
        description="Override default weights. Keys = tickers, values must sum to 1.0. "
                    "Default: SPY 30%, TLT 40%, IEF 15%, GLD 7.5%, GSG 7.5%"
    )


class AllWeatherResponse(BaseModel):
    performance: EquityPerformance
    daily: list[dict]
    trades: list[dict]
    weights_used: dict[str, float]
