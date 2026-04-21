"""
Router: /research

Endpoints:
  POST /research/backtest/dh-straddle    — delta-hedged straddle backtest with costs
  POST /research/backtest/lump-sum       — buy and hold
  POST /research/backtest/dca            — dollar cost averaging
  POST /research/backtest/compare        — lump sum vs DCA side by side
"""

from __future__ import annotations

import asyncio
from functools import partial

from fastapi import APIRouter, HTTPException

from api.schemas.research import (
    DHStraddleRequest, DHStraddleResponse,
    LumpSumRequest, DCARequest, CompareRequest,
    EquityBacktestResponse, CompareResponse,
    AllWeatherRequest, AllWeatherResponse,
    TimingAnalysisRequest,
)
from research.backtest import run_dh_straddle
from research.costs import CostModel
from research.strategies.equity import run_lump_sum, run_dca, run_all_weather, run_timing_analysis

router = APIRouter(prefix="/research", tags=["Research"])


@router.post("/backtest/dh-straddle", response_model=DHStraddleResponse)
async def backtest_dh_straddle(req: DHStraddleRequest) -> DHStraddleResponse:
    """
    Run a delta-hedged short ATM straddle backtest.

    Returns gross P&L (no costs) and net P&L (after spread, commission,
    slippage, and funding costs) side by side — so you can see exactly
    how much friction eats into the theoretical edge.
    """
    costs = CostModel(
        spread_pct=req.costs.spread_pct,
        commission_pct=req.costs.commission_pct,
        slippage_pct=req.costs.slippage_pct,
        funding_rate_daily=req.costs.funding_rate_daily,
    )

    # run_in_executor: offloads the blocking work to a thread pool
    # so the FastAPI event loop stays free for other requests
    loop = asyncio.get_event_loop()
    fn   = partial(
        run_dh_straddle,
        currency=req.currency,
        history_days=req.history_days,
        T_days=req.T_days,
        rebalance_freq=req.rebalance_freq,
        notional_usd=req.notional_usd,
        costs=costs,
    )

    try:
        result = await loop.run_in_executor(None, fn)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Backtest error: {exc}")

    return DHStraddleResponse(**result)


# ─── Lump Sum ─────────────────────────────────────────────────────────────────

@router.post("/backtest/lump-sum", response_model=EquityBacktestResponse)
async def backtest_lump_sum(req: LumpSumRequest) -> EquityBacktestResponse:
    """
    Buy and hold: invest everything on day one, track until end date.
    Benchmark for comparing against DCA or active strategies.
    """
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            partial(run_lump_sum,
                    ticker=req.ticker,
                    start=req.start,
                    end=req.end,
                    amount=req.amount,
                    transaction_cost_pct=req.transaction_cost_pct),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Data error: {exc}")

    return EquityBacktestResponse(**result)


# ─── DCA ──────────────────────────────────────────────────────────────────────

@router.post("/backtest/dca", response_model=EquityBacktestResponse)
async def backtest_dca(req: DCARequest) -> EquityBacktestResponse:
    """
    Dollar Cost Averaging: invest a fixed amount at each period regardless of price.
    Reduces timing risk at the cost of potentially lower returns in bull markets.
    """
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            partial(run_dca,
                    ticker=req.ticker,
                    start=req.start,
                    end=req.end,
                    periodic_amount=req.periodic_amount,
                    frequency=req.frequency,
                    transaction_cost_pct=req.transaction_cost_pct),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Data error: {exc}")

    return EquityBacktestResponse(**result)


# ─── Compare ──────────────────────────────────────────────────────────────────

@router.post("/backtest/compare", response_model=CompareResponse)
async def compare_lump_sum_vs_dca(req: CompareRequest) -> CompareResponse:
    """
    Run lump sum and DCA on the same ticker and period, return both side by side.

    The DCA periodic amount is computed so that the total invested equals
    `total_amount` spread evenly across the number of periods.
    Both strategies receive the same total capital — a fair comparison.
    """
    from research.strategies.equity import _FREQ_DAYS
    from datetime import timedelta

    # Compute periodic_amount so total invested ≈ total_amount
    n_periods = max(1, (req.end - req.start).days // _FREQ_DAYS[req.dca_frequency])
    periodic_amount = req.total_amount / n_periods

    loop = asyncio.get_event_loop()
    try:
        ls_result, dca_result = await asyncio.gather(
            loop.run_in_executor(None, partial(
                run_lump_sum,
                ticker=req.ticker, start=req.start, end=req.end,
                amount=req.total_amount,
                transaction_cost_pct=req.transaction_cost_pct,
            )),
            loop.run_in_executor(None, partial(
                run_dca,
                ticker=req.ticker, start=req.start, end=req.end,
                periodic_amount=periodic_amount,
                frequency=req.dca_frequency,
                transaction_cost_pct=req.transaction_cost_pct,
            )),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Data error: {exc}")

    return CompareResponse(
        lump_sum=EquityBacktestResponse(**ls_result),
        dca=EquityBacktestResponse(**dca_result),
    )


# ─── Timing Risk ──────────────────────────────────────────────────────────────

@router.post("/backtest/timing-analysis")
async def backtest_timing_analysis(req: TimingAnalysisRequest) -> dict:
    """
    For every possible lump-sum entry date, compute final return, max drawdown,
    and time to recover. Identifies worst / best entry timing and shows the
    drawdown path from the worst entry.
    """
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            partial(
                run_timing_analysis,
                ticker=req.ticker, start=req.start, end=req.end,
                total_amount=req.total_amount,
                transaction_cost_pct=req.transaction_cost_pct,
                sample_every_n_days=req.sample_every_n_days,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Analysis error: {exc}")
    return result


# ─── All Weather ──────────────────────────────────────────────────────────────

@router.post("/backtest/all-weather", response_model=AllWeatherResponse)
async def backtest_all_weather(req: AllWeatherRequest) -> AllWeatherResponse:
    """
    Ray Dalio's All Weather Portfolio with configurable rebalancing frequency.

    Default allocation: SPY 30% · TLT 40% · IEF 15% · GLD 7.5% · GSG 7.5%
    Minimum start date: 2007-01-01 (when all ETF proxies have data).

    Comparing rebalancing frequencies lets you quantify the trade-off between
    tighter tracking (frequent rebal) and lower transaction costs (infrequent rebal).
    """
    if req.weights:
        total = sum(req.weights.values())
        if abs(total - 1.0) > 0.01:
            raise HTTPException(status_code=422, detail=f"Weights must sum to 1.0, got {total:.4f}")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            partial(
                run_all_weather,
                start=req.start,
                end=req.end,
                initial_amount=req.initial_amount,
                rebalance_frequency=req.rebalance_frequency,
                transaction_cost_pct=req.transaction_cost_pct,
                weights=req.weights,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Backtest error: {exc}")

    return AllWeatherResponse(**result)
