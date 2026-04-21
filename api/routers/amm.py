"""
AMM (Uniswap v3) LP backtest API
=================================
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/amm", tags=["amm"])


# ── Request / response schemas ────────────────────────────────────────────────

class AMMBacktestRequest(BaseModel):
    symbol:             str   = "SOL/USDT"
    sigma_daily:        float = 0.045
    fee_tier:           float = 0.0005
    range_half_width:   float = 0.10
    initial_capital:    float = 10_000.0
    volume_tvl_ratio:   float = 1.0
    rebalance_strategy: str   = "out_of_range"
    rebalance_cost_bps: float = 10.0
    n_bars:             int   = 43200


class AMMSweepRequest(BaseModel):
    symbol:          str   = "SOL/USDT"
    sigma_daily:     float = 0.045
    fee_tier:        float = 0.0005
    initial_capital: float = 10_000.0
    volume_tvl_ratio: float = 1.0
    rebalance_cost_bps: float = 10.0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/backtest")
def amm_backtest(req: AMMBacktestRequest):
    from research.market_making.data_loader import load_ohlcv_synthetic
    from research.amm.backtest import AMMConfig, run_amm_backtest

    df  = load_ohlcv_synthetic(req.symbol, n_bars=req.n_bars, sigma_daily=req.sigma_daily)
    cfg = AMMConfig(
        symbol=req.symbol,
        fee_tier=req.fee_tier,
        range_half_width=req.range_half_width,
        initial_capital=req.initial_capital,
        volume_tvl_ratio=req.volume_tvl_ratio,
        rebalance_strategy=req.rebalance_strategy,
        rebalance_cost_bps=req.rebalance_cost_bps,
    )
    r = run_amm_backtest(df, cfg)

    step = max(1, len(r.timestamps) // 500)
    return {
        "metrics":        r.metrics,
        "timestamps":     r.timestamps[::step],
        "mid_prices":     r.mid_prices[::step],
        "position_values": r.position_values[::step],
        "hodl_values":    r.hodl_values[::step],
        "fee_cumulative": r.fee_cumulative[::step],
        "il_cumulative":  r.il_cumulative[::step],
        "in_range_flags": r.in_range_flags[::step],
    }


@router.post("/range-sweep")
def range_sweep(req: AMMSweepRequest):
    from research.market_making.data_loader import load_ohlcv_synthetic
    from research.amm.backtest import AMMConfig, run_amm_backtest, range_width_sensitivity

    df  = load_ohlcv_synthetic(req.symbol, n_bars=43200, sigma_daily=req.sigma_daily)
    cfg = AMMConfig(
        symbol=req.symbol,
        fee_tier=req.fee_tier,
        initial_capital=req.initial_capital,
        volume_tvl_ratio=req.volume_tvl_ratio,
        rebalance_cost_bps=req.rebalance_cost_bps,
    )
    rows = range_width_sensitivity(df, cfg)
    return {"rows": rows}


@router.post("/fee-sweep")
def fee_sweep(req: AMMSweepRequest):
    from research.market_making.data_loader import load_ohlcv_synthetic
    from research.amm.backtest import AMMConfig, fee_tier_sensitivity

    df  = load_ohlcv_synthetic(req.symbol, n_bars=43200, sigma_daily=req.sigma_daily)
    cfg = AMMConfig(symbol=req.symbol, initial_capital=req.initial_capital)
    rows = fee_tier_sensitivity(df, cfg)
    return {"rows": rows}


@router.post("/vs-mm")
def vs_mm(req: AMMBacktestRequest):
    """Head-to-head: AMM LP vs AS order book MM on the same asset."""
    import math
    from research.market_making.data_loader import load_ohlcv_synthetic
    from research.amm.backtest import AMMConfig, run_amm_backtest
    from research.amm.metrics import compare_strategies
    from research.market_making.backtest import BacktestConfig, run_backtest

    df = load_ohlcv_synthetic(req.symbol, n_bars=req.n_bars, sigma_daily=req.sigma_daily)

    amm_cfg = AMMConfig(
        symbol=req.symbol,
        fee_tier=req.fee_tier,
        range_half_width=req.range_half_width,
        initial_capital=req.initial_capital,
        volume_tvl_ratio=req.volume_tvl_ratio,
        rebalance_strategy=req.rebalance_strategy,
        rebalance_cost_bps=req.rebalance_cost_bps,
    )
    amm_r = run_amm_backtest(df, amm_cfg)

    # AS MM with comparable capital (max_inventory × price ≈ initial_capital / 2)
    ref_prices = {"SOL/USDT": 150, "ETH/USDT": 3000, "BTC/USDT": 60000,
                  "AVAX/USDT": 35, "BNB/USDT": 400}
    price      = ref_prices.get(req.symbol, 100)
    max_inv    = req.initial_capital / price / 2
    order_sz   = max(max_inv / 50, 0.001)

    mm_cfg = BacktestConfig(
        symbol=req.symbol, model="AS_Gueant",
        gamma=0.10, kappa=3333.0, T_bars=60.0,
        max_inventory=max_inv, order_size=order_sz,
        maker_fee=-0.0001, taker_fee=0.0004,
    )
    mm_r = run_backtest(df, mm_cfg)

    comparison = compare_strategies(amm_r.metrics, mm_r.metrics)
    return {
        "comparison":   comparison,
        "amm_metrics":  amm_r.metrics,
        "mm_metrics":   mm_r.metrics,
    }


@router.get("/optimal-range")
def optimal_range(sigma_daily: float = 0.045, fee_tier: float = 0.0005,
                  volume_tvl_ratio: float = 1.0, bars_per_day: int = 1440):
    """Return the analytically optimal range half-width."""
    import math
    from research.amm.models import optimal_range_width

    sigma_bar   = sigma_daily / math.sqrt(bars_per_day)
    fee_bar     = fee_tier * volume_tvl_ratio / bars_per_day
    w_opt       = optimal_range_width(sigma_bar, fee_bar)
    range_pct   = (math.exp(w_opt) - 1) * 100
    return {
        "optimal_w":      round(w_opt, 4),
        "range_pct_each": round(range_pct, 1),
        "sigma_bar":      round(sigma_bar, 6),
        "fee_bar":        round(fee_bar, 8),
    }
