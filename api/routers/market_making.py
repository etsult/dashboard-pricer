"""
Router: /market-making

Endpoints:
  POST /market-making/backtest          — run AS model backtest
  POST /market-making/sensitivity       — γ or κ sensitivity sweep
  GET  /market-making/asset-scores      — rank candidate assets
"""

from __future__ import annotations

from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from research.market_making.backtest import BacktestConfig, run_backtest, gamma_sensitivity, kappa_sensitivity
from research.market_making.data_loader import fetch_ohlcv
from research.market_making.models import rank_candidates

router = APIRouter(prefix="/market-making", tags=["Market Making"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    symbol:          str   = Field("SOL/USDT",   description="e.g. SOL/USDT, BTC/USDT")
    exchange:        str   = Field("binance",    description="ccxt exchange id")
    model:           Literal["AS_basic", "AS_Gueant"] = "AS_Gueant"
    since_days_ago:  int   = Field(30, ge=1, le=365)
    timeframe:       str   = Field("1m",         description="1m | 5m | 15m | 1h")

    gamma:           float = Field(0.05,  gt=0, description="Risk aversion (0.01–0.5)")
    kappa:           float = Field(1.5,   gt=0, description="Order arrival intensity")
    T_hours:         float = Field(1.0,   gt=0, description="Rolling time horizon (hours)")
    max_inventory:   float = Field(5.0,   gt=0)
    order_size:      float = Field(0.1,   gt=0)
    maker_fee:       float = Field(-0.0001)
    taker_fee:       float = Field(0.0004)
    vol_window_bars: int   = Field(60,    ge=2)


class SensitivityRequest(BacktestRequest):
    param:  Literal["gamma", "kappa"] = "gamma"


class AssetScoreOut(BaseModel):
    symbol:            str
    exchange:          str
    spread_bps:        float
    daily_volume_m:    float
    sigma_daily_pct:   float
    net_spread_bps:    float
    spread_vol_ratio:  float
    mm_score:          float


class BacktestOut(BaseModel):
    symbol:      str
    model:       str
    n_bars:      int
    n_fills:     int
    metrics:     dict
    # Downsampled series for charting (max 500 points)
    timestamps:  list[float]
    mid_prices:  list[float]
    inventories: list[float]
    running_pnl: list[float]
    bid_quotes:  list[float | None]
    ask_quotes:  list[float | None]
    fills:       list[dict]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _downsample(lst: list, max_points: int = 500) -> list:
    if len(lst) <= max_points:
        return lst
    step = len(lst) // max_points
    return lst[::step][:max_points]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/backtest", response_model=BacktestOut)
def backtest(req: BacktestRequest) -> BacktestOut:
    """
    Run an Avellaneda-Stoikov market making backtest.

    Fetches OHLCV data from the exchange (or uses synthetic data offline),
    simulates the MM strategy, and returns the full P&L decomposition.
    """
    df = fetch_ohlcv(
        symbol       = req.symbol,
        exchange_id  = req.exchange,
        timeframe    = req.timeframe,
        since_days_ago = req.since_days_ago,
    )
    if df.empty:
        raise HTTPException(status_code=204, detail="No data returned.")

    cfg = BacktestConfig(
        symbol          = req.symbol,
        exchange        = req.exchange,
        model           = req.model,
        gamma           = req.gamma,
        kappa           = req.kappa,
        T_hours         = req.T_hours,
        max_inventory   = req.max_inventory,
        order_size      = req.order_size,
        maker_fee       = req.maker_fee,
        taker_fee       = req.taker_fee,
        vol_window_bars = req.vol_window_bars,
        flat_at_end     = True,
    )

    result = run_backtest(df, cfg)

    n = len(result.timestamps)
    return BacktestOut(
        symbol      = req.symbol,
        model       = req.model,
        n_bars      = n,
        n_fills     = len(result.fills),
        metrics     = result.metrics,
        timestamps  = _downsample(result.timestamps),
        mid_prices  = _downsample(result.mid_prices),
        inventories = _downsample(result.inventories),
        running_pnl = _downsample(result.running_pnl),
        bid_quotes  = _downsample(result.bid_quotes),
        ask_quotes  = _downsample(result.ask_quotes),
        fills       = [
            {
                "timestamp": f.timestamp, "side": f.side,
                "price": f.price, "qty": f.qty,
                "mid_at_fill": f.mid_at_fill,
                "mid_5s_after": f.mid_5s_after,
                "spread_quoted": f.spread_quoted,
            }
            for f in result.fills
        ],
    )


@router.post("/sensitivity")
def sensitivity(req: SensitivityRequest) -> list[dict]:
    """
    Sweep γ or κ and return metrics for each value.

    Use this to find the optimal parameter for the target asset.
    """
    df = fetch_ohlcv(
        symbol       = req.symbol,
        exchange_id  = req.exchange,
        timeframe    = req.timeframe,
        since_days_ago = req.since_days_ago,
    )
    if df.empty:
        raise HTTPException(status_code=204, detail="No data.")

    cfg = BacktestConfig(
        symbol=req.symbol, exchange=req.exchange, model=req.model,
        gamma=req.gamma, kappa=req.kappa, T_hours=req.T_hours,
        max_inventory=req.max_inventory, order_size=req.order_size,
        maker_fee=req.maker_fee, taker_fee=req.taker_fee,
        vol_window_bars=req.vol_window_bars,
    )

    if req.param == "gamma":
        return gamma_sensitivity(df, cfg)
    else:
        return kappa_sensitivity(df, cfg)


@router.get("/asset-scores", response_model=list[AssetScoreOut])
def asset_scores() -> list[AssetScoreOut]:
    """Return candidate assets ranked by market making suitability score."""
    return [
        AssetScoreOut(
            symbol           = a.symbol,
            exchange         = a.exchange,
            spread_bps       = a.spread_bps,
            daily_volume_m   = a.daily_volume_m,
            sigma_daily_pct  = round(a.sigma_daily * 100, 2),
            net_spread_bps   = round(a.net_spread_bps, 2),
            spread_vol_ratio = round(a.spread_vol_ratio, 4),
            mm_score         = round(a.mm_score, 2),
        )
        for a in rank_candidates()
    ]
