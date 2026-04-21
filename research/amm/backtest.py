"""
Uniswap v3 LP Backtest Engine
==============================

Simulates a concentrated liquidity position on historical OHLCV data.

Fee model
---------
  Fee income per bar (when in range):
      fee_bar = fee_tier × volume_tvl_ratio / bars_per_day × V_position

IL tracking
-----------
  IL is tracked continuously including across rebalances.
  At each rebalance, the IL of the expiring position is crystallised
  into a permanent capital loss (on top of the explicit rebalancing cost).

  Total IL = Σ il_at_each_rebalance + current_open_il

Rebalancing
-----------
  Two strategies:
    1. "none"          — hold until end; IL crystallises at final price
    2. "out_of_range"  — re-centre range whenever price exits [P_a, P_b]
                         (triggers rebalance_cost_bps cost from gas + taker)

  Note: with out_of_range rebalancing, the LP keeps earning fees but
  pays transaction costs. The break-even time out-of-range before
  rebalancing is worth it ≈ rebalance_cost / fee_rate_per_bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .models import UniV3Position
from .metrics import AMMResult


@dataclass
class AMMConfig:
    symbol:             str   = "SOL/USDT"
    fee_tier:           float = 0.0005    # 0.05%
    range_half_width:   float = 0.10      # ±10% in log-price
    initial_capital:    float = 10_000.0
    volume_tvl_ratio:   float = 1.0       # daily volume / TVL
    bars_per_day:       int   = 1440
    rebalance_strategy: str   = "out_of_range"   # "none" | "out_of_range"
    rebalance_cost_bps: float = 10.0      # gas + taker, bps of position value


def run_amm_backtest(df: pd.DataFrame, cfg: AMMConfig) -> AMMResult:
    """Simulate a UniV3 concentrated LP position on OHLCV data."""
    result = AMMResult(symbol=cfg.symbol, fee_tier=cfg.fee_tier, range_w=cfg.range_half_width)

    fee_yield_per_bar = cfg.fee_tier * cfg.volume_tvl_ratio / cfg.bars_per_day

    first   = df.iloc[0]
    P_entry = (float(first["high"]) + float(first["low"])) / 2.0
    pos     = UniV3Position(P_entry, cfg.range_half_width, cfg.initial_capital, cfg.fee_tier)

    # Track initial 50/50 allocation for a clean hodl baseline
    x_init = pos.x_0
    y_init = pos.y_0

    cum_fees      = 0.0
    cum_il        = 0.0     # total realised IL (from rebalances) + open unrealised IL
    realised_il   = 0.0     # IL crystallised at each rebalance
    rebal_costs   = 0.0
    n_rebalances  = 0

    for i in range(len(df)):
        row   = df.iloc[i]
        ts    = float(row["timestamp"])
        P     = (float(row["high"]) + float(row["low"])) / 2.0
        in_rng = pos.in_range(P)

        # ── Fee accrual ────────────────────────────────────────────────────
        if in_rng:
            cum_fees += fee_yield_per_bar * pos.value(P)

        # ── Current open IL ────────────────────────────────────────────────
        open_il  = pos.il_usd(P)
        cum_il   = realised_il + open_il

        # ── Rebalance when out of range ────────────────────────────────────
        if not in_rng and cfg.rebalance_strategy == "out_of_range":
            cost         = pos.value(P) * cfg.rebalance_cost_bps / 10_000.0
            realised_il += open_il                   # crystallise IL
            rebal_costs += cost
            n_rebalances += 1
            new_capital  = pos.value(P) - cost
            pos          = UniV3Position(P, cfg.range_half_width, new_capital, cfg.fee_tier)
            in_rng       = True
            cum_il       = realised_il               # open IL resets to 0 post-rebalance

        # ── Hodl baseline (original 50/50 held without LP) ────────────────
        hodl_val = x_init * P + y_init

        # ── Total value = LP position + accumulated fees ───────────────────
        total_val = pos.value(P) + cum_fees

        result.timestamps.append(ts)
        result.mid_prices.append(P)
        result.position_values.append(total_val)          # total incl. fees
        result.hodl_values.append(hodl_val)
        result.fee_cumulative.append(cum_fees)
        result.il_cumulative.append(cum_il)
        result.in_range_flags.append(in_rng)

    result.n_rebalances    = n_rebalances
    result.rebalance_costs = rebal_costs
    result.compute_metrics()
    return result


# ── Sensitivity sweeps ────────────────────────────────────────────────────────

def range_width_sensitivity(
    df: pd.DataFrame,
    cfg: AMMConfig,
    widths: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.40, 0.80),
) -> list[dict]:
    rows = []
    for w in widths:
        c = AMMConfig(**{**cfg.__dict__, "range_half_width": w})
        r = run_amm_backtest(df, c)
        m = {**r.metrics, "range_half_width": w,
             "range_pct": round((math.exp(w) - 1) * 100, 1)}
        rows.append(m)
    return rows


def fee_tier_sensitivity(
    df: pd.DataFrame,
    cfg: AMMConfig,
    fee_tiers:         tuple[float, ...] = (0.0001, 0.0005, 0.003, 0.01),
    volume_tvl_ratios: tuple[float, ...] = (3.0,    1.0,    0.5,   0.2),
) -> list[dict]:
    """Compare fee tiers with realistic volume/TVL ratios per tier."""
    rows = []
    for ft, vr in zip(fee_tiers, volume_tvl_ratios):
        c = AMMConfig(**{**cfg.__dict__, "fee_tier": ft, "volume_tvl_ratio": vr})
        r = run_amm_backtest(df, c)
        m = {**r.metrics, "fee_tier": ft, "fee_bps": round(ft * 10_000, 0),
             "volume_tvl_ratio": vr}
        rows.append(m)
    return rows
