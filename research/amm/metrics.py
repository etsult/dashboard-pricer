"""
AMM Performance Metrics
========================

P&L decomposition for a Uniswap v3 LP position:

    Net PnL   =  Fee Income  +  IL_total  −  Rebalance Costs
    IL_total  =  LP value  −  hodl value  (always ≤ 0)

A position is profitable when:
    Fee Income  >  |IL_total|  +  Rebalance Costs

Note: IL_total here is the full realised + unrealised impermanent loss
across all rebalancing periods, NOT just the current open position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class AMMResult:
    symbol:   str   = ""
    fee_tier: float = 0.0
    range_w:  float = 0.0

    # All series include accumulated fees in position_values
    timestamps:      list[float] = field(default_factory=list)
    mid_prices:      list[float] = field(default_factory=list)
    position_values: list[float] = field(default_factory=list)  # LP value + fees
    hodl_values:     list[float] = field(default_factory=list)  # 50/50 hodl baseline
    fee_cumulative:  list[float] = field(default_factory=list)
    il_cumulative:   list[float] = field(default_factory=list)  # realised + open IL
    in_range_flags:  list[bool]  = field(default_factory=list)
    n_rebalances:    int   = 0
    rebalance_costs: float = 0.0

    metrics: dict = field(default_factory=dict)

    def compute_metrics(self) -> dict:
        if not self.timestamps or not self.position_values:
            self.metrics = {}
            return {}

        V0         = self.hodl_values[0]    # initial capital (= hodl at t=0)
        final_val  = self.position_values[-1]
        hodl_final = self.hodl_values[-1]
        fees       = self.fee_cumulative[-1]
        il_total   = self.il_cumulative[-1]  # total IL (realised + open)

        net_pnl      = final_val - V0
        il_vs_hodl   = final_val - hodl_final   # LP+fees vs just hodling 50/50
        hodl_pnl     = hodl_final - V0

        # Daily PnL series from total value (position + fees)
        daily_series = _to_daily(self.timestamps, self.position_values)
        mean_d = std_d = sharpe = 0.0
        if len(daily_series) > 1:
            mean_d = sum(daily_series) / len(daily_series)
            var    = sum((x - mean_d) ** 2 for x in daily_series) / max(len(daily_series) - 1, 1)
            std_d  = math.sqrt(var)
            sharpe = (mean_d / std_d * math.sqrt(365)) if std_d > 0 else 0.0

        # Drawdown on total value
        peak   = self.position_values[0]
        max_dd = 0.0
        for v in self.position_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        time_in_range = sum(self.in_range_flags) / max(len(self.in_range_flags), 1)
        n_bars        = len(self.timestamps)
        ann_factor    = 365 * 1440 / n_bars    # annualisation from bar count

        self.metrics = {
            # P&L decomposition
            "total_pnl":         round(net_pnl, 4),
            "fee_income":        round(fees, 4),
            "il_total":          round(il_total, 4),     # always ≤ 0
            "il_vs_hodl":        round(il_vs_hodl, 4),   # LP+fees vs 50/50 hodl
            "hodl_pnl":          round(hodl_pnl, 4),
            "rebalance_costs":   round(self.rebalance_costs, 4),
            "n_rebalances":      self.n_rebalances,
            # Activity
            "time_in_range_pct": round(time_in_range * 100, 2),
            # Daily stats
            "daily_pnl_mean":    round(mean_d, 4),
            "daily_pnl_std":     round(std_d, 4),
            "sharpe_daily":      round(sharpe, 3),
            # Risk
            "max_drawdown_pct":  round(max_dd * 100, 2),
            "ann_return_pct":    round(net_pnl / V0 * ann_factor * 100, 2) if V0 > 0 else 0.0,
            # Capital
            "initial_capital":   round(V0, 2),
            "final_value":       round(final_val, 2),
            "hodl_final_value":  round(hodl_final, 2),
        }
        return self.metrics


def _to_daily(timestamps: list[float], values: list[float]) -> list[float]:
    day_first: dict[int, float] = {}
    day_last:  dict[int, float] = {}
    for ts, v in zip(timestamps, values):
        d = int(ts // 86_400)
        if d not in day_first:
            day_first[d] = v
        day_last[d] = v
    days = sorted(day_first)
    return [day_last[d] - day_first[d] for d in days]


def compare_strategies(amm_metrics: dict, mm_metrics: dict) -> dict:
    """Head-to-head comparison: AMM LP vs AS order book MM."""
    return {
        "amm_total_pnl":      amm_metrics.get("total_pnl", 0),
        "mm_total_pnl":       mm_metrics.get("total_pnl", 0),
        "amm_fee_income":     amm_metrics.get("fee_income", 0),
        "mm_spread_income":   mm_metrics.get("spread_income", 0),
        "amm_il":             amm_metrics.get("il_total", 0),
        "mm_adverse_sel":     mm_metrics.get("adverse_selection", 0),
        "amm_sharpe":         amm_metrics.get("sharpe_daily", 0),
        "mm_sharpe":          mm_metrics.get("sharpe_daily", 0),
        "amm_max_dd_pct":     amm_metrics.get("max_drawdown_pct", 0),
        "mm_max_dd_pct":      mm_metrics.get("max_drawdown", 0),
        "amm_ann_return_pct": amm_metrics.get("ann_return_pct", 0),
        "amm_vs_hodl":        amm_metrics.get("il_vs_hodl", 0),
        "winner":             "AMM LP" if amm_metrics.get("total_pnl", 0) > mm_metrics.get("total_pnl", 0)
                              else "Order Book MM",
    }
