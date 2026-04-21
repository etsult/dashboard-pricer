"""
Market Making Performance Metrics
===================================

Full P&L decomposition and risk metrics for evaluating a market making
strategy.  Based on the framework in:

  Cartea, Jaimungal & Penalva (2015), Chapter 2 & 10.
  Ho & Stoll (1981) — adverse selection decomposition.

P&L decomposition
-----------------
    Total PnL = Spread Income - Adverse Selection - Inventory PnL

  Spread Income     = Σ (filled_ask - filled_bid) / 2  × filled_qty
  Adverse Selection = Σ (mid_after_fill - mid_at_fill) × signed_qty
  Inventory PnL     = inventory × (final_mid - entry_mid)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class Fill:
    """One executed order."""
    timestamp: float        # unix seconds
    side: str               # 'bid' or 'ask'
    price: float            # execution price
    qty: float              # base-asset quantity (positive)
    mid_at_fill: float      # mid price at the moment of fill
    mid_5s_after: float     # mid price 5 seconds after fill (adverse selection proxy)
    spread_quoted: float    # full spread quoted at this moment (bps)


@dataclass
class BacktestResult:
    """
    Complete backtest output.

    All monetary values in quote currency (e.g. USDT).
    """
    # ── raw series ────────────────────────────────────────────────
    timestamps:      list[float] = field(default_factory=list)
    mid_prices:      list[float] = field(default_factory=list)
    inventories:     list[float] = field(default_factory=list)
    running_pnl:     list[float] = field(default_factory=list)
    bid_quotes:      list[float | None] = field(default_factory=list)
    ask_quotes:      list[float | None] = field(default_factory=list)
    fills:           list[Fill]  = field(default_factory=list)

    # ── computed on demand (see compute_metrics) ──────────────────
    metrics: dict = field(default_factory=dict)

    # ── parameters used ───────────────────────────────────────────
    symbol: str = ""
    model:  str = ""

    def compute_metrics(self) -> dict:
        """
        Compute the full suite of MM performance metrics.
        Stores result in self.metrics and returns it.
        """
        fills = self.fills
        if not fills or not self.running_pnl:
            self.metrics = {}
            return {}

        pnl = self.running_pnl
        inv = self.inventories

        # ── Fill stats ────────────────────────────────────────────
        bid_fills = [f for f in fills if f.side == 'bid']
        ask_fills = [f for f in fills if f.side == 'ask']
        n_fills   = len(fills)

        # ── Gross spread income ───────────────────────────────────
        # Each round-trip (1 bid fill + 1 ask fill) captures the spread.
        # Approximate: for each fill, income = |price - mid| × qty
        spread_income = sum(
            abs(f.price - f.mid_at_fill) * f.qty for f in fills
        )

        # ── Adverse selection ─────────────────────────────────────
        # After a bid fill, if mid goes DOWN → adverse (we bought high).
        # After an ask fill, if mid goes UP  → adverse (we sold low).
        adverse = 0.0
        for f in fills:
            delta_mid = f.mid_5s_after - f.mid_at_fill
            if f.side == 'bid':
                adverse += max(0.0, -delta_mid) * f.qty   # mid fell after buy
            else:
                adverse += max(0.0,  delta_mid) * f.qty   # mid rose after sell

        # ── Inventory P&L ─────────────────────────────────────────
        final_inv   = inv[-1]
        final_mid   = self.mid_prices[-1]
        # Mark inventory to market
        inventory_pnl = final_inv * final_mid

        # ── Total PnL ─────────────────────────────────────────────
        total_pnl   = pnl[-1]

        # ── Spread capture ratio ──────────────────────────────────
        # Ratio of actual spread earned vs half-spread quoted
        # < 1 means adverse selection is eroding profit
        quoted_half_spreads = [
            f.spread_quoted / 2 for f in fills if f.spread_quoted > 0
        ]
        avg_quoted_half = sum(quoted_half_spreads) / len(quoted_half_spreads) if quoted_half_spreads else 0
        actual_half_spreads = [abs(f.price - f.mid_at_fill) for f in fills]
        avg_actual_half = sum(actual_half_spreads) / len(actual_half_spreads) if actual_half_spreads else 0
        capture_ratio = avg_actual_half / avg_quoted_half if avg_quoted_half > 0 else 0

        # ── Inventory metrics ─────────────────────────────────────
        abs_inv = [abs(i) for i in inv]
        max_inv = max(abs_inv)
        twai    = sum(abs_inv) / len(abs_inv)  # time-weighted average |inventory|

        # ── Drawdown ──────────────────────────────────────────────
        peak = pnl[0]
        max_dd = 0.0
        for p in pnl:
            if p > peak:
                peak = p
            dd = peak - p
            if dd > max_dd:
                max_dd = dd

        # ── Sharpe (daily, annualised) ────────────────────────────
        # Need daily P&L series
        if len(pnl) > 1:
            daily_pnl = _to_daily(self.timestamps, pnl)
            if len(daily_pnl) > 1:
                mean_d = sum(daily_pnl) / len(daily_pnl)
                std_d  = math.sqrt(sum((x - mean_d)**2 for x in daily_pnl) / max(len(daily_pnl)-1, 1))
                sharpe = (mean_d / std_d * math.sqrt(365)) if std_d > 0 else 0.0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # ── Volume ────────────────────────────────────────────────
        total_volume_base  = sum(f.qty for f in fills)
        total_volume_quote = sum(f.qty * f.price for f in fills)

        # ── Adverse selection ratio ───────────────────────────────
        adv_ratio = adverse / spread_income if spread_income > 0 else 0

        self.metrics = {
            # P&L decomposition
            "total_pnl":          round(total_pnl, 4),
            "spread_income":      round(spread_income, 4),
            "adverse_selection":  round(adverse, 4),
            "inventory_pnl":      round(inventory_pnl, 4),
            "adverse_sel_ratio":  round(adv_ratio, 4),   # adverse / gross spread

            # Fills
            "n_fills":            n_fills,
            "n_bid_fills":        len(bid_fills),
            "n_ask_fills":        len(ask_fills),
            "fill_imbalance":     round((len(bid_fills) - len(ask_fills)) / max(n_fills, 1), 3),

            # Spread
            "avg_quoted_spread_bps": round(avg_quoted_half * 2 * 10_000, 2),
            "spread_capture_ratio":  round(capture_ratio, 3),

            # Inventory
            "final_inventory":    round(final_inv, 6),
            "max_inventory":      round(max_inv, 6),
            "twai":               round(twai, 6),          # time-weighted avg |inventory|

            # Risk
            "max_drawdown":       round(max_dd, 4),
            "sharpe_annualised":  round(sharpe, 3),

            # Volume
            "total_volume_base":  round(total_volume_base, 4),
            "total_volume_quote": round(total_volume_quote, 2),
            "pnl_per_1m_volume":  round(total_pnl / total_volume_quote * 1_000_000, 2)
                                  if total_volume_quote > 0 else 0,
        }
        return self.metrics


def _to_daily(timestamps: Sequence[float], values: Sequence[float]) -> list[float]:
    """Aggregate a running P&L series into daily increments."""
    if not timestamps:
        return []
    day_pnl: dict[int, float] = {}
    prev_val = values[0]
    for ts, val in zip(timestamps, values):
        day = int(ts // 86_400)
        day_pnl[day] = val - prev_val
        prev_val = val
    return list(day_pnl.values())
