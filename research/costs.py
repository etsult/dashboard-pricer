"""
Transaction cost model for option strategy backtests.

Models the real frictions that eat into theoretical P&L:
  1. Bid-ask spread  — you always buy at ask, sell at bid
  2. Commission      — exchange fee per trade
  3. Slippage        — market impact for larger sizes
  4. Funding rate    — daily cost of holding the delta hedge via perp futures

Usage:
    from research.costs import CostModel
    costs = CostModel(spread_pct=0.05, commission_pct=0.03, funding_rate_daily=0.0003)
    entry_cost = costs.option_entry(premium=500, n_legs=2)
    hedge_cost = costs.hedge_rebalance(spot=50000, delta_btc=0.1)
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CostModel:
    """
    All rates are in decimal (e.g. 0.05 = 5 bps = 0.05%).

    Deribit realistic defaults (as of 2024):
      - Spread on ATM options:  ~5-10 bps of underlying
      - Taker commission:       0.03% of underlying per contract
      - Perp funding:           ~0.01% per 8h = 0.03% per day (varies)
      - Slippage:               ~1-2 bps for <$100k notional
    """
    spread_pct: float = 0.05 / 100        # bid-ask spread as % of underlying
    commission_pct: float = 0.03 / 100    # exchange commission per leg
    slippage_pct: float = 0.01 / 100      # additional market impact
    funding_rate_daily: float = 0.0003    # daily funding on delta hedge (perp)

    def option_entry(self, premium: float, n_legs: int = 1) -> float:
        """
        Cost to enter `n_legs` option legs.
        Spread + commission applied on each leg.
        Returns total cost in $.
        """
        # Each leg costs: spread (half on each side) + commission
        cost_per_leg = premium * (self.spread_pct / 2 + self.commission_pct + self.slippage_pct)
        return cost_per_leg * n_legs

    def option_exit(self, premium: float, n_legs: int = 1) -> float:
        """Cost to exit `n_legs` option legs (same model as entry)."""
        return self.option_entry(premium, n_legs)

    def hedge_rebalance(self, spot: float, delta_btc: float) -> float:
        """
        Cost of rebalancing the delta hedge by `delta_btc` BTC at `spot`.
        Uses the spot/perp market — spread + commission on the trade size.
        """
        notional = abs(delta_btc) * spot
        return notional * (self.spread_pct / 2 + self.commission_pct + self.slippage_pct)

    def daily_funding(self, hedge_notional_usd: float) -> float:
        """
        Daily funding cost for holding `hedge_notional_usd` of perp futures.
        Positive = you pay (long perp); negative = you receive (short perp).
        For a delta-hedged short straddle, the hedge oscillates around zero.
        """
        return abs(hedge_notional_usd) * self.funding_rate_daily

    @classmethod
    def zero(cls) -> "CostModel":
        """No-cost model — useful for comparing gross vs net P&L."""
        return cls(
            spread_pct=0.0,
            commission_pct=0.0,
            slippage_pct=0.0,
            funding_rate_daily=0.0,
        )

    @classmethod
    def deribit_taker(cls) -> "CostModel":
        """Realistic Deribit taker costs."""
        return cls(
            spread_pct=0.05 / 100,
            commission_pct=0.03 / 100,
            slippage_pct=0.01 / 100,
            funding_rate_daily=0.0003,
        )
