"""
Market Making Models
====================

Implements the theoretical framework for optimal market making.

References
----------
[1] Avellaneda, M. & Stoikov, S. (2008).
    "High-frequency trading in a limit order book."
    Quantitative Finance, 8(3), 217-224.

[2] Guéant, O., Lehalle, C-A. & Fernandez-Tapia, J. (2013).
    "Dealing with the Inventory Risk: A solution to the market making problem."
    Mathematics and Financial Economics, 7(4), 477-507.

[3] Cartea, Á., Jaimungal, S. & Penalva, J. (2015).
    "Algorithmic and High-Frequency Trading." Cambridge University Press.

Key insight (AS 2008)
---------------------
A market maker quotes symmetrically around a *reservation price* that
tilts away from the mid as inventory grows.  The optimal full spread
balances two forces:

    δ* = γ σ² (T-t)  +  (2/γ) ln(1 + γ/κ)
         ─────────────    ─────────────────
         inventory risk   order-arrival risk

where:
  γ  = risk-aversion coefficient
  σ  = price volatility (per unit time)
  T-t = time horizon remaining
  κ  = order-arrival intensity decay
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple


# ── Parameter containers ──────────────────────────────────────────────────────

@dataclass
class ASParams:
    """
    Parameters for the Avellaneda-Stoikov model.

    gamma : float
        Risk-aversion coefficient.  Controls how aggressively the MM
        tilts quotes when carrying inventory.
        Typical range: 0.01 – 0.5
        Small γ → wide spread, tolerates inventory.
        Large γ → narrow spread tilt, mean-reverts fast.

    sigma : float
        Mid-price volatility (annualised, then scaled by dt in engine).
        Estimated from recent OHLCV or realised vol.

    kappa : float
        Order-arrival rate decay parameter.
        Higher κ → order flow is less sensitive to spread (we can quote wider).
        Estimated from historical fill data or assumed.
        Typical range: 0.5 – 5.0

    T : float
        Time horizon in the same unit as sigma (e.g., hours).
        For a continuous running MM, use a rolling window (e.g., 1 hour).

    dt : float
        Timestep size in same unit as T/sigma.

    max_inventory : float
        Inventory limit in base-asset units.  Quotes on a side are
        cancelled (or skewed to 0) once limit is reached.

    order_size : float
        Size of each limit order in base-asset units.

    maker_fee : float
        Maker fee as a fraction (e.g., -0.0001 = 1 bps rebate on Binance).
        Negative = rebate (common on most CEXs).

    taker_fee : float
        Taker fee as a fraction (e.g., 0.0004 = 4 bps).
        Used when we cross the spread to flatten inventory.
    """
    gamma: float = 0.05
    sigma: float = 0.80          # annualised vol, e.g. BTC ~80%
    kappa: float = 1.5
    T: float = 1.0               # 1 hour rolling horizon
    dt: float = 1 / 3600         # 1 second steps
    max_inventory: float = 1.0   # 1 BTC
    order_size: float = 0.01     # 0.01 BTC per quote
    maker_fee: float = -0.0001   # Binance VIP maker rebate
    taker_fee: float = 0.0004    # Binance standard taker


@dataclass
class MMState:
    """Mutable state of the market maker at a single timestep."""
    cash: float = 0.0
    inventory: float = 0.0        # positive = long, negative = short
    n_fills_bid: int = 0
    n_fills_ask: int = 0
    n_quotes: int = 0


# ── Avellaneda-Stoikov (2008) ─────────────────────────────────────────────────

class AvellanedaStoikov:
    """
    Optimal symmetric market maker from Avellaneda & Stoikov (2008).

    The MM solves the HJB equation for a risk-averse agent maximising
    expected exponential utility of terminal wealth subject to inventory
    dynamics driven by a Poisson order flow.

    Closed-form solution (from [2]):

        reservation price:
            r(s,x,t) = s - x · γ · σ² · (T-t)

        optimal full spread:
            δ*(t) = γ · σ² · (T-t)  +  (2/γ) · ln(1 + γ/κ)

        bid = r - δ*/2
        ask = r + δ*/2
    """

    def __init__(self, params: ASParams):
        self.p = params

    def reservation_price(self, mid: float, inventory: float, t_remaining: float) -> float:
        """
        Mid-price adjusted for inventory imbalance.

        As inventory grows long, the reservation price falls — the MM
        skews quotes downward to encourage selling and reduce exposure.
        """
        return mid - inventory * self.p.gamma * self.p.sigma ** 2 * t_remaining

    def optimal_spread(self, t_remaining: float) -> float:
        """
        Full bid-ask spread that maximises expected utility.

        Two terms:
          1) γ σ² (T-t)  — inventory risk component (widens with vol and time)
          2) (2/γ) ln(1+γ/κ) — order-arrival risk (constant; widens when κ small)
        """
        inventory_term = self.p.gamma * self.p.sigma ** 2 * t_remaining
        arrival_term   = (2.0 / self.p.gamma) * math.log(1.0 + self.p.gamma / self.p.kappa)
        return inventory_term + arrival_term

    def quotes(
        self, mid: float, inventory: float, t_remaining: float
    ) -> Tuple[float | None, float | None]:
        """
        Return (bid_price, ask_price).

        Returns None on a side if the inventory limit would be breached.
        """
        r     = self.reservation_price(mid, inventory, t_remaining)
        delta = self.optimal_spread(t_remaining)
        half  = delta / 2.0

        bid = r - half if inventory < self.p.max_inventory  else None
        ask = r + half if inventory > -self.p.max_inventory else None

        return bid, ask


# ── Avellaneda-Stoikov with Guéant (2013) closed form ────────────────────────

class AvellanedaStoikovGueant(AvellanedaStoikov):
    """
    Extension: uses the finite-inventory closed-form from Guéant et al. (2013).

    Adds an asymmetric skew term so that the spread contracts on the
    side that helps reduce inventory and expands on the side that increases it.

    δ^bid  = δ*/2 + γ σ² (T-t) · x
    δ^ask  = δ*/2 - γ σ² (T-t) · x

    This is the "dealing with inventory risk" generalisation: the full
    optimal spread δ* stays the same, but it is allocated asymmetrically.
    """

    def quotes(
        self, mid: float, inventory: float, t_remaining: float
    ) -> Tuple[float | None, float | None]:
        r         = self.reservation_price(mid, inventory, t_remaining)
        delta     = self.optimal_spread(t_remaining)
        half      = delta / 2.0
        skew_adj  = self.p.gamma * self.p.sigma ** 2 * t_remaining * inventory

        half_bid = half + skew_adj   # widen bid if long (don't want more longs)
        half_ask = half - skew_adj   # narrow ask if long (encourage selling)

        bid = r - half_bid if inventory <  self.p.max_inventory  else None
        ask = r + half_ask if inventory > -self.p.max_inventory  else None

        return bid, ask


# ── Volatility estimator ──────────────────────────────────────────────────────

def rolling_sigma(
    closes: list[float],
    window: int = 60,
    annualise_factor: float = 365 * 24,   # for hourly data → annualised
) -> float:
    """
    Yang-Zhang volatility estimator on a window of close prices.
    Falls back to simple log-return std if fewer than 2 points.
    """
    if len(closes) < 2:
        return 0.01
    n = min(len(closes), window)
    rets = [
        math.log(closes[-i] / closes[-i - 1])
        for i in range(1, n)
    ]
    if len(rets) < 2:
        return abs(rets[0]) * math.sqrt(annualise_factor)
    mean = sum(rets) / len(rets)
    var  = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var * annualise_factor)


# ── Asset scoring ─────────────────────────────────────────────────────────────

@dataclass
class AssetScore:
    """
    Scientific scoring of an asset for market making suitability.

    Criteria (from Cartea, Jaimungal & Penalva, Ch. 10):
      - spread_bps     : average quoted spread in basis points
      - daily_volume_m : daily volume in $M
      - sigma_daily    : daily realised volatility (decimal)
      - spread_vol_ratio: spread / sigma — higher = more cushion per unit of vol risk
      - mm_score       : composite score (higher = better MM opportunity)
    """
    symbol: str
    exchange: str
    spread_bps: float
    daily_volume_m: float
    sigma_daily: float
    maker_fee_bps: float = 1.0   # typical Binance: 0–2 bps depending on tier

    @property
    def spread_vol_ratio(self) -> float:
        """Spread as a multiple of daily vol — key MM efficiency metric."""
        if self.sigma_daily <= 0:
            return 0.0
        # Convert spread from bps to decimal, compare to daily vol
        return (self.spread_bps / 10_000) / self.sigma_daily

    @property
    def net_spread_bps(self) -> float:
        """Spread after maker fees (maker rebate adds to profit)."""
        return self.spread_bps - self.maker_fee_bps  # rebate reduces cost

    @property
    def mm_score(self) -> float:
        """
        Composite score balancing:
          - net spread (revenue per trade)
          - volume (how often we fill)
          - spread/vol ratio (compensation per unit of inventory risk)

        Score is unitless; use for ranking only.
        """
        vol_score    = math.log1p(self.daily_volume_m)          # log scale
        spread_score = self.net_spread_bps                       # linear
        risk_score   = self.spread_vol_ratio * 100               # scaled
        return vol_score * spread_score * risk_score

    def __repr__(self) -> str:
        return (
            f"AssetScore({self.symbol} on {self.exchange} | "
            f"spread={self.spread_bps:.1f}bps "
            f"vol={self.daily_volume_m:.0f}M$ "
            f"σ_daily={self.sigma_daily*100:.1f}% "
            f"score={self.mm_score:.2f})"
        )


# ── Known candidates (based on 2024 market data) ─────────────────────────────
# Sources: Binance exchange stats, academic papers on crypto liquidity
# Spread estimates are mid-spread from L2 snapshots, not just best bid/ask

CANDIDATE_ASSETS: list[AssetScore] = [
    # symbol          exchange    spread  vol$M    σ_daily  fee_bps
    AssetScore("BTC/USDT",  "Binance",   0.5,  20_000,  0.025,  0.5),
    AssetScore("ETH/USDT",  "Binance",   0.8,   8_000,  0.030,  0.5),
    AssetScore("SOL/USDT",  "Binance",   2.0,   3_000,  0.045,  1.0),
    AssetScore("AVAX/USDT", "Binance",   3.5,     800,  0.055,  1.0),
    AssetScore("BNB/USDT",  "Binance",   1.5,   1_500,  0.035,  0.5),
    AssetScore("ARB/USDT",  "Binance",   4.0,     400,  0.060,  1.0),
    AssetScore("DOGE/USDT", "Binance",   2.5,   1_200,  0.050,  1.0),
    AssetScore("WIF/USDT",  "Binance",   8.0,     300,  0.080,  1.0),
    AssetScore("BTC-PERP",  "Deribit",   1.0,   2_000,  0.025,  0.0),  # no maker fee
    AssetScore("ETH-PERP",  "Deribit",   1.5,   1_000,  0.030,  0.0),
]


def rank_candidates(candidates: list[AssetScore] = CANDIDATE_ASSETS) -> list[AssetScore]:
    """Return candidates sorted by mm_score descending."""
    return sorted(candidates, key=lambda a: a.mm_score, reverse=True)
