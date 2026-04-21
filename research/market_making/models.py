"""
Market Making Models
====================

Implements the theoretical framework for optimal market making.

References
----------
[1] Avellaneda, M. & Stoikov, S. (2008).
    "High-frequency trading in a limit order book."
    Quantitative Finance, 8(3), 217-224.

[2] Guéant, O., Lehalle & Fernandez-Tapia (2013).
    "Dealing with the Inventory Risk."
    Mathematics and Financial Economics, 7(4), 477-507.

[3] Cartea, Jaimungal & Penalva (2015).
    "Algorithmic and High-Frequency Trading." Cambridge University Press.

Calibration note — FRACTIONAL SPACE
-------------------------------------
All quantities live in *return / fractional* space so that the same
parameters work for any asset price level (BTC at $60k or SOL at $150).

The AS spread formula:

    δ*_frac = γ · σ̃² · T  +  (2/γ) · ln(1 + γ/κ)
              ──────────────   ───────────────────
              inventory risk    order-arrival risk

where:
  σ̃  = per-bar log-return std (e.g. 0.00112 for SOL 1-min, not annualised)
  T   = time horizon IN BARS (e.g. 60 for 1-hour window on 1-min bars)
  γ   = dimensionless risk-aversion coefficient
  κ   = 1 / (target half-spread as fraction)
        e.g. κ=3333 → target 3 bps half-spread (1/3333 = 0.03%)

Physical intuition:
  • The arrival term (2/γ)·ln(1+γ/κ) ≈ 2/κ for small γ/κ.
    So the "baseline" full spread ≈ 2/κ.
    κ = 3333 → full spread ≈ 6 bps. κ = 1666 → 12 bps.
  • The inventory term γ·σ̃²·T widens the spread as vol or horizon grows.
  • As inventory x grows, the reservation price r tilts away from mid,
    shifting both quotes toward the side that reduces inventory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass
class ASParams:
    """
    All parameters in dimensionless / fractional space.

    gamma : float
        Risk-aversion (dimensionless).
        γ = 0.1  → moderate; γ = 0.5  → aggressive inventory management.

    kappa : float
        Order-arrival sensitivity = 1 / (target half-spread fraction).
        κ = 3333 → 3 bps target  (liquid, e.g. SOL/USDT)
        κ = 1666 → 6 bps target  (mid-cap, e.g. AVAX)
        κ = 833  → 12 bps target (small-cap)

    T_bars : int / float
        Rolling time horizon in bars (e.g. 60 on 1-min data = 1 hour).

    max_inventory : float
        Max |inventory| in base-asset units before quotes are cancelled.

    order_size : float
        Base-asset units per order.

    maker_fee : float
        Maker fee as a fraction (negative = rebate, e.g. -0.0001).

    taker_fee : float
        Taker fee for emergency flattening (e.g. 0.0004).

    max_half_spread_frac : float
        Hard cap on half-spread as a fraction of mid (default 0.005 = 50 bps).
        Prevents absurd quotes during extreme volatility.
    """
    gamma:               float = 0.1
    kappa:               float = 3333.0
    T_bars:              float = 60.0
    max_inventory:       float = 5.0
    order_size:          float = 0.1
    maker_fee:           float = -0.0001
    taker_fee:           float = 0.0004
    max_half_spread_frac: float = 0.005   # 50 bps hard cap


@dataclass
class MMState:
    cash:        float = 0.0
    inventory:   float = 0.0
    n_fills_bid: int   = 0
    n_fills_ask: int   = 0
    n_quotes:    int   = 0


# ── Avellaneda-Stoikov (2008) — fractional space ──────────────────────────────

class AvellanedaStoikov:
    """
    Optimal symmetric market maker (Avellaneda & Stoikov 2008).

    All quantities in fractional / log-return space.
    Quotes are returned as absolute prices.
    """

    def __init__(self, params: ASParams):
        self.p = params

    def reservation_price(
        self, mid: float, inventory: float, sigma_per_bar: float, t_bars: float
    ) -> float:
        """
        Price adjusted for inventory imbalance (AS eq. 7).

          r = mid × (1 − x · γ · σ̃² · T)

        Long inventory (x > 0) → r tilts below mid (encourage selling).
        """
        tilt = inventory * self.p.gamma * sigma_per_bar ** 2 * t_bars
        tilt = max(-0.05, min(0.05, tilt))   # clamp to ±5% for stability
        return mid * (1.0 - tilt)

    def optimal_half_spread_frac(self, sigma_per_bar: float, t_bars: float) -> float:
        """
        Half-spread as a fraction of mid (AS eq. 9 in fractional space).

          half = [γ·σ̃²·T  +  (2/γ)·ln(1+γ/κ)] / 2
        """
        inv_term  = self.p.gamma * sigma_per_bar ** 2 * t_bars
        arr_term  = (2.0 / self.p.gamma) * math.log(1.0 + self.p.gamma / self.p.kappa)
        half_frac = (inv_term + arr_term) / 2.0
        return min(half_frac, self.p.max_half_spread_frac)

    def quotes(
        self, mid: float, inventory: float, sigma_per_bar: float, t_bars: float
    ) -> Tuple[float | None, float | None]:
        """
        Return (bid_price, ask_price).
        Returns None on a side if inventory limit reached.
        """
        r    = self.reservation_price(mid, inventory, sigma_per_bar, t_bars)
        half = self.optimal_half_spread_frac(sigma_per_bar, t_bars)

        bid = r * (1.0 - half) if inventory <  self.p.max_inventory  else None
        ask = r * (1.0 + half) if inventory > -self.p.max_inventory  else None
        return bid, ask


class AvellanedaStoikovGueant(AvellanedaStoikov):
    """
    Extension from Guéant et al. (2013): asymmetric spread allocation.

    Instead of splitting δ* symmetrically, the full spread is kept the
    same but allocated asymmetrically based on inventory:

        bid distance = δ*/2 + skew   (wider if long → discourages more longs)
        ask distance = δ*/2 − skew   (tighter if long → encourages selling)

    where skew = γ · σ̃² · T · x   (fraction of mid)

    This produces faster mean-reversion of inventory with the SAME total
    spread as the basic AS model.
    """

    def quotes(
        self, mid: float, inventory: float, sigma_per_bar: float, t_bars: float
    ) -> Tuple[float | None, float | None]:
        r         = self.reservation_price(mid, inventory, sigma_per_bar, t_bars)
        half      = self.optimal_half_spread_frac(sigma_per_bar, t_bars)
        skew      = self.p.gamma * sigma_per_bar ** 2 * t_bars * inventory
        skew      = max(-half, min(half, skew))   # skew cannot flip the spread

        half_bid  = half + skew
        half_ask  = half - skew

        bid = r * (1.0 - half_bid) if inventory <  self.p.max_inventory  else None
        ask = r * (1.0 + half_ask) if inventory > -self.p.max_inventory  else None
        return bid, ask


# ── Asset scoring ─────────────────────────────────────────────────────────────

@dataclass
class AssetScore:
    """
    Scoring an asset for MM suitability.

    Key metric: spread_vol_ratio = spread / daily_vol
    Higher = more revenue per unit of inventory risk.
    Theoretical bound: a MM is profitable only when
       net_spread > adverse_selection_cost ≈ σ_per_trade
    """
    symbol:          str
    exchange:        str
    spread_bps:      float
    daily_volume_m:  float
    sigma_daily:     float
    maker_fee_bps:   float = 1.0

    @property
    def spread_vol_ratio(self) -> float:
        if self.sigma_daily <= 0:
            return 0.0
        return (self.spread_bps / 10_000) / self.sigma_daily

    @property
    def net_spread_bps(self) -> float:
        return self.spread_bps - self.maker_fee_bps

    @property
    def mm_score(self) -> float:
        import math
        vol_score    = math.log1p(self.daily_volume_m)
        spread_score = self.net_spread_bps
        risk_score   = self.spread_vol_ratio * 100
        return vol_score * spread_score * risk_score

    def __repr__(self) -> str:
        return (f"AssetScore({self.symbol} on {self.exchange} | "
                f"spread={self.spread_bps:.1f}bps σ_daily={self.sigma_daily*100:.1f}% "
                f"score={self.mm_score:.2f})")


CANDIDATE_ASSETS: list[AssetScore] = [
    AssetScore("BTC/USDT",  "Binance",   0.5,  20_000,  0.025,  0.5),
    AssetScore("ETH/USDT",  "Binance",   0.8,   8_000,  0.030,  0.5),
    AssetScore("SOL/USDT",  "Binance",   2.0,   3_000,  0.045,  1.0),
    AssetScore("AVAX/USDT", "Binance",   3.5,     800,  0.055,  1.0),
    AssetScore("BNB/USDT",  "Binance",   1.5,   1_500,  0.035,  0.5),
    AssetScore("ARB/USDT",  "Binance",   4.0,     400,  0.060,  1.0),
    AssetScore("DOGE/USDT", "Binance",   2.5,   1_200,  0.050,  1.0),
    AssetScore("WIF/USDT",  "Binance",   8.0,     300,  0.080,  1.0),
    AssetScore("BTC-PERP",  "Deribit",   1.0,   2_000,  0.025,  0.0),
    AssetScore("ETH-PERP",  "Deribit",   1.5,   1_000,  0.030,  0.0),
]


def rank_candidates(candidates: list[AssetScore] = CANDIDATE_ASSETS) -> list[AssetScore]:
    return sorted(candidates, key=lambda a: a.mm_score, reverse=True)
