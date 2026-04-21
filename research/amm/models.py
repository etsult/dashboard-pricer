"""
Uniswap v3 Concentrated Liquidity Model
=========================================

References
----------
[1] Adams et al. (2021) "Uniswap v3 Core" — whitepaper
[2] Milionis, Moallemi, Roughgarden & Zhang (2022)
    "Automated Market Making and Arbitrage Profits under Invariant Function AMMs"
    https://arxiv.org/abs/2208.06046
[3] Fukasawa, Maire & Wunsch (2023)
    "Weighted variance swaps hedge against impermanent loss"

The connection to Avellaneda-Stoikov
--------------------------------------
Both models solve the same tradeoff:

    AS order book MM          Uniswap v3 LP
    ─────────────────         ──────────────
    half-spread  δ      ↔     range half-width  w  (log-price)
    spread income       ↔     fee income
    adverse selection   ↔     impermanent loss (IL)
    inventory skew      ↔     delta (auto-rebalanced by AMM curve)

IL formula (Milionis 2022, Thm 1)
----------------------------------
For a Uniswap v2 LP (full range) over time T:

    IL ≈  −σ²·T / 2  ×  V₀       (short-variance position)

For Uniswap v3 concentrated liquidity at range half-width w:

    IL ≈  −σ²·T / 2  ×  V₀ × c(w)

where c(w) = concentration multiplier ≈ 1/(2w) × σ√T for narrow ranges.
The tighter the range, the more capital efficiency but the more IL per bar
when price moves.

Optimal range width (analogous to AS δ*)
-----------------------------------------
    w* = σ / √(2 × fee_yield_per_bar)

    where fee_yield_per_bar = fee_tier × volume_tvl_ratio / bars_per_day

This mirrors AS:  δ* ∝ σ / √κ  (larger κ → tighter spread → same tradeoff).

LP as short-vol trade
----------------------
An AMM LP earns theta (fees) and loses through gamma (IL). It is
structurally equivalent to selling a strangle:
  - Short call at P_b
  - Short put  at P_a
  - Premium earned = accumulated fees
  - Loss from gamma = IL
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class UniV3Params:
    """
    Parameters for a Uniswap v3 LP strategy.

    fee_tier : float
        Pool fee tier as a fraction (0.0001=0.01%, 0.0005=0.05%,
        0.003=0.30%, 0.01=1.00%).

    range_half_width : float
        Half-width of the liquidity range in log-price units.
        w=0.10 means range = [P·e^{-0.10}, P·e^{+0.10}] ≈ [P/1.105, P×1.105].
        Typical values: 0.05 (tight, ~5% each side) to 0.50 (wide, ~65%).

    volume_tvl_ratio : float
        Daily trading volume / total value locked in the pool.
        Calibration: Uniswap v3 ETH/USDC 0.05%: ~0.8; SOL pools: ~1.0-2.0.

    rebalance_cost_bps : float
        Round-trip cost to close + re-open the position when out of range:
        gas fees + taker spread. Typically 5-20 bps on Ethereum, 1-5 on L2s.
    """
    fee_tier:           float = 0.0005   # 5 bps
    range_half_width:   float = 0.10     # ±10% in log-price
    volume_tvl_ratio:   float = 1.0      # daily V/TVL
    rebalance_cost_bps: float = 10.0     # 10 bps per rebalance


class UniV3Position:
    """
    A single concentrated liquidity position on Uniswap v3.

    Entry at price P_0 with symmetric range [P_0·e^{-w}, P_0·e^{+w}].
    Assumes a 50/50 balanced entry (equal USD value on each side).

    Virtual reserve formulas (Adams 2021, eq. 2.2):
        x(P) = L · (1/√P_eff − 1/√P_b)      token0 (base, e.g. ETH)
        y(P) = L · (√P_eff − √P_a)            token1 (quote, e.g. USDC)

    where P_eff = clamp(P, P_a, P_b).
    """

    def __init__(self, P_0: float, w: float, V_0: float, fee_tier: float):
        self.P_0      = P_0
        self.w        = w
        self.P_a      = P_0 * math.exp(-w)
        self.P_b      = P_0 * math.exp(+w)
        self.V_0      = V_0
        self.fee_tier = fee_tier

        # Liquidity from balanced entry:  V_0 = 2L√P_0·(1 − e^{−w/2})
        self.L = V_0 / (2.0 * math.sqrt(P_0) * (1.0 - math.exp(-w / 2.0)))

        # Initial reserves (used for hodl comparison)
        self.x_0, self.y_0 = self._reserves(P_0)

    # ── Core reserve math ─────────────────────────────────────────────────────

    def _reserves(self, P: float):
        P_eff = max(self.P_a, min(P, self.P_b))
        x = self.L * (1.0 / math.sqrt(P_eff) - 1.0 / math.sqrt(self.P_b))
        y = self.L * (math.sqrt(P_eff) - math.sqrt(self.P_a))
        return x, y

    def value(self, P: float) -> float:
        """Position mark-to-market value (USD)."""
        x, y = self._reserves(P)
        return x * P + y

    def hodl_value(self, P: float) -> float:
        """Value of simply holding the entry amounts (USD)."""
        return self.x_0 * P + self.y_0

    # ── Risk metrics ───────────────────────────────────────────────────────────

    def il(self, P: float) -> float:
        """
        Impermanent loss as fraction of hodl value (always ≤ 0).
        IL = V_LP(P) / V_hodl(P) − 1
        """
        h = self.hodl_value(P)
        return (self.value(P) / h - 1.0) if h > 0 else 0.0

    def il_usd(self, P: float) -> float:
        """Impermanent loss in USD (negative = loss vs hodl)."""
        return self.value(P) - self.hodl_value(P)

    def in_range(self, P: float) -> bool:
        return self.P_a <= P <= self.P_b

    def delta(self, P: float) -> float:
        """
        dV/dP — directional exposure in base-asset units.
        In range: delta = L / (2√P)  (partial position in token0).
        Out of range (below): delta = full token0 position.
        Out of range (above): delta = 0 (all in stablecoin).
        """
        if P <= self.P_a:
            return self.L * (1.0 / math.sqrt(self.P_a) - 1.0 / math.sqrt(self.P_b))
        if P >= self.P_b:
            return 0.0
        return self.L / (2.0 * math.sqrt(P))

    def gamma(self, P: float) -> float:
        """
        d²V/dP² — convexity (always negative: short gamma).
        In range: Γ = −L / (4 P^{3/2}).
        Out of range: 0.
        """
        if not self.in_range(P):
            return 0.0
        return -self.L / (4.0 * P ** 1.5)

    # ── Approximations ─────────────────────────────────────────────────────────

    def il_approx(self, sigma_bar: float) -> float:
        """
        Expected IL per bar (Milionis 2022 approximation, in-range only):
            E[IL_bar] ≈ −Γ(P_0) / (2V_0) × σ̃² × P_0²
                      = L / (8 V_0 √P_0) × σ̃²
        """
        return -(self.L / (8.0 * self.V_0 * math.sqrt(self.P_0))) * sigma_bar ** 2


# ── Closed-form analytics ──────────────────────────────────────────────────────

def optimal_range_width(
    sigma_bar: float,
    fee_yield_per_bar: float,
) -> float:
    """
    Optimal log-price range half-width w* that maximises expected net PnL.

    Derived by equating marginal fee income and marginal IL:
        fee_yield / w  =  σ̃² / (4w²) × w   →  w* = σ̃ / √(2 × f_bar)

    Parameters
    ----------
    sigma_bar       : per-bar log-return std (same units as backtest σ̃)
    fee_yield_per_bar : fee income per bar per unit capital (= fee_tier × V/TVL / bars_per_day)
    """
    if fee_yield_per_bar <= 0:
        return 0.20   # fallback
    return sigma_bar / math.sqrt(2.0 * fee_yield_per_bar)


def il_full_range(r: float) -> float:
    """
    Exact Uniswap v2 IL for price ratio r = P_final / P_entry.
    IL = 2√r / (1+r) − 1
    """
    return 2.0 * math.sqrt(r) / (1.0 + r) - 1.0


def concentration_factor(w: float) -> float:
    """
    Capital efficiency multiplier vs full-range LP.
    A concentrated position with half-width w earns the same fees as a
    full-range position with c(w) × more capital.
    c(w) = 1 / (2(1 − e^{−w/2})) / √P_0   (in normalised units ≈ 1/(2w) for small w)
    """
    return 1.0 / (2.0 * (1.0 - math.exp(-w / 2.0)))
