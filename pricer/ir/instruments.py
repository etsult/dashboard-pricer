"""
IR option position and book dataclasses.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class IRPosition:
    """Single IR option position in the book."""

    instrument: Literal["cap", "floor", "payer_swaption", "receiver_swaption"]
    index_key: str           # key in INDEX_CATALOG
    notional: float          # in currency units
    strike: float            # decimal (e.g. 0.04 = 4%)
    expiry_y: float          # years to expiry / option expiry
    tenor_y: float           # cap maturity or swap tenor (years)
    sigma_n: float           # ATM normal vol (decimal, e.g. 0.006 = 60 bps/yr)
    direction: int = 1       # +1 long, -1 short

    # Convention offsets (in years)
    start_y: float = 0.0    # settlement delay (e.g. 2/365 for T+2)
    reset_lag_y: float = 0.0
    pay_lag_y: float = 0.0

    # Optional label for display
    label: str = ""

    def __post_init__(self):
        if not self.label:
            side = "Long" if self.direction > 0 else "Short"
            instr_map = {
                "cap": "Cap", "floor": "Floor",
                "payer_swaption": "Payer Swptn", "receiver_swaption": "Rcvr Swptn",
            }
            self.label = (
                f"{side} {instr_map.get(self.instrument, self.instrument)} "
                f"{self.expiry_y:.1f}Y×{self.tenor_y:.1f}Y "
                f"K={self.strike*100:.2f}% "
                f"σ={self.sigma_n*10000:.0f}bps"
            )


@dataclass
class Book:
    """Portfolio of IR option positions."""
    positions: list[IRPosition] = field(default_factory=list)

    def add(self, pos: IRPosition) -> None:
        self.positions.append(pos)

    def remove(self, idx: int) -> None:
        self.positions.pop(idx)

    def __len__(self) -> int:
        return len(self.positions)

    def is_empty(self) -> bool:
        return len(self.positions) == 0
