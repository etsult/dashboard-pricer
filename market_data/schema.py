from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class OptionQuote:
    ticker: str
    expiry: datetime
    strike: float
    option_type: str  # "call" or "put"

    bid: float
    ask: float
    mid: float
    volume: Optional[int]

    implied_vol: Optional[float]
    delta: Optional[float]  # Yahoo does NOT provide this, we compute later
