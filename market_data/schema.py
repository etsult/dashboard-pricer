# market_data/schema.py
from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class OptionQuote:
    ticker: str
    call_put: str       # "C" ou "P"
    strike: float
    expiry: date

    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]

    delta: Optional[float]
    implied_vol: Optional[float]
    volume: Optional[int]
