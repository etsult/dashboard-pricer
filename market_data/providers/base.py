# market_data/providers/base.py

from abc import ABC, abstractmethod
from typing import List
from market_data.schema import OptionQuote


class MarketDataProvider(ABC):
    """
    Abstract base class for all market data providers (Yahoo, Bloomberg, CSV, etc.)

    Every provider must implement the following:
    - get_forward(ticker): return forward or spot price
    - get_option_chain(ticker): return list of OptionQuote for ALL expiries
    """

    # -----------------------------
    # Forward / Spot
    # -----------------------------
    @abstractmethod
    def get_forward(self, ticker: str) -> float:
        """
        Returns the forward (or spot) for the underlying ticker.
        Yahoo returns the spot.
        Bloomberg will return the future price or spot based on field PX_LAST.
        """
        pass

    # -----------------------------
    # Option Chain
    # -----------------------------
    @abstractmethod
    def get_option_chain(self, ticker: str) -> List[OptionQuote]:
        """
        Returns ALL option expiries and strikes for the underlying ticker.

        The returned structure is a flat list of OptionQuote objects:
            [
                OptionQuote(ticker="SPX", expiry=..., strike=..., option_type="call", ...),
                OptionQuote(...),
                ...
            ]

        The provider is responsible for:
        - mapping tickers correctly
        - retrieving expiries
        - converting timestamps to datetime
        - normalizing bid/ask/mid fields
        - including implied vol if available
        - including volume if available
        """
        pass

    # ---------------------------------------------------------------------
    # OPTIONAL EXTENSIONS (implemented only by Bloomberg or custom sources)
    # ---------------------------------------------------------------------

    def get_rate(self, currency: str = "USD") -> float:
        """
        Optional method: returns risk-free rate for currency.
        Yahoo does NOT support this → default constant.
        Bloomberg supports this via FIELD: "PX_LAST" on curve tickers.
        """
        return 0.02  # fallback value

    def get_dividend_yield(self, ticker: str) -> float:
        """
        Optional method: returns dividend yield.
        Yahoo provides dividends but not forward-implied dividend.
        Bloomberg provides both.
        """
        return 0.00
