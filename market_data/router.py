from typing import List
from market_data.schema import OptionQuote
from market_data.providers.yahoo import YahooProvider
from market_data.providers.deribit import DeribitProvider


class MarketRouter:

    def __init__(self, provider: str = "yahoo"):
        if provider == "yahoo":
            self.provider = YahooProvider()
        elif provider == "deribit":
            self.provider = DeribitProvider()
        else:
            raise ValueError(f"Unknown provider: {provider!r}. Choose 'yahoo' or 'deribit'.")

    def get_option_chain(self, ticker: str) -> List[OptionQuote]:
        return self.provider.get_option_chain(ticker)

    def get_forward(self, ticker: str) -> float:
        return self.provider.get_forward(ticker)
