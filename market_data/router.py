from typing import List
from market_data.schema import OptionQuote
from market_data.providers.yahoo import YahooProvider


class MarketRouter:

    def __init__(self, provider="yahoo"):
        if provider == "yahoo":
            self.provider = YahooProvider()
        else:
            raise ValueError("Unknown provider")

    def get_option_chain(self, ticker: str) -> List[OptionQuote]:
        return self.provider.get_option_chain(ticker)

    def get_forward(self, ticker: str) -> float:
        return self.provider.get_forward(ticker)
