from typing import Dict
from market_data.providers.bloomberg import BloombergProvider
# from market_data.providers.yahoo_provider import YahooProvider

class MarketDataRouter:
    def __init__(self, provider="bbg"):
        if provider == "bbg":
            self.provider = BloombergProvider()
        elif provider == "yahoo":
            raise NotImplementedError("Yahoo provider coming soon")
        else:
            raise ValueError("Unknown provider")

    def get_quotes(self, strategy):
        """
        strategy.legs = [(opt, qty, label)]
        returns list[ dict : market data ]
        """
        return self.provider.get_option_quotes(strategy.legs)
