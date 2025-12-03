from .base import MarketDataProvider

class DummyProvider(MarketDataProvider):
    """Valeurs fixes pour mode éducatif."""

    def get_forward(self, ticker):
        return 100

    def get_rate(self, ticker):
        return 0.02

    def get_dividend(self, ticker):
        return 0.00

    def get_sigma(self, ticker, tenor="1M"):
        return 0.20
