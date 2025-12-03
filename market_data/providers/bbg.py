from .base import MarketDataProvider
from test_blp_api import bbg_get

class BloombergProvider(MarketDataProvider):
    """Appelle l'API Bloomberg."""
    
    def get_forward(self, ticker):
        return bbg_get(ticker, "PX_LAST")

    def get_rate(self, ticker):
        return bbg_get("USSWAP1 Curncy", "PX_LAST") / 100

    def get_dividend(self, ticker):
        return 0.00

    def get_sigma(self, ticker, tenor="1M"):
        return bbg_get(ticker, "IVOL_M1") / 100
