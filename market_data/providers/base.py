class MarketDataProvider:
    """Interface pour toutes les sources de marché."""

    def get_forward(self, ticker):
        raise NotImplementedError

    def get_rate(self, ticker):
        raise NotImplementedError

    def get_dividend(self, ticker):
        raise NotImplementedError

    def get_sigma(self, ticker, tenor="1M"):
        raise NotImplementedError
