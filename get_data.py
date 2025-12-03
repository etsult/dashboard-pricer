from data_providers import DummyProvider, YahooProvider, BloombergProvider

providers = {
    "dummy": DummyProvider(),
    "yahoo": YahooProvider(),
    "bbg": BloombergProvider()
}

def get_market_params(ticker, source="dummy"):
    p = providers[source]

    return {
        "source": source,
        "F": p.get_forward(ticker),
        "r": p.get_rate(ticker),
        "q": p.get_dividend(ticker),
        "sigma": p.get_sigma(ticker),
    }
