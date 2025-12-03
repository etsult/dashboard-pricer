import yfinance as yf
from .base import MarketDataProvider

class YahooProvider(MarketDataProvider):

    def get_forward(self, ticker):
        data = yf.Ticker(ticker).history(period="1d")
        return float(data["Close"].iloc[-1])

    def get_rate(self, ticker):
        return 0.02  # tu peux ajouter un vrai rate si tu veux

    def get_dividend(self, ticker):
        return 0.00

    def get_sigma(self, ticker, tenor="1M"):
        return 0.20  # TODO : récupérer la vol réelle
