import yfinance as yf
from datetime import datetime
from typing import List
import numpy as np

from market_data.schema import OptionQuote
from market_data.providers.base import MarketDataProvider


class YahooProvider(MarketDataProvider):

    def get_forward(self, ticker: str) -> float:
        data = yf.Ticker(ticker).history(period="1d")
        return float(data["Close"].iloc[-1])

    def get_option_chain(self, ticker: str) -> List[OptionQuote]:
        yf_ticker = yf.Ticker(ticker)
        expiries = yf_ticker.options
        results: List[OptionQuote] = []

        for expiry_str in expiries:
            expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d")

            chain = yf_ticker.option_chain(expiry_str)
            calls, puts = chain.calls, chain.puts

            # ---------------- CALLS ----------------
            for _, row in calls.iterrows():
                bid = float(row["bid"]) if not np.isnan(row["bid"]) else None
                ask = float(row["ask"]) if not np.isnan(row["ask"]) else None
                mid = 0.5 * (bid + ask) if bid and ask else None

                iv = row.get("impliedVolatility", None)
                iv = float(iv) if iv == iv else None

                results.append(
                    OptionQuote(
                        ticker=ticker,
                        expiry=expiry_dt,
                        strike=float(row["strike"]),
                        option_type="call",
                        bid=bid,
                        ask=ask,
                        mid=mid,
                        volume=int(row["volume"]) if row["volume"] == row["volume"] else None,
                        implied_vol=iv,   # CHANGED HERE
                        delta=None,
                    )
                )

            # ---------------- PUTS ----------------
            for _, row in puts.iterrows():
                bid = float(row["bid"]) if not np.isnan(row["bid"]) else None
                ask = float(row["ask"]) if not np.isnan(row["ask"]) else None
                mid = 0.5 * (bid + ask) if bid and ask else None

                iv = row.get("impliedVolatility", None)
                iv = float(iv) if iv == iv else None

                results.append(
                    OptionQuote(
                        ticker=ticker,
                        expiry=expiry_dt,
                        strike=float(row["strike"]),
                        option_type="put",
                        bid=bid,
                        ask=ask,
                        mid=mid,
                        volume=int(row["volume"]) if row["volume"] == row["volume"] else None,
                        implied_vol=iv,   # CHANGED HERE
                        delta=None,
                    )
                )

        return results
