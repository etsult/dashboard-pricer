import yfinance as yf
from datetime import datetime
from typing import List
import numpy as np
import pandas as pd
from pathlib import Path

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
        raw_rows = []

        for expiry_str in expiries:
            expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d")

            chain = yf_ticker.option_chain(expiry_str)
            calls, puts = chain.calls, chain.puts

            print(f"Expiry {expiry_str}:")
            print("  Calls strikes:", calls["strike"].min(), "→", calls["strike"].max())
            print("  Puts strikes :", puts["strike"].min(), "→", puts["strike"].max())

            # ---------------- CALLS ----------------
            for _, row in calls.iterrows():
                bid = float(row["bid"]) if row["bid"] == row["bid"] else None
                ask = float(row["ask"]) if row["ask"] == row["ask"] else None
                mid = 0.5 * (bid + ask) if (bid is not None and ask is not None) else None

                iv = row.get("impliedVolatility", None)
                iv = float(iv) if iv == iv else None   # FIXED scaling

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
                        implied_vol=iv,
                        delta=None,
                    )
                )

                raw_rows.append(
                    {
                        "expiry": expiry_dt,
                        "option_type": "call",
                        **row.to_dict(),
                    }
                )

            # ---------------- PUTS ----------------
            for _, row in puts.iterrows():
                bid = float(row["bid"]) if row["bid"] == row["bid"] else None
                ask = float(row["ask"]) if row["ask"] == row["ask"] else None
                mid = 0.5 * (bid + ask) if (bid is not None and ask is not None) else None

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
                        implied_vol=iv,
                        delta=None,
                    )
                )

                raw_rows.append(
                    {
                        "expiry": expiry_dt,
                        "option_type": "put",
                        **row.to_dict(),
                    }
                )

        # Persist raw yahoo chain for debugging before any filtering
        try:
            storage_dir = Path(__file__).resolve().parents[2] / "storage"
            storage_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(raw_rows).to_csv(storage_dir / "last_yahoo_chain_raw.csv", index=False)
        except OSError as exc:
            print(f"[yahoo] ⚠️ could not write raw option chain CSV: {exc}")

        return results
