# market_data/providers/deribit.py
"""
Fetches crypto option chains from Deribit via their public REST API.
No API key required for market data.

Supported underlyings: BTC, ETH, SOL, XRP, MATIC, ...
"""

from __future__ import annotations
import requests
from datetime import datetime
from typing import List, Optional, Tuple

from market_data.schema import OptionQuote
from market_data.providers.base import MarketDataProvider

_BASE = "https://www.deribit.com/api/v2/public"

# Deribit index names for spot price lookups
_INDEX_MAP = {
    "BTC":   "btc_usd",
    "ETH":   "eth_usd",
    "SOL":   "sol_usd",
    "XRP":   "xrp_usd",
    "MATIC": "matic_usd",
}


class DeribitProvider(MarketDataProvider):
    """
    Deribit public API provider.

    Usage:
        provider = DeribitProvider()
        spot     = provider.get_forward("BTC")
        quotes   = provider.get_option_chain("BTC")
    """

    def get_forward(self, ticker: str) -> float:
        """Returns Deribit's index price (spot) for the given crypto."""
        currency   = ticker.upper().split("-")[0]  # handle "BTC" or "BTC-PERP"
        index_name = _INDEX_MAP.get(currency, f"{currency.lower()}_usd")
        resp = requests.get(
            f"{_BASE}/get_index_price",
            params={"index_name": index_name},
            timeout=10,
        )
        resp.raise_for_status()
        return float(resp.json()["result"]["index_price"])

    def get_option_chain(self, ticker: str) -> List[OptionQuote]:
        """
        Returns all live option quotes for a crypto underlying.

        Uses get_book_summary_by_currency (one request for all options).
        Prices are returned in USD (converted from BTC-denominated quotes).
        """
        currency = ticker.upper().split("-")[0]
        resp = requests.get(
            f"{_BASE}/get_book_summary_by_currency",
            params={"currency": currency, "kind": "option"},
            timeout=30,
        )
        resp.raise_for_status()
        summaries = resp.json().get("result", [])

        # Try to get spot from response to avoid a second request
        spot = None
        for item in summaries:
            underlying = item.get("underlying_price") or item.get("index_price")
            if underlying:
                spot = float(underlying)
                break
        if spot is None or spot == 0:
            try:
                spot = self.get_forward(currency)
            except Exception:
                spot = 1.0

        quotes: List[OptionQuote] = []
        for item in summaries:
            try:
                name                  = item["instrument_name"]
                expiry, strike, opt_t = _parse_instrument(name)
                if expiry is None:
                    continue

                # Prices on Deribit are in units of underlying (BTC), convert to USD
                underlying = float(item.get("underlying_price") or item.get("index_price") or spot)

                bid = _safe_float(item.get("bid_price"))
                ask = _safe_float(item.get("ask_price"))
                mid = _safe_float(item.get("mid_price"))

                if bid is not None:
                    bid *= underlying
                if ask is not None:
                    ask *= underlying
                if mid is not None:
                    mid *= underlying
                elif bid is not None and ask is not None:
                    mid = 0.5 * (bid + ask)

                # mark_iv is in percent on Deribit
                iv_raw = item.get("mark_iv")
                iv = float(iv_raw) / 100.0 if iv_raw else None
                if iv is not None and (iv <= 0 or iv > 20.0):  # sanity
                    iv = None

                volume = _safe_int(item.get("volume"))

                quotes.append(OptionQuote(
                    ticker=currency,
                    expiry=expiry,
                    strike=float(strike),
                    option_type=opt_t,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    volume=volume,
                    implied_vol=iv,
                    delta=_safe_float(item.get("greeks", {}).get("delta") if item.get("greeks") else None),
                ))
            except Exception:
                continue

        return quotes


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_instrument(name: str) -> Tuple[Optional[datetime], Optional[float], Optional[str]]:
    """
    Parse Deribit instrument name.
    Format: 'BTC-28MAR25-100000-C'
    Returns (expiry_datetime, strike_float, 'call'|'put') or (None, None, None).
    """
    try:
        parts = name.split("-")
        if len(parts) != 4:
            return None, None, None
        _, expiry_str, strike_str, type_char = parts
        expiry   = datetime.strptime(expiry_str, "%d%b%y")
        strike   = float(strike_str)
        opt_type = "call" if type_char.upper() == "C" else "put"
        return expiry, strike, opt_type
    except Exception:
        return None, None, None


def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return f if f == f else None  # reject NaN
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> Optional[int]:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None
