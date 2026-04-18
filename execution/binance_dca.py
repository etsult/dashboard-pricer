"""
Binance DCA executor.

Uses ccxt — a unified exchange library that works with 100+ exchanges.
If you ever want to switch to Kraken, Coinbase, or OKX, you change
one line (the exchange class) and the rest stays the same.

Safety defaults:
  - testnet=True by default — you must explicitly opt in to live trading
  - Only places market buys (no limit, no stop, no leverage)
  - Logs every attempt to the database whether it succeeds or fails
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Literal

import ccxt

log = logging.getLogger(__name__)


class BinanceDCAExecutor:

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
    ) -> None:
        self.testnet = testnet

        self.exchange = ccxt.binance({
            "apiKey":          api_key,
            "secret":          api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
            },
        })

        if testnet:
            # Binance spot testnet — real API, fake money
            self.exchange.set_sandbox_mode(True)
            log.info("BinanceDCAExecutor initialised in TESTNET mode")
        else:
            log.warning("BinanceDCAExecutor initialised in LIVE mode — real money")

    # ── Connectivity ──────────────────────────────────────────────────

    def check_connection(self) -> dict:
        """Verify API keys work and return account balances."""
        balance = self.exchange.fetch_balance()
        return {
            "status":  "ok",
            "testnet": self.testnet,
            "balances": {
                k: v for k, v in balance["total"].items()
                if v and v > 0
            },
        }

    # ── Market info ───────────────────────────────────────────────────

    def get_ticker(self, symbol: str = "BTC/EUR") -> dict:
        """Fetch current price and 24h stats for a symbol."""
        ticker = self.exchange.fetch_ticker(symbol)
        return {
            "symbol":    ticker["symbol"],
            "last":      ticker["last"],
            "bid":       ticker["bid"],
            "ask":       ticker["ask"],
            "volume_24h": ticker["baseVolume"],
        }

    def get_min_order(self, symbol: str = "BTC/EUR") -> dict:
        """Return minimum order size for a symbol (important for small DCA amounts)."""
        markets = self.exchange.load_markets()
        if symbol not in markets:
            raise ValueError(f"Symbol {symbol!r} not found on this exchange/testnet")
        m = markets[symbol]
        return {
            "symbol":         symbol,
            "min_amount_btc": m["limits"]["amount"]["min"],
            "min_cost_eur":   m["limits"]["cost"]["min"] if m["limits"].get("cost") else None,
            "price_precision": m["precision"]["price"],
            "amount_precision": m["precision"]["amount"],
        }

    # ── Order placement ───────────────────────────────────────────────

    def buy_quote_amount(
        self,
        symbol: str = "BTC/EUR",
        quote_amount: float = 100.0,
    ) -> dict:
        """
        Buy exactly `quote_amount` EUR worth of BTC at market price.

        This is the key method for DCA — you specify how much EUR to spend,
        Binance calculates how much BTC you receive.

        Binance calls this a "reverse market order" (quoteOrderQty).
        ccxt exposes it via create_market_buy_order_with_cost().
        """
        if quote_amount <= 0:
            raise ValueError(f"quote_amount must be positive, got {quote_amount}")

        log.info(
            "Placing market buy: %s %.2f EUR [testnet=%s]",
            symbol, quote_amount, self.testnet,
        )

        try:
            order = self.exchange.create_market_buy_order_with_cost(
                symbol=symbol,
                cost=quote_amount,
            )
        except ccxt.InsufficientFunds as e:
            raise RuntimeError(f"Insufficient funds: {e}") from e
        except ccxt.InvalidOrder as e:
            raise RuntimeError(f"Invalid order: {e}") from e
        except ccxt.NetworkError as e:
            raise RuntimeError(f"Network error (will retry next cycle): {e}") from e

        return {
            "order_id":     order["id"],
            "symbol":       order["symbol"],
            "side":         order["side"],
            "status":       order["status"],
            "cost_eur":     order.get("cost"),          # EUR spent
            "filled_btc":   order.get("filled"),        # BTC received
            "avg_price":    order.get("average"),       # effective price
            "fee":          order.get("fee"),
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "testnet":      self.testnet,
        }


def executor_from_env(testnet: bool = True) -> BinanceDCAExecutor:
    """
    Build an executor from environment variables.

    Supports Ed25519 / RSA asymmetric keys (recommended by Binance):
      BINANCE_API_KEY          — the key ID shown on Binance API Management page
      BINANCE_PRIVATE_KEY_PATH — path to your local PEM private key file

    The public key was uploaded to Binance; the private key never leaves your machine.
    """
    api_key   = os.environ.get("BINANCE_API_KEY", "")
    key_path  = os.environ.get("BINANCE_PRIVATE_KEY_PATH", "")

    if not api_key:
        raise RuntimeError("BINANCE_API_KEY must be set in .env")
    if not key_path:
        raise RuntimeError(
            "BINANCE_PRIVATE_KEY_PATH must be set in .env\n"
            "Generate with: openssl genpkey -algorithm ed25519 -out ~/.ssh/binance_private.pem"
        )

    key_path = os.path.expanduser(key_path)
    if not os.path.exists(key_path):
        raise RuntimeError(f"Private key file not found: {key_path}")

    with open(key_path) as f:
        private_key = f.read()

    return BinanceDCAExecutor(api_key=api_key, api_secret=private_key, testnet=testnet)
