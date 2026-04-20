# market_data/providers/bbg_provider.py

from __future__ import annotations
import blpapi
from datetime import datetime
from typing import List, Dict, Any

SESSION_OPTIONS = blpapi.SessionOptions()
SESSION_OPTIONS.setServerHost("localhost")
SESSION_OPTIONS.setServerPort(8194)


class BloombergProvider:
    """
    Fetch option quotes from Bloomberg OMON.
    This provider ONLY retrieves market data.
    Strategy quantities are handled downstream in valuation.
    """

    FIELDS = [
        "BID",
        "ASK",
        "BID_SIZE",
        "ASK_SIZE",
        "VOLUME",
        "DELTA",
        "IMPLIED_VOLATILITY_MID",
    ]

    def __init__(self):
        self.session = blpapi.Session(SESSION_OPTIONS)
        if not self.session.start():
            raise RuntimeError("Could not start Bloomberg session")
        if not self.session.openService("//blp/refdata"):
            raise RuntimeError("Could not open Bloomberg refdata service")

        self.refdata = self.session.getService("//blp/refdata")

    # ------------------------------------------------------------
    def _option_ticker(self, opt) -> str:
        """
        Example:
        SX5E 03/20/26 C6000 Index
        """
        expiry = opt.expiry.strftime("%m/%d/%y")
        CP = "C" if opt.option_type == "call" else "P"
        ticker = f"{opt.underlying} {expiry} {CP}{int(opt.K)} Index"
        return ticker

    # ------------------------------------------------------------
    def get_option_quotes(self, legs) -> List[Dict[str, Any]]:
        """
        legs = [(option_object, qty, label), ...]
        """
        tickers = [self._option_ticker(opt) for opt, _, _ in legs]

        request = self.refdata.createRequest("ReferenceDataRequest")
        for t in tickers:
            request.append("securities", t)

        for f in self.FIELDS:
            request.append("fields", f)

        self.session.sendRequest(request)

        results = []
        i = 0

        while True:
            ev = self.session.nextEvent()
            for msg in ev:
                if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                    data = msg["securityData"]
                    for sec in data.values():
                        f = sec["fieldData"]

                        # assign fields safely
                        def g(field):
                            return f.get(field, None)

                        quotes = {
                            "Ticker": tickers[i],
                            "Bid": g("BID"),
                            "BidSize": g("BID_SIZE"),
                            "Mid": None if g("BID") is None or g("ASK") is None else (g("BID") + g("ASK")) / 2,
                            "Ask": g("ASK"),
                            "AskSize": g("ASK_SIZE"),
                            "Volume": g("VOLUME"),
                            "Delta": g("DELTA"),
                            "IVMid": g("IMPLIED_VOLATILITY_MID"),
                        }
                        results.append(quotes)
                        i += 1

            if ev.eventType() == blpapi.Event.RESPONSE:
                break

        return results
