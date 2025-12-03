# get_data.py
import pandas as pd
import random

try:
    import blpapi
    from blpapi import SessionOptions, Session
    BLOOMBERG_AVAILABLE = True
except ImportError:
    BLOOMBERG_AVAILABLE = False


def fetch_from_bloomberg(ticker, fields):
    """Try fetching data from Bloomberg, return dict or raise RuntimeError."""
    options = SessionOptions()
    options.setServerHost("localhost")
    options.setServerPort(8194)
    session = Session(options)

    if not session.start():
        raise RuntimeError("Cannot start Bloomberg session.")
    if not session.openService("//blp/refdata"):
        raise RuntimeError("Cannot open refdata service.")

    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")
    request.getElement("securities").appendValue(ticker)
    for f in fields:
        request.getElement("fields").appendValue(f)
    session.sendRequest(request)

    data = {}
    while True:
        ev = session.nextEvent()
        for msg in ev:
            if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                fd = msg.getElement("securityData").getValue(0).getElement("fieldData")
                for f in fields:
                    if fd.hasElement(f):
                        data[f] = fd.getElementAsString(f)
        if ev.eventType() == blpapi.Event.RESPONSE:
            break
    session.stop()
    return data


def get_market_params(ticker):
    """Return dict of useful parameters for an equity or index."""
    fields = ["PX_LAST", "EQY_DVD_YLD_EST"]

    if BLOOMBERG_AVAILABLE:
        try:
            raw = fetch_from_bloomberg(ticker, fields)
            return {
                "source": "bloomberg",
                "spot": float(raw.get("PX_LAST", 0)),
                "dividend_yield": float(raw.get("EQY_DVD_YLD_EST", 0)) / 100,
                "risk_free_rate": 0.02,  # Default static curve, can be refined later
            }
        except Exception as e:
            print(f"⚠️ Bloomberg unavailable: {e}")

    # fallback mock data
    print("Using fallback mock data.")
    return {
        "source": "manual",
        "spot": 5500.0 + random.uniform(-100, 100),
        "dividend_yield": 0.015,
        "risk_free_rate": 0.02,
    }
