"""
WebSocket router — real-time streaming endpoints.

Usage from the React frontend:
  const ws = new WebSocket('ws://localhost:5173/ws/live-monitor?currency=BTC&interval=30')
  ws.onmessage = e => console.log(JSON.parse(e.data))

The Vite dev server proxies /ws/* → ws://localhost:8000/ws/*, so the frontend
never needs to know the backend port.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.schemas.market_data import VolTermStructureResponse, VolTermStructurePoint
from market_data.providers.deribit import DeribitProvider
from market_data.curves.vol_term_structure import build_term_structure

router = APIRouter(prefix="/ws", tags=["WebSocket"])


def _fetch_term_structure(
    currency: str, rate: float
) -> VolTermStructureResponse:
    """Synchronous fetch — called inside asyncio.to_thread to avoid blocking the event loop."""
    provider = DeribitProvider()
    spot = provider.get_forward(currency)
    quotes = provider.get_option_chain(currency)
    fetched_at = datetime.datetime.utcnow().strftime("%H:%M:%S UTC")

    ts = build_term_structure(quotes, spot=spot, rate=rate)

    points = [
        VolTermStructurePoint(
            tenor_label=row["tenor_label"],
            days=round(row["days"], 1),
            atm_iv_pct=round(row["atm_iv"] * 100, 2),
            fwd_vol_pct=round(row["fwd_vol"] * 100, 2)
            if row["fwd_vol"] is not None and str(row["fwd_vol"]) != "nan"
            else None,
            fwd_label=row.get("fwd_label"),
            total_var=round(row["total_var"], 6),
            is_calendar_arb=bool(row["is_calendar_arb"]),
        )
        for _, row in ts.iterrows()
    ]

    return VolTermStructureResponse(
        currency=currency,
        spot=round(spot, 2),
        fetched_at=fetched_at,
        n_quotes=len(quotes),
        term_structure=points,
        n_arb_violations=int(ts["is_calendar_arb"].sum()),
    )


@router.websocket("/live-monitor")
async def live_monitor(
    websocket: WebSocket,
    currency: Literal["BTC", "ETH", "SOL"] = "BTC",
    rate: float = 0.05,
    interval: int = 30,
) -> None:
    """
    Stream vol term structure updates every `interval` seconds.

    Query params:
      currency  — BTC | ETH | SOL  (default BTC)
      rate      — funding rate      (default 0.05)
      interval  — seconds between   (default 30, min 5)

    Each message is a JSON object matching VolTermStructureResponse.
    On error, sends {"error": "<message>"} instead.
    """
    await websocket.accept()

    # Clamp interval: never faster than 5 s to respect Deribit rate limits
    interval = max(5, interval)

    try:
        while True:
            try:
                data = await asyncio.to_thread(_fetch_term_structure, currency, rate)
                await websocket.send_text(data.model_dump_json())
            except Exception as exc:
                await websocket.send_json({"error": str(exc)})

            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        pass  # client closed — clean exit
