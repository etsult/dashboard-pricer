import logging
import re
import threading
import time
from datetime import datetime
from queue import Queue, Empty
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

from market_data.providers.base import MarketDataProvider
from market_data.schema import OptionQuote

LOG = logging.getLogger(__name__)

try:
    import blpapi
except Exception:
    blpapi = None

# lazy import xbbg at runtime to surface import errors in the calling process
try:
    from xbbg import blp as xbbg_blp  # type: ignore
except Exception:
    xbbg_blp = None  # will handle later


class BloombergProvider(MarketDataProvider):
    """
    Bloomberg provider with:
      - get_forward(ticker)
      - get_option_chain(ticker)
      - live_stream(tickers, fields, interval) -> generator of DataFrame snapshots (polling fallback)
      - start_background_poll(tickers, fields, interval) -> Queue of updates (useful for Streamlit)
      - run_strategies_live(strategies, feed_queue, publish_cb) to apply strategy pricers to live market data

    Notes:
      - Prefers xbbg for convenience (bds/bdp). If xbbg not available, some methods will raise.
      - For robust live behavior we use two modes:
         1) If xbbg.subscribe exists, use it (push mode).
         2) Otherwise use polling via bdp in a background thread.
    """

    def __init__(self, host: str = "localhost", port: int = 8194, timeout_ms: int = 2000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._session = None
        if blpapi is not None:
            self._start_session()
        if xbbg_blp is None:
            LOG.warning("xbbg.blp not available -- install xbbg for convenience methods (bds/bdp/subscribe).")

    def _start_session(self):
        if self._session is not None and self._session.isStarted():
            return
        opts = blpapi.SessionOptions()
        opts.setServerHost(self.host)
        opts.setServerPort(self.port)
        sess = blpapi.Session(opts)
        if not sess.start():
            raise RuntimeError("Failed to start Bloomberg session")
        if not sess.openService("//blp/refdata"):
            sess.stop()
            raise RuntimeError("Failed to open //blp/refdata")
        self._session = sess

    # -----------------------------
    # Basic data
    # -----------------------------
    def get_forward(self, ticker: str) -> float:
        if xbbg_blp is None:
            raise RuntimeError("xbbg is required for get_forward() in this implementation")
        df = xbbg_blp.bdp(ticker, "PX_LAST")
        return float(df.iloc[0]["px_last"])

    # -----------------------------
    # Option chain + option details
    # -----------------------------
    def get_option_chain(self, ticker: str) -> List[OptionQuote]:
        if xbbg_blp is None:
            raise RuntimeError("xbbg is required for get_option_chain()")
        chain_df = xbbg_blp.bds(ticker, "OPT_CHAIN")
        if chain_df is None or chain_df.empty:
            return []

        option_details = []
        for sec_desc in chain_df["security_description"]:
            m = re.search(r"(\d{2}/\d{2}/\d{2})\s+([CP])(\d+)", sec_desc)
            if not m:
                continue
            expiry_s, cp_flag, strike_s = m.groups()
            try:
                expiry = datetime.strptime(expiry_s, "%m/%d/%y").date()
                strike = float(strike_s)
            except Exception:
                continue
            option_details.append({"sec": sec_desc, "expiry": expiry, "strike": strike, "type": "call" if cp_flag == "C" else "put"})

        # batch fields to retrieve for each option
        fields = [
            "PX_BID", "PX_ASK", "BID_SIZE", "ASK_SIZE",
            "IVOL_MID_RT", "OPT_DELTA", "OPT_GAMMA", "OPT_VEGA", "OPT_THETA",
            "VOLUME", "OPEN_INT", "PX_LAST", "MID"
        ]

        quotes: List[OptionQuote] = []
        # fetch in batches to avoid enormous requests (simple sequential for clarity)
        for od in option_details:
            try:
                df = xbbg_blp.bdp(od["sec"], fields)
                row = df.iloc[0]
                q = OptionQuote(
                    ticker=od["sec"],
                    expiry=od["expiry"],
                    strike=od["strike"],
                    option_type=od["type"],
                    bid=float(row.get("px_bid")) if pd.notna(row.get("px_bid")) else None,
                    ask=float(row.get("px_ask")) if pd.notna(row.get("px_ask")) else None,
                    mid=float(row.get("mid")) if pd.notna(row.get("mid")) else None,
                    implied_vol=float(row.get("ivol_mid_rt")) if pd.notna(row.get("ivol_mid_rt")) else None,
                    volume=int(row.get("volume")) if pd.notna(row.get("volume")) else None,
                )
                # attach greeks/sizes if OptionQuote supports them (best-effort)
                setattr(q, "delta", float(row.get("opt_delta")) if pd.notna(row.get("opt_delta")) else None)
                setattr(q, "gamma", float(row.get("opt_gamma")) if pd.notna(row.get("opt_gamma")) else None)
                setattr(q, "vega", float(row.get("opt_vega")) if pd.notna(row.get("opt_vega")) else None)
                setattr(q, "theta", float(row.get("opt_theta")) if pd.notna(row.get("opt_theta")) else None)
                setattr(q, "bid_size", int(row.get("bid_size")) if pd.notna(row.get("bid_size")) else None)
                setattr(q, "ask_size", int(row.get("ask_size")) if pd.notna(row.get("ask_size")) else None)
                setattr(q, "open_interest", int(row.get("open_int")) if pd.notna(row.get("open_int")) else None)
                quotes.append(q)
            except Exception:
                LOG.debug("Failed fetching option details for %s", od["sec"], exc_info=True)
                continue

        return quotes

    # -----------------------------
    # Live streaming / polling support
    # -----------------------------
    def live_stream(self, tickers: Iterable[str], fields: Iterable[str], interval: float = 2.0):
        """
        Generator that yields DataFrame snapshots for the requested tickers+fields.
        Prefers push subscribe if available, otherwise falls back to polling (bdp).
        Use in Streamlit with st.empty() and a loop to update UI.
        """
        # try push subscription if available
        if xbbg_blp is not None and hasattr(xbbg_blp, "subscribe"):
            updates_q: Queue = Queue()

            def _on_msg(msg):
                # xbbg.subscribe callback receives (msg) — shape depends on xbbg version
                try:
                    updates_q.put(msg)
                except Exception:
                    LOG.exception("subscribe callback failed")

            sub = None
            try:
                sub = xbbg_blp.subscribe(list(tickers), list(fields), callback=_on_msg)
            except Exception:
                LOG.exception("xbbg.subscribe failed, falling back to polling")

            if sub is not None:
                try:
                    while True:
                        try:
                            msg = updates_q.get(timeout=interval)
                            # yield raw message DataFrame/Dict for consumer to interpret
                            yield msg
                        except Empty:
                            # heartbeat: still yield latest snapshot via bdp for consistency
                            try:
                                df = xbbg_blp.bdp(list(tickers), list(fields))
                                yield df
                            except Exception:
                                yield None
                finally:
                    try:
                        sub.close()
                    except Exception:
                        pass

        # fallback: polling using bdp
        if xbbg_blp is None:
            raise RuntimeError("xbbg is required for live_stream fallback polling mode")

        while True:
            try:
                df = xbbg_blp.bdp(list(tickers), list(fields))
                # normalize colnames to lower case for convenience
                df.columns = [c.lower() for c in df.columns]
                df["_timestamp"] = datetime.now()
                yield df
            except Exception:
                LOG.exception("Polling live_stream bdp failed")
                yield None
            time.sleep(interval)

    def start_background_poll(self, tickers: Iterable[str], fields: Iterable[str], interval: float = 2.0) -> Queue:
        """
        Start a background thread polling bdp(tickers, fields) every interval seconds.
        Returns a Queue where DataFrame snapshots are pushed. Caller should consume and empty queue.
        """
        q: Queue = Queue()

        def _poll_loop():
            LOG.info("Background poll started for %d tickers", len(list(tickers)))
            while True:
                try:
                    df = xbbg_blp.bdp(list(tickers), list(fields))
                    df.columns = [c.lower() for c in df.columns]
                    df["_timestamp"] = datetime.now()
                    q.put(df)
                except Exception:
                    LOG.exception("Background poll failed")
                    q.put(None)
                time.sleep(interval)

        t = threading.Thread(target=_poll_loop, daemon=True)
        t.start()
        return q

    # -----------------------------
    # Strategy runner
    # -----------------------------
    def run_strategies_live(
        self,
        strategies: Iterable[Callable[[Dict[str, pd.Series]], Dict]],
        feed_queue: Queue,
        publish_cb: Callable[[Dict], None],
        stop_event: Optional[threading.Event] = None,
    ):
        """
        Consume feed_queue (DataFrame snapshots) and run strategy callables.
        Each strategy is a callable that receives market_data: Dict[ticker -> pd.Series/row]
        and returns a dict of metrics to publish. publish_cb is called with the combined results.
        This method runs in current thread and returns when stop_event is set or queue is exhausted.
        """
        if stop_event is None:
            stop_event = threading.Event()

        while not stop_event.is_set():
            try:
                df = feed_queue.get(timeout=1.0)
            except Empty:
                continue
            if df is None:
                continue
            try:
                # convert snapshot to dict ticker->row Series
                snapshot = {t: df.loc[t] if t in df.index else None for t in df.index.unique()}
                results = {}
                for s in strategies:
                    try:
                        res = s(snapshot)
                        results.update(res if isinstance(res, dict) else {s.__name__: res})
                    except Exception:
                        LOG.exception("Strategy %s failed", getattr(s, "__name__", str(s)))
                # include timestamp
                results["_ts"] = df.attrs.get("_timestamp") if "_timestamp" in df.attrs else datetime.now()
                publish_cb(results)
            except Exception:
                LOG.exception("run_strategies_live iteration failed")
                continue
