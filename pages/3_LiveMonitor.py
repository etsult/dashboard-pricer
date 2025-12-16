import re
import threading
import time
import pkgutil
import importlib
import uuid
from queue import Empty
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import streamlit as st
import logging

from market_data.providers.bbg import BloombergProvider

st.set_page_config(page_title="Live Monitor", layout="wide")
st.title("Live Monitor — Multi-Strategy Live Pricing")

# ---------- Provider ----------
provider: Optional[BloombergProvider] = None
try:
    provider = BloombergProvider()
except Exception as e:
    st.error(f"Bloomberg provider init failed: {e}")

# ---------- session state ----------
if "strategies" not in st.session_state:
    st.session_state.strategies: List[Dict] = []  # each: {id, name, legs}
if "feed_queue" not in st.session_state:
    st.session_state.feed_queue = None
if "stop_event" not in st.session_state:
    st.session_state.stop_event = None
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "last_snapshot_time" not in st.session_state:
    st.session_state.last_snapshot_time = None

# configure logging for server console
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("live_monitor")

# ---------- Helpers ----------
def normalize_snapshot(snapshot):
    """Normalize snapshot to DataFrame with lowercase columns and stripped string index.
    Accepts DataFrame or other types; returns DataFrame or original object.
    """
    try:
        if isinstance(snapshot, pd.DataFrame):
            # lowercase columns
            snapshot = snapshot.copy()
            snapshot.columns = [c.lower() for c in snapshot.columns]
            # coerce index to string and strip whitespace where possible
            try:
                snapshot.index = [str(i).strip() for i in snapshot.index]
            except Exception:
                pass
            return snapshot
    except Exception:
        pass
    return snapshot

def get_snapshot_value(snapshot, row_key, possible_cols):
    """Safely fetch a value for row_key and any of possible_cols (lowercase) from snapshot DataFrame or dict.
    Returns None if not found.
    """
    if snapshot is None:
        return None
    # if DataFrame
    if isinstance(snapshot, pd.DataFrame):
        # ensure row exists; allow approximate match if exact not found
        key = row_key
        if key not in snapshot.index:
            # try trimming or searching by contains
            key_candidates = [idx for idx in snapshot.index if key.lower() in idx.lower()]
            key = key_candidates[0] if key_candidates else None
        if key is None:
            return None
        row = snapshot.loc[key]
        for col in possible_cols:
            if col in snapshot.columns:
                try:
                    val = row.get(col)
                    if pd.notna(val):
                        return val
                except Exception:
                    continue
        return None
    # if dict-like
    if isinstance(snapshot, dict):
        if row_key in snapshot:
            row = snapshot[row_key]
            if isinstance(row, dict):
                for col in possible_cols:
                    v = row.get(col)
                    if v is not None:
                        return v
        # fallback: try first matching key
        for k, row in snapshot.items():
            try:
                if row_key.lower() in str(k).lower() and isinstance(row, dict):
                    for col in possible_cols:
                        v = row.get(col)
                        if v is not None:
                            return v
            except Exception:
                continue
    return None

def build_bbg_ticker(underlying: str, maturity_date: datetime, strike: float, opt_type: str) -> str:
    """Build Bloomberg security ticker: UNDERLYING MM/DD/YY C|P STRIKE Index"""
    mat_str = maturity_date.strftime("%m/%d/%y")
    cp = "C" if opt_type.lower().startswith("c") else "P"
    return f"{underlying} {mat_str} {cp}{int(strike)} Index"

def drain_latest():
    """Get most recent snapshot from queue"""
    q = st.session_state.get("feed_queue")
    if q is None:
        logger.debug("drain_latest: no feed_queue in session_state")
        st.sidebar.text("drain_latest: no feed_queue")
        return None
    last = None
    try:
        # show qsize before draining
        try:
            if hasattr(q, "qsize"):
                st.sidebar.text(f"drain_latest: queue size before drain: {q.qsize()}")
        except Exception:
            pass
        # drain everything, keep last
        while True:
            last = q.get_nowait()
    except Empty:
        pass
    logger.debug("drain_latest: returning last snapshot type=%s", type(last) if last is not None else None)
    st.sidebar.text(f"drain_latest: last snapshot type: {type(last).__name__ if last is not None else 'None'}")
    return last

# ---------- Modal: Create/Edit Strategy ----------
@st.dialog("Create Strategy")
def create_strategy_modal():
    st.write("Add legs to your strategy:")
    
    # Strategy name
    strat_name = st.text_input("Strategy name (e.g., 'Bull Call Spread')", value="")
    
    # Reference price for delta adjustment
    st.markdown("### Reference Price (for delta-adjusted pricing)")
    ref_price = st.number_input("Underlying reference price", value=5740.0, help="Used to compute delta-adjusted leg values")
    
    # Legs editor
    st.markdown("### Legs")
    legs = []
    
    leg_count = st.number_input("Number of legs", min_value=1, max_value=5, value=1, step=1)
    
    for i in range(int(leg_count)):
        st.markdown(f"#### Leg {i+1}")
        col1, col2, col3 = st.columns(3)
        with col1:
            underlying = st.text_input(f"Underlying (L{i+1})", value="SX5E Index", key=f"underlying_{i}")
        with col2:
            strike = st.number_input(f"Strike (L{i+1})", value=6000.0, key=f"strike_{i}")
        with col3:
            opt_type = st.selectbox(f"Type (L{i+1})", ["Call", "Put"], key=f"type_{i}")
        
        col4, col5, col6 = st.columns(3)
        with col4:
            maturity = st.date_input(f"Maturity (L{i+1})", key=f"maturity_{i}")
        with col5:
            qty = st.number_input(f"Qty (L{i+1})", value=1, min_value=1, key=f"qty_{i}")
        with col6:
            side = st.selectbox(f"Side (L{i+1})", ["Buy", "Sell"], key=f"side_{i}")
        
        legs.append({
            "underlying": underlying,
            "strike": strike,
            "type": opt_type.lower(),
            "maturity": maturity,
            "qty": int(qty),
            "side": side.lower(),
        })
    
    if st.button("Create Strategy"):
        if not strat_name:
            st.error("Strategy name required")
        elif not all(l.get("underlying") for l in legs):
            st.error("All legs need underlying")
        else:
            # Build BBG tickers for each leg
            for leg in legs:
                leg["ticker"] = build_bbg_ticker(
                    leg["underlying"], 
                    leg["maturity"], 
                    leg["strike"], 
                    leg["type"]
                )
            # Add strategy
            sid = str(uuid.uuid4())[:8]
            st.session_state.strategies.insert(0, {
                "id": sid,
                "name": strat_name,
                "created_at": datetime.now(),
                "ref_price": ref_price,
                "legs": legs,
            })
            st.success(f"Strategy '{strat_name}' created with ref price {ref_price}!")
            st.rerun()

# ---------- Main UI ----------
# Top buttons
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    if st.button("➕ Create Strategy", width='stretch'):
        create_strategy_modal()

with col2:
    if st.button("▶ Start Monitoring", width='stretch'):
        if not st.session_state.strategies:
            st.error("No strategies to monitor")
        else:
            # Collect all tickers + underlyings
            all_tickers = set()
            for strat in st.session_state.strategies:
                for leg in strat.get("legs", []):
                    all_tickers.add(leg["ticker"])
                    all_tickers.add(leg["underlying"])
            
            fields = ["PX_LAST", "PX_BID", "PX_ASK", "IVOL_MID_RT", "OPT_DELTA", "BID_SIZE", "ASK_SIZE", "VOLUME"]
            
            # Debug: show what we'll request
            logger.debug("Starting monitoring. tickers=%s", all_tickers)
            st.sidebar.markdown("### Debug: Start Monitoring")
            st.sidebar.text(f"Tickers({len(all_tickers)}): {list(all_tickers)[:30]}")
            st.sidebar.text(f"Fields: {fields}")
            
            # Stop previous
            if st.session_state.stop_event is not None:
                st.session_state.stop_event.set()
            
            # Start new
            stop_ev = threading.Event()
            st.session_state.stop_event = stop_ev
            q = provider.start_background_poll(list(all_tickers), fields, interval=2.0)
            st.session_state.feed_queue = q
            st.session_state.monitoring = True
            st.success(f"Monitoring {len(all_tickers)} tickers")
            
            # Debug: confirm queue object returned
            logger.debug("start_background_poll returned queue=%s", type(q))
            st.sidebar.text(f"Queue object: {type(q)}")
            # show qsize if available
            try:
                if hasattr(q, "qsize"):
                    st.sidebar.text(f"Queue size (initial): {q.qsize()}")
            except Exception as e:
                logger.debug("Couldn't read qsize: %s", e)
            
            # Try to probe one sample from the queue (non-fatal)
            try:
                sample = q.get(timeout=3)  # consume one sample to inspect
                logger.debug("Probe: received sample type=%s", type(sample))
                if isinstance(sample, pd.DataFrame):
                    st.sidebar.text(f"Probe snapshot rows: {sample.shape[0]}")
                    st.sidebar.text(str(sample.head(3)))
                else:
                    # show trimmed representation
                    st.sidebar.text(f"Probe sample: {str(sample)[:500]}")
                # after probing, queue likely continues to receive new samples from backend thread
            except Exception as e:
                logger.debug("Probe: no sample within timeout or probe failed: %s", e)
                st.sidebar.text("Probe: no sample within timeout")
            
            # Provider introspection (best-effort)
            try:
                prov_info = {}
                if provider is not None:
                    prov_info["class"] = provider.__class__.__name__
                    for attr in ("thread", "_thread", "is_running", "running", "_running"):
                        if hasattr(provider, attr):
                            prov_info[attr] = str(getattr(provider, attr))
                st.sidebar.text(f"Provider info: {prov_info}")
            except Exception:
                pass

with col3:
    if st.button("⏹ Stop Monitoring", width='stretch'):
        if st.session_state.stop_event is not None:
            st.session_state.stop_event.set()
        st.session_state.feed_queue = None
        st.session_state.monitoring = False
        st.info("Monitoring stopped")

st.divider()

# ---------- Strategies List & Live View ----------
if not st.session_state.strategies:
    st.info("No strategies created. Click '➕ Create Strategy' to get started.")
else:
    # Get latest snapshot
    snapshot = drain_latest()
    snapshot = normalize_snapshot(snapshot)
    
    # Debug: print snapshot basic info to server console and sidebar
    if snapshot is None:
        logger.debug("No snapshot received (snapshot is None)")
        st.sidebar.text("Latest snapshot: None")
    else:
        logger.debug("Snapshot received type=%s", type(snapshot))
        try:
            if isinstance(snapshot, pd.DataFrame):
                logger.debug("Snapshot shape=%s, index sample=%s", snapshot.shape, snapshot.index[:10].tolist())
                st.sidebar.text(f"Snapshot rows: {snapshot.shape[0]}")
                st.sidebar.text(str(snapshot.head(5)))
            else:
                st.sidebar.text(f"Snapshot type: {type(snapshot)}")
                st.sidebar.text(str(snapshot)[:500])
        except Exception as e:
            logger.exception("Error while debugging snapshot: %s", e)
    
    for strat in st.session_state.strategies:
        with st.container(border=True):
            # Strategy header + delete button
            col_hdr1, col_hdr2 = st.columns([10, 1])
            with col_hdr1:
                st.markdown(f"### {strat['name']}")
                st.caption(f"Created: {strat['created_at'].strftime('%Y-%m-%d %H:%M:%S')} | Ref Price: {strat.get('ref_price', '—')}")
            with col_hdr2:
                if st.button("🗑", key=f"del_{strat['id']}", help="Delete strategy"):
                    st.session_state.strategies = [s for s in st.session_state.strategies if s["id"] != strat["id"]]
                    st.rerun()
            
            # Legs table
            leg_rows = []
            total_mark = 0.0
            total_delta = 0.0
            
            # determine current spot for this strategy (try first leg underlying)
            spot_current = None
            try:
                first_underlying = strat.get("legs", [])[0].get("underlying")
                # try common column names for spot
                spot_val = get_snapshot_value(snapshot, first_underlying, ["px_last", "px_last_trd", "px_last"] )
                if spot_val is not None and pd.notna(spot_val):
                    spot_current = float(spot_val)
            except Exception:
                spot_current = None
            
            # accumulate total delta-adjusted value (if needed)
            total_delta_adjusted = 0.0
            
            for leg_idx, leg in enumerate(strat.get("legs", [])):
                ticker = leg["ticker"]
                underlying = leg["underlying"]
                ref_price = strat.get("ref_price", None)
                
                # Fetch live data for this leg using robust lookups
                bid = get_snapshot_value(snapshot, ticker, ["px_bid", "bid", "bid_price"])
                ask = get_snapshot_value(snapshot, ticker, ["px_ask", "ask", "ask_price"])
                px_last = get_snapshot_value(snapshot, ticker, ["px_last", "last", "px_last_trd", "mid"])
                bid_size = get_snapshot_value(snapshot, ticker, ["bid_size", "bidsize"])
                ask_size = get_snapshot_value(snapshot, ticker, ["ask_size", "asksize"])
                volume = get_snapshot_value(snapshot, ticker, ["volume", "vol"])
                delta = get_snapshot_value(snapshot, ticker, ["opt_delta", "delta"])
                iv = get_snapshot_value(snapshot, ticker, ["ivol_mid_rt", "implied_vol", "iv"])
                
                # fallback mid calc
                mid = None
                if px_last is not None and pd.notna(px_last):
                    mid = float(px_last)
                elif bid is not None and ask is not None and pd.notna(bid) and pd.notna(ask):
                    try:
                        mid = (float(bid) + float(ask)) / 2.0
                    except Exception:
                        mid = None
                
                # Per-leg net price (per contract) and values
                net_price = mid  # per contract
                side_mult = 1.0 if leg["side"] == "buy" else -1.0
                net_value = (float(net_price) * leg["qty"] * side_mult) if net_price is not None else None
                if net_value is not None:
                    total_mark += net_value
                
                # Delta price (per contract) using current spot and reference
                delta_price = None
                if net_price is not None and delta is not None and spot_current is not None and ref_price is not None and pd.notna(delta):
                    try:
                        delta_val = float(delta)
                        delta_price = float(net_price) + delta_val * (spot_current - float(ref_price))
                        # accumulate if you want a delta-adjusted total
                        total_delta_adjusted += (delta_price * leg["qty"] * side_mult)
                    except Exception:
                        delta_price = None
                
                # Accumulate weighted delta
                if delta is not None and pd.notna(delta):
                    try:
                        total_delta += float(delta) * leg["qty"] * side_mult
                    except Exception:
                        pass
                
                # Format for display
                leg_rows.append({
                    "#": leg_idx + 1,
                    "Underlying": underlying,
                    "Ticker": ticker,
                    "Strike": int(leg["strike"]),
                    "Type": leg["type"].upper(),
                    "Maturity": leg["maturity"].strftime("%m/%d/%y"),
                    "Qty": leg["qty"],
                    "Side": "BUY" if leg["side"] == "buy" else "SELL",
                    "Bid": f"{float(bid):.2f}" if bid and pd.notna(bid) else "—",
                    "Ask": f"{float(ask):.2f}" if ask and pd.notna(ask) else "—",
                    "Delta": f"{float(delta):.4f}" if delta and pd.notna(delta) else "—",
                    "Net Price": f"{float(net_price):.2f}" if net_price is not None else "—",
                    "Net Value": f"{float(net_value):.2f}" if net_value is not None else "—",
                    "Delta Price": f"{float(delta_price):.2f}" if delta_price is not None else "—",
                })
            
            # Display legs table (use width='stretch' to avoid deprecation)
            leg_df = pd.DataFrame(leg_rows)
            st.dataframe(leg_df, width='stretch', hide_index=True)
            
            # Strategy summary metrics
            total_abs_qty = sum(abs(l.get("qty", 0)) for l in strat.get("legs", [])) or 1
            net_price_per_unit = total_mark / total_abs_qty
            delta_adj_total = total_delta_adjusted if total_delta_adjusted != 0.0 else None
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Total Net Price (Mark)", f"{total_mark:.2f}")
            with col_m2:
                st.metric("Net Price / Unit", f"{net_price_per_unit:.4f}")
            with col_m3:
                st.metric("Total Delta", f"{total_delta:.4f}")
            with col_m4:
                # show delta-adjusted total only if computed
                st.metric("Delta Price Total", f"{delta_adj_total:.2f}" if delta_adj_total is not None else "—")
            
            # show live spot (if available)
            if spot_current is not None:
                st.caption(f"Live spot/future price: {spot_current:.2f} (used for delta-price calc)")
            else:
                st.caption("Live spot/future price: —")

st.markdown("---")
st.markdown("""
**Usage:**
1. Click **➕ Create Strategy** to define a new strategy with underlying, strikes, maturity, qty, side
2. Enter **reference price** for delta-adjusted pricing
3. Click **▶ Start Monitoring** to begin live polling from Bloomberg
4. Live data updates in real-time for all legs
5. **Leg Value** = mid price × qty × side (buy=+1, sell=-1)
6. **Total Delta** = sum of (delta × qty × side) across all legs
7. **Delta-Adjusted Price** available if ref price provided (for future enhancement)
8. Click **🗑** to delete a strategy
""")