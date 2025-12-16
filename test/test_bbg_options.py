import re
import time
from datetime import datetime
from xbbg import blp
import pandas as pd

print("="*60)
print("Fetching SX5E Option Chain")
print("="*60)

ticker = "SX5E Index"
chain_df = blp.bds(ticker, "OPT_CHAIN")
print(f"Total options in chain: {len(chain_df)}")
print(f"Sample tickers:\n{chain_df.head(10)}\n")

# Parse security_description to extract expiry, strike, call/put
print("="*60)
print("Parsing Option Details")
print("="*60)

option_details = []
for sec_desc in chain_df['security_description'].head(10):  # reduced to 10 for demo
    match = re.search(r'(\d{2}/\d{2}/\d{2})\s+([CP])(\d+)', sec_desc)
    if match:
        expiry_str, cp_flag, strike_str = match.groups()
        expiry = datetime.strptime(expiry_str, "%m/%d/%y").date()
        strike = float(strike_str)
        option_type = "call" if cp_flag == "C" else "put"
        option_details.append({
            'ticker': sec_desc,
            'expiry': expiry,
            'strike': strike,
            'type': option_type,
        })

print(f"Parsed {len(option_details)} options\n")

# Fetch fields for each option
fields = [
    "PX_BID", "PX_ASK", "BID_SIZE", "ASK_SIZE",
    "IVOL_MID_RT", "OPT_DELTA", "OPT_GAMMA", "OPT_VEGA", "OPT_THETA",
    "VOLUME", "OPEN_INT", "PX_LAST", "MID"
]

def fetch_option_data(options_list):
    """Fetch current option data"""
    all_data = []
    for opt in options_list:
        try:
            data = blp.bdp(opt['ticker'], fields)
            row = data.iloc[0].to_dict()
            row.update({
                'ticker': opt['ticker'],
                'expiry': opt['expiry'],
                'strike': opt['strike'],
                'type': opt['type'],
                'timestamp': datetime.now()
            })
            all_data.append(row)
        except Exception as e:
            print(f"Error fetching {opt['ticker']}: {e}")
    return pd.DataFrame(all_data)

print("="*60)
print("Live Price Monitoring (3 iterations)")
print("="*60)

# Monitor for 3 iterations with 5-second intervals
for iteration in range(3):
    print(f"\n--- Iteration {iteration + 1} at {datetime.now().strftime('%H:%M:%S')} ---")
    
    result_df = fetch_option_data(option_details)
    
    # Display key columns
    display_cols = ['ticker', 'strike', 'type', 'px_bid', 'px_ask', 'mid', 
                   'ivol_mid_rt', 'opt_delta', 'volume', 'timestamp']
    print(result_df[display_cols].to_string(index=False))
    
    # Also fetch and display spot price
    spot = blp.bdp(ticker, ["PX_LAST"])
    spot_price = spot.iloc[0]["px_last"]
    print(f"\nSpot Price ({ticker}): {spot_price}")
    
    if iteration < 2:  # Don't sleep after last iteration
        print("\nWaiting 5 seconds before next update...", end="")
        for i in range(5):
            time.sleep(1)
            print(".", end="", flush=True)
        print()

print("\n" + "="*60)
print("Monitoring Complete")
print("="*60)