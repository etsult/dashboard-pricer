from curl_cffi import requests as curl_requests
import yfinance as yf

# Create a curl_cffi session with SSL verify disabled (for your firewall)
unsafe_session = curl_requests.Session()
unsafe_session.verify = False

ticker = "^FTSE"  # Single ticker example

# Download 1 day of 1-minute interval data
data = yf.download(ticker, period="1d", interval="1m", session=unsafe_session)

if not data.empty:
    last_price = data['Close'][-1]
    print(f"Latest price for {ticker}: {last_price:.2f}")
else:
    print(f"No data found for {ticker}")
