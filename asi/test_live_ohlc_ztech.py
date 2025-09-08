import yfinance as yf
import pandas as pd

symbol = "ZTECH.NS"  # Ztech India NSE symbol
interval = "1m"
period = "1d"

df = yf.download(symbol, interval=interval, period=period, progress=False)
if df.empty:
    print(f"No data returned for {symbol}. Check symbol or market status.")
else:
    df = df.reset_index()
    latest = df.iloc[-1]
    print(f"Latest 1-min OHLCV for {symbol}:")
    print(latest)
    print(f"Live price: {latest['Close']}")
