import yfinance as yf
import pandas as pd

symbol = "RELIANCE.NS"
interval = "1m"
period = "1d"

df = yf.download(symbol, interval=interval, period=period, progress=False)
if df.empty:
    print(f"No data returned for {symbol}")
else:
    df = df.reset_index()
    print(f"Latest {interval} OHLCV for {symbol} (period={period}):")
    print(df.tail(5))
