from fastapi import FastAPI
from typing import List
import yfinance as yf
import pandas as pd

app = FastAPI()

@app.get("/ohlc/live")
def get_live_ohlc(symbol: str, interval: str = "1m", period: str = "1d"):
    """
    Fetch live OHLCV bars for a symbol (intraday or daily).
    """
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    df = df.reset_index()
    bars = []
    for _, row in df.iterrows():
        time_col = "Datetime" if "Datetime" in row else "Date"
        bars.append({
            "time": int(row[time_col].timestamp()),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"])
        })
    return {"symbol": symbol, "interval": interval, "bars": bars}
