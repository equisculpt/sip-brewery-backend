import yfinance as yf
import psycopg2
import time

def fetch_and_store_eod(symbols, db_conn):
    for i, symbol in enumerate(symbols):
        try:
            df = yf.download(symbol, interval="1d", period="2d", progress=False)
            if df.empty:
                continue
            last_row = df.iloc[-1]
            with db_conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO stock_ohlcv (symbol, date, open, high, low, close, volume)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (symbol, date) DO UPDATE
                       SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                           close=EXCLUDED.close, volume=EXCLUDED.volume
                    """, (
                        symbol, last_row.name.date(), float(last_row["Open"]), float(last_row["High"]),
                        float(last_row["Low"]), float(last_row["Close"]), int(last_row["Volume"])
                    )
                )
            db_conn.commit()
            # Throttle to avoid overloading Yahoo (not usually needed, but safe)
            if i % 50 == 0:
                time.sleep(2)
        except Exception as e:
            print(f"Error for {symbol}: {e}")

if __name__ == "__main__":
    # List of NSE symbols, e.g., ["RELIANCE.NS", "TCS.NS", ...]
    symbols = ["RELIANCE.NS", "TCS.NS", "SBIN.NS"]  # Replace with your full list!
    db_conn = psycopg2.connect(
        dbname="YOUR_DB", user="YOUR_USER", password="YOUR_PASS", host="localhost", port=5432
    )
    fetch_and_store_eod(symbols, db_conn)
    db_conn.close()
