import requests
import json

# NSE symbol for KRN Heat Exchanger (replace with the correct symbol if different)
symbol = "KRNHEAT"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive"
}

session = requests.Session()
session.headers.update(headers)
import time
# Hit homepage to set cookies, then wait a bit
home_resp = session.get("https://www.nseindia.com")
time.sleep(2)

url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
resp = session.get(url)
if resp.status_code == 200:
    data = resp.json()
    price = data.get("priceInfo", {}).get("lastPrice")
    print(f"KRN Heat Exchanger (NSE: {symbol}) live price: {price}")
else:
    print(f"Failed to fetch price. Status: {resp.status_code}")
    print(f"Response headers: {resp.headers}")
    print(f"Response text: {resp.text[:500]}")
