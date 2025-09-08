import asyncio
from real_market_data_asi import RealMarketDataASI

SYMBOLS = ["RELIANCE", "TCS", "SBIN"]

async def main():
    asi = RealMarketDataASI()
    await asi.initialize()
    try:
        for symbol in SYMBOLS:
            print(f"\n=== LIVE DATA FOR {symbol} ===")
            data = await asi.get_real_time_data(symbol)
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
            print(f"Price: {data.get('price')}")
            print(f"Change: {data.get('change')}")
            print(f"Percent Change: {data.get('change_percent')}")
            print(f"Volume: {data.get('volume')}")
            print(f"Timestamp: {data.get('timestamp')}")
            print(f"Sources: {data.get('sources')}")
            print(f"Confidence: {data.get('confidence')}")
            print(f"Raw: {data}")
            # Print which sources provided the price
            if 'sources' in data:
                print(f"Data sources: {data['sources']}")
            await asyncio.sleep(1)  # Gentle throttle
    finally:
        await asi.close()

if __name__ == "__main__":
    asyncio.run(main())
