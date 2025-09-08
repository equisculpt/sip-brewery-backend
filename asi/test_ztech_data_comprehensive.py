"""
Comprehensive ZTECH Data Test with Error Handling and Alternative Sources
Tests ZTECH data fetching with fallback mechanisms
"""
import asyncio
import yfinance as yf
import requests
from datetime import datetime
import json

async def test_ztech_symbol_validation():
    """Test if ZTECH symbol is valid and actively traded"""
    print("=== ZTECH Symbol Validation ===")
    
    symbols_to_test = [
        "ZTECH.NS",    # NSE
        "543654.BO",   # BSE
        "ZTECH.BO",    # BSE alternative
        "ZTECH"        # Base symbol
    ]
    
    for symbol in symbols_to_test:
        try:
            print(f"\nTesting symbol: {symbol}")
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            
            if info and len(info) > 1:  # More than just symbol
                print(f"✅ Valid symbol: {symbol}")
                print(f"   Company: {info.get('longName', 'N/A')}")
                print(f"   Exchange: {info.get('exchange', 'N/A')}")
                print(f"   Currency: {info.get('currency', 'N/A')}")
                print(f"   Market State: {info.get('marketState', 'N/A')}")
                
                # Try to get recent data
                hist = ticker.history(period="5d")
                if not hist.empty:
                    latest = hist.iloc[-1]
                    print(f"   Latest Close: {latest['Close']:.2f}")
                    print(f"   Latest Volume: {latest['Volume']:,}")
                else:
                    print("   ⚠️ No recent trading data")
            else:
                print(f"❌ Invalid or inactive symbol: {symbol}")
                
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")

async def test_alternative_data_sources():
    """Test alternative data sources for ZTECH"""
    print("\n=== Alternative Data Sources ===")
    
    # Test NSE website directly
    try:
        print("\n1. Testing NSE Direct API...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://www.nseindia.com/'
        }
        
        # Try NSE Emerge API
        url = "https://www.nseindia.com/api/emerge-quote-equity?symbol=ZTECH"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ NSE Emerge API working")
            print(f"   Company: {data.get('securityInfo', {}).get('companyName', 'N/A')}")
            
            price_info = data.get('priceInfo', {})
            if price_info:
                print(f"   Last Price: ₹{price_info.get('lastPrice', 'N/A')}")
                print(f"   Change: ₹{price_info.get('change', 'N/A')} ({price_info.get('pChange', 'N/A')}%)")
        else:
            print(f"❌ NSE API failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ NSE API error: {e}")
    
    # Test BSE website
    try:
        print("\n2. Testing BSE Data...")
        # BSE scrip code for ZTECH
        bse_url = "https://api.bseindia.com/BseIndiaAPI/api/StockReachGraph/w"
        params = {
            'scripcode': '543654',
            'flag': 'sp',
            'fromdate': '',
            'todate': '',
            'seriesid': ''
        }
        
        response = requests.get(bse_url, params=params, timeout=10)
        if response.status_code == 200:
            print("✅ BSE API accessible")
            # BSE API structure varies, would need specific parsing
        else:
            print(f"❌ BSE API failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ BSE API error: {e}")

async def test_ztech_with_fallbacks():
    """Test ZTECH data with multiple fallback mechanisms"""
    print("\n=== ZTECH Data with Fallbacks ===")
    
    # Import our service
    from ztech_ohlc_live_service import ZTechDataService
    
    async with ZTechDataService() as service:
        print("\n1. Testing Live Price...")
        
        # Test yfinance
        live_yf = service.get_live_price_yfinance()
        if live_yf:
            print("✅ YFinance live price working")
            print(f"   Price: ₹{live_yf.price}")
            print(f"   Source: {live_yf.source}")
        else:
            print("❌ YFinance live price failed")
        
        # Test NSE API
        live_nse = await service.get_live_price_nse_api()
        if live_nse:
            print("✅ NSE API live price working")
            print(f"   Price: ₹{live_nse.price}")
            print(f"   Source: {live_nse.source}")
        else:
            print("❌ NSE API live price failed")
        
        print("\n2. Testing OHLC Data...")
        
        # Test different timeframes
        timeframes = ["1d", "1wk"]
        periods = ["1mo", "3mo"]
        
        for tf in timeframes:
            for period in periods:
                try:
                    from ztech_ohlc_live_service import TimeFrame, Period
                    ohlc_data = service.get_ohlc_data(TimeFrame(tf), Period(period))
                    
                    if ohlc_data:
                        print(f"✅ OHLC data for {tf}/{period}: {len(ohlc_data)} records")
                        if ohlc_data:
                            latest = ohlc_data[-1]
                            print(f"   Latest: O:{latest.open} H:{latest.high} L:{latest.low} C:{latest.close}")
                    else:
                        print(f"❌ No OHLC data for {tf}/{period}")
                        
                except Exception as e:
                    print(f"❌ OHLC error for {tf}/{period}: {e}")

async def test_mock_ztech_data():
    """Create mock ZTECH data for testing when real data is unavailable"""
    print("\n=== Mock ZTECH Data for Testing ===")
    
    from ztech_ohlc_live_service import LivePriceData, OHLCData
    from datetime import datetime, timedelta
    import random
    
    # Create mock live price data
    mock_live_price = LivePriceData(
        symbol="ZTECH",
        price=275.50,
        change=8.25,
        change_percent=3.09,
        volume=125000,
        high=278.90,
        low=267.30,
        open=270.00,
        previous_close=267.25,
        market_cap=2750000000,  # 275 crores
        timestamp=datetime.now(),
        source="mock_data"
    )
    
    print("✅ Mock Live Price Data Created:")
    print(f"   Price: ₹{mock_live_price.price}")
    print(f"   Change: ₹{mock_live_price.change} ({mock_live_price.change_percent}%)")
    print(f"   Volume: {mock_live_price.volume:,}")
    print(f"   Day Range: ₹{mock_live_price.low} - ₹{mock_live_price.high}")
    
    # Create mock OHLC data
    mock_ohlc = []
    base_price = 270.0
    
    for i in range(30):  # 30 days of data
        date = datetime.now() - timedelta(days=29-i)
        
        # Simulate price movement
        open_price = base_price + random.uniform(-5, 5)
        close_price = open_price + random.uniform(-10, 10)
        high_price = max(open_price, close_price) + random.uniform(0, 5)
        low_price = min(open_price, close_price) - random.uniform(0, 5)
        volume = random.randint(50000, 200000)
        
        ohlc = OHLCData(
            timestamp=date,
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=volume
        )
        
        mock_ohlc.append(ohlc)
        base_price = close_price  # Use close as next day's base
    
    print(f"✅ Mock OHLC Data Created: {len(mock_ohlc)} records")
    latest_ohlc = mock_ohlc[-1]
    print(f"   Latest OHLC: O:{latest_ohlc.open} H:{latest_ohlc.high} L:{latest_ohlc.low} C:{latest_ohlc.close}")
    
    # Create comprehensive mock data
    mock_comprehensive = {
        "symbol": "ZTECH",
        "company_name": "Zentech Systems Limited",
        "exchange": "NSE_EMERGE",
        "live_price": mock_live_price.to_dict(),
        "daily_ohlc": [ohlc.to_dict() for ohlc in mock_ohlc],
        "technical_indicators": {
            "sma_20": 272.45,
            "sma_50": 268.30,
            "rsi": 65.4,
            "bollinger_upper": 285.20,
            "bollinger_lower": 255.80,
            "52_week_high": 320.50,
            "52_week_low": 180.25,
            "avg_volume_20": 145000,
            "volume_ratio": 0.86
        },
        "data_sources": ["mock_data"],
        "timestamp": datetime.now().isoformat()
    }
    
    print("✅ Mock Comprehensive Data Created")
    print(f"   Technical Indicators: {len(mock_comprehensive['technical_indicators'])}")
    print(f"   RSI: {mock_comprehensive['technical_indicators']['rsi']}")
    print(f"   SMA 20: {mock_comprehensive['technical_indicators']['sma_20']}")
    
    return mock_comprehensive

async def main():
    """Run all ZTECH tests"""
    print("🚀 ZTECH Comprehensive Data Testing")
    print("=" * 50)
    
    # Test symbol validation
    await test_ztech_symbol_validation()
    
    # Test alternative sources
    await test_alternative_data_sources()
    
    # Test with fallbacks
    await test_ztech_with_fallbacks()
    
    # Create mock data for testing
    mock_data = await test_mock_ztech_data()
    
    print("\n" + "=" * 50)
    print("✅ ZTECH Testing Complete!")
    print("\n📋 Summary:")
    print("- Symbol validation tested across NSE/BSE")
    print("- Alternative data sources explored")
    print("- Fallback mechanisms implemented")
    print("- Mock data created for testing")
    print("\n💡 Recommendation:")
    print("- Use mock data for development/testing")
    print("- Implement real-time data when ZTECH becomes actively traded")
    print("- Monitor NSE Emerge platform for ZTECH activity")

if __name__ == "__main__":
    asyncio.run(main())
