"""
ZTECH API Demo - Live Price and OHLC Data
Demonstrates all ZTECH endpoints with real and mock data
"""
import asyncio
import json
from datetime import datetime
from ztech_ohlc_live_service import get_ztech_live_price, get_ztech_ohlc, get_ztech_comprehensive

async def demo_ztech_live_price():
    """Demo ZTECH live price endpoint"""
    print("🔴 ZTECH Live Price Demo")
    print("-" * 40)
    
    try:
        result = await get_ztech_live_price()
        
        if result["status"] == "success":
            data = result["data"]
            print(f"✅ Live Price Retrieved Successfully")
            print(f"📊 Current Price: ₹{data['price']}")
            print(f"📈 Change: ₹{data['change']} ({data['change_percent']:.2f}%)")
            print(f"📊 Volume: {data['volume']:,}")
            print(f"📊 Day Range: ₹{data['low']} - ₹{data['high']}")
            print(f"📊 Previous Close: ₹{data['previous_close']}")
            print(f"🕐 Last Updated: {data['timestamp']}")
            print(f"📡 Data Source: {data['source']}")
            
            # Determine market sentiment
            if data['change_percent'] > 2:
                sentiment = "🟢 Strongly Bullish"
            elif data['change_percent'] > 0:
                sentiment = "🟢 Bullish"
            elif data['change_percent'] < -2:
                sentiment = "🔴 Strongly Bearish"
            elif data['change_percent'] < 0:
                sentiment = "🔴 Bearish"
            else:
                sentiment = "⚪ Neutral"
            
            print(f"📊 Market Sentiment: {sentiment}")
            
        else:
            print(f"❌ Error: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

async def demo_ztech_ohlc():
    """Demo ZTECH OHLC data endpoint"""
    print("\n📊 ZTECH OHLC Data Demo")
    print("-" * 40)
    
    timeframes = [
        ("1d", "1mo", "Daily data for 1 month"),
        ("1wk", "3mo", "Weekly data for 3 months"),
        ("1d", "1y", "Daily data for 1 year")
    ]
    
    for timeframe, period, description in timeframes:
        try:
            print(f"\n📈 {description}")
            result = await get_ztech_ohlc(timeframe, period)
            
            if result["status"] == "success":
                data = result["data"]
                ohlc_records = data["ohlc"]
                
                print(f"✅ Retrieved {data['count']} OHLC records")
                
                if ohlc_records:
                    # Show first and last records
                    first_record = ohlc_records[0]
                    last_record = ohlc_records[-1]
                    
                    print(f"📅 Period: {first_record['timestamp'][:10]} to {last_record['timestamp'][:10]}")
                    print(f"📊 Latest OHLC:")
                    print(f"   Open: ₹{last_record['open']}")
                    print(f"   High: ₹{last_record['high']}")
                    print(f"   Low: ₹{last_record['low']}")
                    print(f"   Close: ₹{last_record['close']}")
                    print(f"   Volume: {last_record['volume']:,}")
                    
                    # Calculate basic statistics
                    prices = [float(record['close']) for record in ohlc_records]
                    volumes = [int(record['volume']) for record in ohlc_records if record['volume']]
                    
                    if prices:
                        print(f"📊 Period Statistics:")
                        print(f"   Highest: ₹{max(prices):.2f}")
                        print(f"   Lowest: ₹{min(prices):.2f}")
                        print(f"   Average: ₹{sum(prices)/len(prices):.2f}")
                        
                    if volumes:
                        print(f"   Avg Volume: {sum(volumes)//len(volumes):,}")
                        
                else:
                    print("⚠️ No OHLC data available")
                    
            else:
                print(f"❌ Error: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception for {timeframe}/{period}: {e}")

async def demo_ztech_comprehensive():
    """Demo comprehensive ZTECH data"""
    print("\n🎯 ZTECH Comprehensive Data Demo")
    print("-" * 40)
    
    try:
        result = await get_ztech_comprehensive()
        
        if result["status"] == "success":
            data = result["data"]
            
            print(f"✅ Comprehensive Data Retrieved")
            print(f"🏢 Company: {data['company_name']}")
            print(f"🏛️ Exchange: {data['exchange']}")
            print(f"📡 Data Sources: {', '.join(data['data_sources'])}")
            
            # Live price summary
            if data.get("live_price"):
                live = data["live_price"]
                print(f"\n💰 Live Price Summary:")
                print(f"   Current: ₹{live['price']} ({live['change_percent']:+.2f}%)")
                print(f"   Day Range: ₹{live['low']} - ₹{live['high']}")
            
            # OHLC summary
            daily_ohlc = data.get("daily_ohlc", [])
            intraday_ohlc = data.get("intraday_ohlc", [])
            
            print(f"\n📊 OHLC Data Summary:")
            print(f"   Daily Records: {len(daily_ohlc)}")
            print(f"   Intraday Records: {len(intraday_ohlc)}")
            
            # Technical indicators
            indicators = data.get("technical_indicators", {})
            if indicators:
                print(f"\n🔬 Technical Analysis:")
                
                if "sma_20" in indicators:
                    print(f"   SMA 20: ₹{indicators['sma_20']:.2f}")
                if "sma_50" in indicators:
                    print(f"   SMA 50: ₹{indicators['sma_50']:.2f}")
                if "rsi" in indicators:
                    rsi = indicators['rsi']
                    rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    print(f"   RSI: {rsi:.2f} ({rsi_signal})")
                if "bollinger_upper" in indicators and "bollinger_lower" in indicators:
                    print(f"   Bollinger Bands: ₹{indicators['bollinger_lower']:.2f} - ₹{indicators['bollinger_upper']:.2f}")
                if "52_week_high" in indicators and "52_week_low" in indicators:
                    print(f"   52W Range: ₹{indicators['52_week_low']:.2f} - ₹{indicators['52_week_high']:.2f}")
                if "volume_ratio" in indicators:
                    vol_ratio = indicators['volume_ratio']
                    vol_signal = "High" if vol_ratio > 1.5 else "Low" if vol_ratio < 0.5 else "Normal"
                    print(f"   Volume Ratio: {vol_ratio:.2f} ({vol_signal})")
            
        else:
            print(f"❌ Error: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

async def demo_api_endpoints():
    """Demo API endpoint URLs"""
    print("\n🌐 ZTECH API Endpoints")
    print("-" * 40)
    
    base_url = "http://localhost:8000"  # Adjust as needed
    
    endpoints = [
        ("Live Price", f"{base_url}/api/v2/ztech/live-price"),
        ("OHLC Data", f"{base_url}/api/v2/ztech/ohlc?timeframe=1d&period=1mo"),
        ("Comprehensive", f"{base_url}/api/v2/ztech/comprehensive"),
        ("Intraday", f"{base_url}/api/v2/ztech/intraday?timeframe=5m"),
        ("Technical Analysis", f"{base_url}/api/v2/ztech/technical-analysis"),
        ("Company Info", f"{base_url}/api/v2/ztech/company-info"),
        ("Entity Resolution", f"{base_url}/api/v2/enhanced-entity-resolution?query=ztech"),
        ("SME Data", f"{base_url}/api/v2/sme-emerge/ZTECH?exchange=NSE_EMERGE")
    ]
    
    print("📋 Available Endpoints:")
    for name, url in endpoints:
        print(f"   {name}: {url}")
    
    print("\n📝 Usage Examples:")
    print("   curl http://localhost:8000/api/v2/ztech/live-price")
    print("   curl http://localhost:8000/api/v2/ztech/ohlc?timeframe=1d&period=1mo")
    print("   curl http://localhost:8000/api/v2/ztech/comprehensive")

def demo_query_examples():
    """Demo natural language query examples"""
    print("\n🗣️ Natural Language Query Examples")
    print("-" * 40)
    
    queries = [
        "ztech share price",
        "zentech systems stock",
        "ztech live price",
        "zentech ohlc data",
        "ztech technical analysis",
        "emerge:ztech",
        "nse emerge ztech",
        "small cap it stocks",
        "sme it companies"
    ]
    
    print("📝 Supported Queries:")
    for query in queries:
        print(f"   '{query}' → Resolves to ZTECH data")
    
    print("\n🎯 Query Resolution:")
    print("   - 'ztech' → ZTECH (Zentech Systems)")
    print("   - 'zentech' → ZTECH")
    print("   - 'zentech systems' → ZTECH")
    print("   - 'emerge:ztech' → ZTECH on NSE Emerge")

async def main():
    """Run complete ZTECH API demo"""
    print("🚀 ZTECH API Complete Demo")
    print("=" * 50)
    print("Zentech Systems Limited (ZTECH)")
    print("NSE Emerge Platform | IT Services Sector")
    print("=" * 50)
    
    # Run all demos
    await demo_ztech_live_price()
    await demo_ztech_ohlc()
    await demo_ztech_comprehensive()
    await demo_api_endpoints()
    demo_query_examples()
    
    print("\n" + "=" * 50)
    print("✅ ZTECH API Demo Complete!")
    print("\n🎯 Key Features Demonstrated:")
    print("✅ Real-time live price data")
    print("✅ Historical OHLC data (multiple timeframes)")
    print("✅ Technical indicators and analysis")
    print("✅ Comprehensive data aggregation")
    print("✅ Natural language query resolution")
    print("✅ NSE Emerge platform integration")
    print("✅ RESTful API endpoints")
    print("✅ Error handling and fallbacks")
    
    print("\n💡 Next Steps:")
    print("1. Start the enhanced API server")
    print("2. Test endpoints using curl or Postman")
    print("3. Integrate with frontend applications")
    print("4. Monitor real-time data feeds")
    print("5. Set up alerts and notifications")

if __name__ == "__main__":
    asyncio.run(main())
