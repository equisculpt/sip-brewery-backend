"""
ZTECH India Comprehensive Search and Analysis
Searches for all ZTECH variations including ZTECH India across Indian stock exchanges
"""
import yfinance as yf
import requests
import asyncio
import aiohttp
from datetime import datetime
import json
import re
from typing import List, Dict, Any, Optional

class ZTechIndiaSearchEngine:
    """Comprehensive search engine for ZTECH India variations"""
    
    def __init__(self):
        self.search_variations = [
            # Direct variations
            "ZTECH",
            "ZTECHINDIA", 
            "ZTECH-INDIA",
            "ZTECH_INDIA",
            "ZTECHIND",
            
            # Company name variations
            "ZENTECH",
            "ZENTECHINDIA",
            "ZENTECH-INDIA", 
            "ZENTECH_INDIA",
            "ZENTECHIND",
            "ZENTECHSYSTEMS",
            "ZENTECH-SYSTEMS",
            
            # Full company names
            "ZENTECH SYSTEMS",
            "ZENTECH SYSTEMS LIMITED",
            "ZENTECH SYSTEMS LTD",
            "ZTECH SYSTEMS",
            "ZTECH SYSTEMS LIMITED",
            "ZTECH SYSTEMS LTD",
            
            # India specific
            "ZENTECH INDIA",
            "ZENTECH INDIA LIMITED", 
            "ZENTECH INDIA LTD",
            "ZTECH INDIA",
            "ZTECH INDIA LIMITED",
            "ZTECH INDIA LTD"
        ]
        
        self.exchange_suffixes = [
            ".NS",    # NSE
            ".BO",    # BSE
            ".BSE",   # BSE alternative
            ".NSE"    # NSE alternative
        ]
        
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def test_yfinance_symbols(self) -> List[Dict[str, Any]]:
        """Test all ZTECH variations with Yahoo Finance"""
        print("🔍 Testing Yahoo Finance Symbols...")
        print("-" * 50)
        
        valid_symbols = []
        
        # Test base variations
        for variation in self.search_variations:
            for suffix in self.exchange_suffixes:
                symbol = f"{variation}{suffix}"
                
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    # Check if symbol has meaningful data
                    if info and len(info) > 5:  # More than just basic fields
                        company_name = info.get('longName', info.get('shortName', 'N/A'))
                        exchange = info.get('exchange', 'N/A')
                        currency = info.get('currency', 'N/A')
                        market_state = info.get('marketState', 'N/A')
                        
                        print(f"✅ Found: {symbol}")
                        print(f"   Company: {company_name}")
                        print(f"   Exchange: {exchange}")
                        print(f"   Currency: {currency}")
                        print(f"   Market State: {market_state}")
                        
                        # Try to get recent price data
                        try:
                            hist = ticker.history(period="5d")
                            if not hist.empty:
                                latest = hist.iloc[-1]
                                print(f"   Latest Price: {latest['Close']:.2f}")
                                print(f"   Volume: {latest['Volume']:,}")
                                
                                valid_symbols.append({
                                    'symbol': symbol,
                                    'company_name': company_name,
                                    'exchange': exchange,
                                    'currency': currency,
                                    'market_state': market_state,
                                    'latest_price': float(latest['Close']),
                                    'volume': int(latest['Volume']),
                                    'data_available': True
                                })
                            else:
                                print(f"   ⚠️ No recent trading data")
                                valid_symbols.append({
                                    'symbol': symbol,
                                    'company_name': company_name,
                                    'exchange': exchange,
                                    'currency': currency,
                                    'market_state': market_state,
                                    'data_available': False
                                })
                        except Exception as e:
                            print(f"   ⚠️ Price data error: {e}")
                            valid_symbols.append({
                                'symbol': symbol,
                                'company_name': company_name,
                                'exchange': exchange,
                                'currency': currency,
                                'market_state': market_state,
                                'data_available': False
                            })
                        
                        print()
                        
                except Exception as e:
                    # Skip invalid symbols silently
                    continue
        
        return valid_symbols
    
    async def search_nse_symbols(self) -> List[Dict[str, Any]]:
        """Search NSE for ZTECH variations"""
        print("🔍 Searching NSE Database...")
        print("-" * 50)
        
        nse_results = []
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Referer': 'https://www.nseindia.com/'
            }
            
            # Search NSE main board
            for variation in self.search_variations[:10]:  # Limit to avoid rate limiting
                try:
                    url = f"https://www.nseindia.com/api/search/autocomplete?q={variation}"
                    
                    async with self.session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if 'symbols' in data:
                                for symbol_data in data['symbols']:
                                    symbol = symbol_data.get('symbol', '')
                                    company_name = symbol_data.get('symbol_info', '')
                                    
                                    if any(term.lower() in symbol.lower() or term.lower() in company_name.lower() 
                                          for term in ['ztech', 'zentech']):
                                        
                                        print(f"✅ NSE Found: {symbol}")
                                        print(f"   Company: {company_name}")
                                        
                                        nse_results.append({
                                            'symbol': symbol,
                                            'company_name': company_name,
                                            'exchange': 'NSE',
                                            'source': 'nse_search'
                                        })
                        
                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.5)
                        
                except Exception as e:
                    print(f"   Error searching {variation}: {e}")
                    continue
            
            # Search NSE Emerge
            try:
                emerge_url = "https://www.nseindia.com/api/emerge-equity-list"
                
                async with self.session.get(emerge_url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'data' in data:
                            for company in data['data']:
                                symbol = company.get('symbol', '')
                                company_name = company.get('companyName', '')
                                
                                if any(term.lower() in symbol.lower() or term.lower() in company_name.lower() 
                                      for term in ['ztech', 'zentech']):
                                    
                                    print(f"✅ NSE Emerge Found: {symbol}")
                                    print(f"   Company: {company_name}")
                                    
                                    nse_results.append({
                                        'symbol': symbol,
                                        'company_name': company_name,
                                        'exchange': 'NSE_EMERGE',
                                        'source': 'nse_emerge'
                                    })
                            
            except Exception as e:
                print(f"   Error searching NSE Emerge: {e}")
                
        except Exception as e:
            print(f"❌ NSE search error: {e}")
        
        return nse_results
    
    def search_bse_symbols(self) -> List[Dict[str, Any]]:
        """Search BSE for ZTECH variations"""
        print("🔍 Searching BSE Database...")
        print("-" * 50)
        
        bse_results = []
        
        try:
            # BSE search is more limited, try common scrip codes
            potential_codes = [
                '543654',  # Known ZTECH code
                '543655', '543656', '543657',  # Adjacent codes
                '500001', '500002',  # Common ranges
            ]
            
            for code in potential_codes:
                try:
                    # Try to get company info
                    url = f"https://api.bseindia.com/BseIndiaAPI/api/ComHeader/w?quotetype=EQ&scripcode={code}&seriesid="
                    
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'Table' in data and data['Table']:
                            company_info = data['Table'][0]
                            company_name = company_info.get('FullN', '')
                            script_name = company_info.get('ScripName', '')
                            
                            if any(term.lower() in company_name.lower() or term.lower() in script_name.lower() 
                                  for term in ['ztech', 'zentech']):
                                
                                print(f"✅ BSE Found: {code}")
                                print(f"   Company: {company_name}")
                                print(f"   Script: {script_name}")
                                
                                bse_results.append({
                                    'scrip_code': code,
                                    'company_name': company_name,
                                    'script_name': script_name,
                                    'exchange': 'BSE',
                                    'source': 'bse_search'
                                })
                
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"❌ BSE search error: {e}")
        
        return bse_results
    
    def search_company_databases(self) -> List[Dict[str, Any]]:
        """Search our enhanced company database"""
        print("🔍 Searching Enhanced Company Database...")
        print("-" * 50)
        
        from enhanced_indian_companies_database import enhanced_indian_companies_db
        
        db_results = []
        
        # Search all companies in our database
        for symbol, company in enhanced_indian_companies_db.companies.items():
            # Check symbol
            if any(term.lower() in symbol.lower() for term in ['ztech', 'zentech']):
                print(f"✅ Database Found (Symbol): {symbol}")
                print(f"   Company: {company.name}")
                print(f"   Exchange: {company.exchange.value}")
                
                db_results.append({
                    'symbol': symbol,
                    'company_name': company.name,
                    'exchange': company.exchange.value,
                    'sector': company.sector,
                    'industry': company.industry,
                    'source': 'enhanced_db'
                })
            
            # Check company name
            elif any(term.lower() in company.name.lower() for term in ['ztech', 'zentech']):
                print(f"✅ Database Found (Name): {symbol}")
                print(f"   Company: {company.name}")
                print(f"   Exchange: {company.exchange.value}")
                
                db_results.append({
                    'symbol': symbol,
                    'company_name': company.name,
                    'exchange': company.exchange.value,
                    'sector': company.sector,
                    'industry': company.industry,
                    'source': 'enhanced_db'
                })
            
            # Check aliases
            elif any(any(term.lower() in alias.lower() for term in ['ztech', 'zentech']) 
                    for alias in company.aliases):
                print(f"✅ Database Found (Alias): {symbol}")
                print(f"   Company: {company.name}")
                print(f"   Exchange: {company.exchange.value}")
                print(f"   Matching Aliases: {[a for a in company.aliases if any(term.lower() in a.lower() for term in ['ztech', 'zentech'])]}")
                
                db_results.append({
                    'symbol': symbol,
                    'company_name': company.name,
                    'exchange': company.exchange.value,
                    'sector': company.sector,
                    'industry': company.industry,
                    'source': 'enhanced_db'
                })
        
        return db_results
    
    async def comprehensive_search(self) -> Dict[str, Any]:
        """Perform comprehensive search across all sources"""
        print("🚀 ZTECH India Comprehensive Search")
        print("=" * 60)
        
        results = {
            'search_timestamp': datetime.now().isoformat(),
            'search_variations': self.search_variations,
            'yfinance_results': [],
            'nse_results': [],
            'bse_results': [],
            'database_results': [],
            'summary': {}
        }
        
        # Search Yahoo Finance
        results['yfinance_results'] = self.test_yfinance_symbols()
        
        # Search NSE
        results['nse_results'] = await self.search_nse_symbols()
        
        # Search BSE
        results['bse_results'] = self.search_bse_symbols()
        
        # Search our database
        results['database_results'] = self.search_company_databases()
        
        # Generate summary
        total_found = (len(results['yfinance_results']) + 
                      len(results['nse_results']) + 
                      len(results['bse_results']) + 
                      len(results['database_results']))
        
        results['summary'] = {
            'total_variations_searched': len(self.search_variations),
            'total_symbols_found': total_found,
            'yfinance_symbols': len(results['yfinance_results']),
            'nse_symbols': len(results['nse_results']),
            'bse_symbols': len(results['bse_results']),
            'database_symbols': len(results['database_results']),
            'data_available_count': len([r for r in results['yfinance_results'] if r.get('data_available', False)])
        }
        
        return results

async def test_specific_ztech_india_symbols():
    """Test specific ZTECH India symbol variations"""
    print("\n🎯 Testing Specific ZTECH India Symbols")
    print("-" * 50)
    
    specific_symbols = [
        "ZTECHINDIA.NS",
        "ZTECHINDIA.BO", 
        "ZENTECHINDIA.NS",
        "ZENTECHINDIA.BO",
        "ZTECH.NS",
        "ZTECH.BO",
        "ZENTECH.NS", 
        "ZENTECH.BO"
    ]
    
    found_symbols = []
    
    for symbol in specific_symbols:
        try:
            print(f"\nTesting: {symbol}")
            ticker = yf.Ticker(symbol)
            
            # Get info
            info = ticker.info
            
            if info and len(info) > 5:
                company_name = info.get('longName', info.get('shortName', 'N/A'))
                print(f"✅ Valid: {company_name}")
                
                # Get recent data
                hist = ticker.history(period="5d")
                if not hist.empty:
                    latest = hist.iloc[-1]
                    print(f"   Price: ₹{latest['Close']:.2f}")
                    print(f"   Volume: {latest['Volume']:,}")
                    
                    found_symbols.append({
                        'symbol': symbol,
                        'company_name': company_name,
                        'price': float(latest['Close']),
                        'volume': int(latest['Volume'])
                    })
                else:
                    print(f"   ⚠️ No trading data")
            else:
                print(f"❌ Invalid or no data")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return found_symbols

async def main():
    """Run comprehensive ZTECH India search"""
    print("🔍 ZTECH India Comprehensive Analysis")
    print("=" * 60)
    
    # Test specific symbols first
    specific_results = await test_specific_ztech_india_symbols()
    
    # Run comprehensive search
    async with ZTechIndiaSearchEngine() as search_engine:
        comprehensive_results = await search_engine.comprehensive_search()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("📋 FINAL SUMMARY")
    print("=" * 60)
    
    print(f"🔍 Search Variations Tested: {comprehensive_results['summary']['total_variations_searched']}")
    print(f"✅ Total Symbols Found: {comprehensive_results['summary']['total_symbols_found']}")
    print(f"📊 Yahoo Finance Results: {comprehensive_results['summary']['yfinance_symbols']}")
    print(f"🏛️ NSE Results: {comprehensive_results['summary']['nse_symbols']}")
    print(f"🏛️ BSE Results: {comprehensive_results['summary']['bse_symbols']}")
    print(f"💾 Database Results: {comprehensive_results['summary']['database_symbols']}")
    print(f"📈 With Trading Data: {comprehensive_results['summary']['data_available_count']}")
    
    # Show all found symbols with data
    print(f"\n📈 SYMBOLS WITH ACTIVE TRADING DATA:")
    print("-" * 40)
    
    active_symbols = [r for r in comprehensive_results['yfinance_results'] if r.get('data_available', False)]
    
    if active_symbols:
        for symbol_data in active_symbols:
            print(f"✅ {symbol_data['symbol']}")
            print(f"   Company: {symbol_data['company_name']}")
            print(f"   Exchange: {symbol_data['exchange']}")
            print(f"   Price: ₹{symbol_data.get('latest_price', 'N/A')}")
            print(f"   Volume: {symbol_data.get('volume', 'N/A'):,}")
            print()
    else:
        print("❌ No symbols found with active trading data")
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    if active_symbols:
        print("✅ Found active ZTECH symbols - integrate these into your system")
        print("✅ Use the symbols with highest volume for primary data")
        print("✅ Set up monitoring for all found symbols")
    else:
        print("⚠️ No active ZTECH India symbols found")
        print("💡 Consider using ZTECH.NS (Zentech Systems) as primary")
        print("💡 Monitor NSE Emerge platform for new listings")
        print("💡 Set up alerts for ZTECH India IPO announcements")
    
    # Save results
    with open('ztech_india_search_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: ztech_india_search_results.json")

if __name__ == "__main__":
    asyncio.run(main())
