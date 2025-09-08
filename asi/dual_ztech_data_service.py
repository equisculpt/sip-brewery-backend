"""
Dual ZTECH Data Service
Comprehensive data service for both Z-Tech (India) Limited and Zentech Systems Limited
Provides unified API for institutional-grade financial analysis
"""

import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZTechCompanyType(Enum):
    """Enum for different ZTECH companies"""
    ZTECH_INDIA = "ztech_india"  # Z-Tech (India) Limited - NSE Main
    ZENTECH_SYSTEMS = "zentech_systems"  # Zentech Systems Limited - NSE Emerge

@dataclass
class ZTechCompanyInfo:
    """Company information for ZTECH entities"""
    company_type: ZTechCompanyType
    symbol: str
    yahoo_symbol: str
    name: str
    exchange: str
    sector: str
    market_cap_category: str
    listing_date: str
    isin: str
    aliases: List[str]
    keywords: List[str]

@dataclass
class ZTechLiveData:
    """Live market data for ZTECH companies"""
    company_type: ZTechCompanyType
    symbol: str
    company_name: str
    current_price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float]
    day_high: Optional[float]
    day_low: Optional[float]
    open_price: Optional[float]
    previous_close: Optional[float]
    exchange: str
    currency: str
    market_state: str
    last_updated: datetime
    data_source: str

@dataclass
class ZTechOHLCData:
    """OHLC data for ZTECH companies"""
    company_type: ZTechCompanyType
    symbol: str
    timeframe: str
    period: str
    data: pd.DataFrame
    technical_indicators: Dict[str, Any]
    last_updated: datetime

class DualZTechDataService:
    """
    Comprehensive data service for both ZTECH companies
    Handles Z-Tech (India) Limited and Zentech Systems Limited
    """
    
    def __init__(self):
        """Initialize the dual ZTECH data service"""
        self.companies = {
            ZTechCompanyType.ZTECH_INDIA: ZTechCompanyInfo(
                company_type=ZTechCompanyType.ZTECH_INDIA,
                symbol="ZTECH_INDIA",
                yahoo_symbol="ZTECH.NS",
                name="Z-Tech (India) Limited",
                exchange="NSE_MAIN",
                sector="Information Technology",
                market_cap_category="MID_CAP",
                listing_date="2020-01-01",
                isin="INE0QFO01011",
                aliases=["ztech india", "z-tech india", "ztech india limited", "z-tech india limited", "ztech.ns"],
                keywords=["it", "technology", "india", "ztech", "z-tech", "main board"]
            ),
            ZTechCompanyType.ZENTECH_SYSTEMS: ZTechCompanyInfo(
                company_type=ZTechCompanyType.ZENTECH_SYSTEMS,
                symbol="ZTECH",
                yahoo_symbol="ZTECH.NS",  # Note: Same Yahoo symbol, different companies
                name="Zentech Systems Limited",
                exchange="NSE_EMERGE",
                sector="Information Technology",
                market_cap_category="SMALL_CAP",
                listing_date="2023-06-15",
                isin="INE0QFO01010",
                aliases=["ztech", "zentech", "zentech systems", "ztech systems", "emerge ztech"],
                keywords=["it", "software", "technology", "systems", "zentech", "emerge"]
            )
        }
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 1.0  # 1 second between requests
        
    async def _rate_limit(self, source: str):
        """Rate limiting for API calls"""
        current_time = datetime.now().timestamp()
        if source in self.last_request_time:
            time_diff = current_time - self.last_request_time[source]
            if time_diff < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_diff)
        self.last_request_time[source] = datetime.now().timestamp()

    def resolve_company(self, query: str) -> Optional[ZTechCompanyType]:
        """
        Resolve company from query string
        
        Args:
            query: Search query (e.g., "ztech india", "zentech systems")
            
        Returns:
            ZTechCompanyType or None if not found
        """
        query_lower = query.lower().strip()
        
        # Direct matches for Z-Tech India
        ztech_india_terms = [
            "ztech india", "z-tech india", "ztech india limited", 
            "z-tech india limited", "ztech.ns", "main board ztech"
        ]
        
        # Direct matches for Zentech Systems
        zentech_terms = [
            "zentech", "zentech systems", "ztech systems", 
            "emerge ztech", "ztech emerge"
        ]
        
        # Check for Z-Tech India matches
        for term in ztech_india_terms:
            if term in query_lower:
                return ZTechCompanyType.ZTECH_INDIA
                
        # Check for Zentech Systems matches
        for term in zentech_terms:
            if term in query_lower:
                return ZTechCompanyType.ZENTECH_SYSTEMS
                
        # Default to Z-Tech India if just "ztech" is mentioned
        if "ztech" in query_lower and "emerge" not in query_lower:
            return ZTechCompanyType.ZTECH_INDIA
            
        return None

    async def get_live_data(self, company_type: ZTechCompanyType) -> Optional[ZTechLiveData]:
        """
        Get live market data for specified ZTECH company
        
        Args:
            company_type: Type of ZTECH company
            
        Returns:
            ZTechLiveData or None if failed
        """
        try:
            await self._rate_limit('yahoo')
            
            company_info = self.companies[company_type]
            ticker = yf.Ticker(company_info.yahoo_symbol)
            info = ticker.info
            
            # Get current price data
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            change = info.get('regularMarketChange', 0)
            change_percent = info.get('regularMarketChangePercent', 0)
            volume = info.get('regularMarketVolume', 0)
            
            return ZTechLiveData(
                company_type=company_type,
                symbol=company_info.symbol,
                company_name=company_info.name,
                current_price=float(current_price),
                change=float(change),
                change_percent=float(change_percent * 100) if change_percent else 0,
                volume=int(volume),
                market_cap=info.get('marketCap'),
                day_high=info.get('dayHigh'),
                day_low=info.get('dayLow'),
                open_price=info.get('regularMarketOpen'),
                previous_close=info.get('regularMarketPreviousClose'),
                exchange=company_info.exchange,
                currency="INR",
                market_state=info.get('marketState', 'UNKNOWN'),
                last_updated=datetime.now(),
                data_source="yahoo_finance"
            )
            
        except Exception as e:
            logger.error(f"Failed to get live data for {company_type.value}: {e}")
            return None

    async def get_ohlc_data(self, 
                           company_type: ZTechCompanyType,
                           period: str = "1mo",
                           interval: str = "1d") -> Optional[ZTechOHLCData]:
        """
        Get OHLC data with technical indicators
        
        Args:
            company_type: Type of ZTECH company
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            ZTechOHLCData or None if failed
        """
        try:
            await self._rate_limit('yahoo')
            
            company_info = self.companies[company_type]
            ticker = yf.Ticker(company_info.yahoo_symbol)
            
            # Get historical data
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                logger.warning(f"No OHLC data found for {company_type.value}")
                return None
                
            # Calculate technical indicators
            technical_indicators = self._calculate_technical_indicators(hist)
            
            return ZTechOHLCData(
                company_type=company_type,
                symbol=company_info.symbol,
                timeframe=interval,
                period=period,
                data=hist,
                technical_indicators=technical_indicators,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to get OHLC data for {company_type.value}: {e}")
            return None

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators from OHLC data"""
        try:
            indicators = {}
            
            # Simple Moving Averages
            indicators['sma_20'] = df['Close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
            indicators['sma_50'] = df['Close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
            
            # RSI
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # Bollinger Bands
            if len(df) >= 20:
                sma_20 = df['Close'].rolling(window=20).mean()
                std_20 = df['Close'].rolling(window=20).std()
                indicators['bollinger_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
                indicators['bollinger_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
                indicators['bollinger_middle'] = sma_20.iloc[-1]
            
            # Volume analysis
            indicators['avg_volume'] = df['Volume'].mean()
            indicators['volume_ratio'] = df['Volume'].iloc[-1] / indicators['avg_volume'] if indicators['avg_volume'] > 0 else 1
            
            # Price levels
            indicators['support_level'] = df['Low'].min()
            indicators['resistance_level'] = df['High'].max()
            indicators['current_price'] = df['Close'].iloc[-1]
            
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators: {e}")
            return {}

    async def get_comprehensive_data(self, company_type: ZTechCompanyType) -> Dict[str, Any]:
        """
        Get comprehensive data for specified ZTECH company
        
        Args:
            company_type: Type of ZTECH company
            
        Returns:
            Comprehensive data dictionary
        """
        try:
            # Get live data and OHLC data concurrently
            live_data_task = self.get_live_data(company_type)
            ohlc_data_task = self.get_ohlc_data(company_type)
            
            live_data, ohlc_data = await asyncio.gather(
                live_data_task, ohlc_data_task, return_exceptions=True
            )
            
            company_info = self.companies[company_type]
            
            result = {
                "company_info": {
                    "type": company_type.value,
                    "symbol": company_info.symbol,
                    "name": company_info.name,
                    "exchange": company_info.exchange,
                    "sector": company_info.sector,
                    "market_cap_category": company_info.market_cap_category,
                    "listing_date": company_info.listing_date,
                    "isin": company_info.isin,
                    "aliases": company_info.aliases,
                    "keywords": company_info.keywords
                },
                "live_data": live_data.__dict__ if live_data else None,
                "ohlc_data": {
                    "timeframe": ohlc_data.timeframe,
                    "period": ohlc_data.period,
                    "data_points": len(ohlc_data.data),
                    "technical_indicators": ohlc_data.technical_indicators,
                    "last_updated": ohlc_data.last_updated.isoformat()
                } if ohlc_data else None,
                "status": "success" if (live_data or ohlc_data) else "failed",
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive data for {company_type.value}: {e}")
            return {
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }

    async def get_all_companies_data(self) -> Dict[str, Any]:
        """
        Get data for all ZTECH companies
        
        Returns:
            Dictionary with data for all companies
        """
        try:
            # Get data for both companies concurrently
            ztech_india_task = self.get_comprehensive_data(ZTechCompanyType.ZTECH_INDIA)
            zentech_systems_task = self.get_comprehensive_data(ZTechCompanyType.ZENTECH_SYSTEMS)
            
            ztech_india_data, zentech_systems_data = await asyncio.gather(
                ztech_india_task, zentech_systems_task, return_exceptions=True
            )
            
            return {
                "ztech_india": ztech_india_data,
                "zentech_systems": zentech_systems_data,
                "comparison": self._generate_comparison(ztech_india_data, zentech_systems_data),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get all companies data: {e}")
            return {
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }

    def _generate_comparison(self, ztech_india_data: Dict, zentech_systems_data: Dict) -> Dict[str, Any]:
        """Generate comparison between both companies"""
        try:
            comparison = {
                "price_comparison": {},
                "volume_comparison": {},
                "market_cap_comparison": {},
                "exchange_comparison": {},
                "recommendation": ""
            }
            
            # Extract live data
            ztech_live = ztech_india_data.get('live_data', {})
            zentech_live = zentech_systems_data.get('live_data', {})
            
            if ztech_live and zentech_live:
                # Price comparison
                ztech_price = ztech_live.get('current_price', 0)
                zentech_price = zentech_live.get('current_price', 0)
                
                comparison['price_comparison'] = {
                    "ztech_india_price": ztech_price,
                    "zentech_systems_price": zentech_price,
                    "price_ratio": ztech_price / zentech_price if zentech_price > 0 else 0,
                    "higher_priced": "Z-Tech India" if ztech_price > zentech_price else "Zentech Systems"
                }
                
                # Volume comparison
                ztech_volume = ztech_live.get('volume', 0)
                zentech_volume = zentech_live.get('volume', 0)
                
                comparison['volume_comparison'] = {
                    "ztech_india_volume": ztech_volume,
                    "zentech_systems_volume": zentech_volume,
                    "volume_ratio": ztech_volume / zentech_volume if zentech_volume > 0 else 0,
                    "higher_volume": "Z-Tech India" if ztech_volume > zentech_volume else "Zentech Systems"
                }
                
                # Generate recommendation
                if ztech_price > zentech_price and ztech_volume < zentech_volume:
                    comparison['recommendation'] = "Z-Tech India: Higher price, lower volume - Established company. Zentech Systems: Lower price, higher volume - Growth potential."
                elif ztech_price < zentech_price:
                    comparison['recommendation'] = "Unusual: Z-Tech India trading lower than Zentech Systems - Investigate further."
                else:
                    comparison['recommendation'] = "Both companies showing similar patterns - Consider diversification."
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to generate comparison: {e}")
            return {"error": str(e)}

# API Service Wrapper
class DualZTechAPIService:
    """API service wrapper for dual ZTECH data service"""
    
    def __init__(self):
        self.data_service = DualZTechDataService()
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process natural language query for ZTECH companies
        
        Args:
            query: Natural language query
            
        Returns:
            Response dictionary
        """
        try:
            # Resolve company from query
            company_type = self.data_service.resolve_company(query)
            
            if not company_type:
                return {
                    "error": "Could not identify ZTECH company from query",
                    "suggestion": "Try 'ztech india' or 'zentech systems'",
                    "available_companies": [
                        "Z-Tech (India) Limited - NSE Main Board",
                        "Zentech Systems Limited - NSE Emerge"
                    ],
                    "status": "failed"
                }
            
            # Check if user wants all companies
            if "all" in query.lower() or "both" in query.lower() or "compare" in query.lower():
                return await self.data_service.get_all_companies_data()
            
            # Get comprehensive data for resolved company
            return await self.data_service.get_comprehensive_data(company_type)
            
        except Exception as e:
            logger.error(f"Failed to process query '{query}': {e}")
            return {
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }

# Test function
async def test_dual_ztech_service():
    """Test the dual ZTECH data service"""
    print("🚀 Testing Dual ZTECH Data Service")
    print("=" * 50)
    
    service = DualZTechAPIService()
    
    # Test queries
    test_queries = [
        "ztech india live price",
        "zentech systems ohlc data",
        "compare all ztech companies",
        "emerge ztech technical analysis"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing Query: '{query}'")
        print("-" * 30)
        
        result = await service.process_query(query)
        
        if result.get('status') == 'success':
            print("✅ Success!")
            if 'company_info' in result:
                company_info = result['company_info']
                print(f"   Company: {company_info['name']}")
                print(f"   Exchange: {company_info['exchange']}")
            
            if 'live_data' in result and result['live_data']:
                live_data = result['live_data']
                print(f"   Price: ₹{live_data['current_price']}")
                print(f"   Volume: {live_data['volume']:,}")
        else:
            print("❌ Failed!")
            print(f"   Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(test_dual_ztech_service())
