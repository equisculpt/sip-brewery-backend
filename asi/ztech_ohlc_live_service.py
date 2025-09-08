"""
ZTECH (Zentech Systems) OHLC and Live Price Data Service
Comprehensive real-time and historical data for ZTECH using multiple sources
"""
import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeFrame(Enum):
    """Time frame for OHLC data"""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"

class Period(Enum):
    """Period for historical data"""
    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"
    FIVE_YEARS = "5y"
    TEN_YEARS = "10y"
    MAX = "max"

@dataclass
class OHLCData:
    """OHLC data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "adj_close": self.adj_close
        }

@dataclass
class LivePriceData:
    """Live price data structure"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float
    open: float
    previous_close: float
    market_cap: Optional[float]
    timestamp: datetime
    source: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "change": self.change,
            "change_percent": self.change_percent,
            "volume": self.volume,
            "high": self.high,
            "low": self.low,
            "open": self.open,
            "previous_close": self.previous_close,
            "market_cap": self.market_cap,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }

class ZTechDataService:
    """Comprehensive ZTECH data service"""
    
    def __init__(self):
        self.symbol_nse = "ZTECH.NS"
        self.symbol_bse = "543654.BO"  # BSE code
        self.symbol_base = "ZTECH"
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def get_live_price_yfinance(self) -> Optional[LivePriceData]:
        """Get live price using yfinance"""
        try:
            ticker = yf.Ticker(self.symbol_nse)
            info = ticker.info
            
            # Get current price data
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            previous_close = info.get('previousClose', info.get('regularMarketPreviousClose', 0))
            
            # Calculate change
            change = current_price - previous_close if current_price and previous_close else 0
            change_percent = (change / previous_close * 100) if previous_close else 0
            
            return LivePriceData(
                symbol=self.symbol_base,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=info.get('volume', info.get('regularMarketVolume', 0)),
                high=info.get('dayHigh', info.get('regularMarketDayHigh', 0)),
                low=info.get('dayLow', info.get('regularMarketDayLow', 0)),
                open=info.get('open', info.get('regularMarketOpen', 0)),
                previous_close=previous_close,
                market_cap=info.get('marketCap'),
                timestamp=datetime.now(),
                source="yfinance"
            )
            
        except Exception as e:
            logger.error(f"Error fetching live price from yfinance: {e}")
            return None
    
    async def get_live_price_nse_api(self) -> Optional[LivePriceData]:
        """Get live price from NSE API"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.nseindia.com/'
            }
            
            # Try NSE Emerge API first
            url = f"https://www.nseindia.com/api/emerge-quote-equity?symbol={self.symbol_base}"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    price_info = data.get('priceInfo', {})
                    
                    current_price = float(price_info.get('lastPrice', 0))
                    change = float(price_info.get('change', 0))
                    change_percent = float(price_info.get('pChange', 0))
                    
                    return LivePriceData(
                        symbol=self.symbol_base,
                        price=current_price,
                        change=change,
                        change_percent=change_percent,
                        volume=int(data.get('securityWiseDP', {}).get('quantityTraded', 0)),
                        high=float(price_info.get('intraDayHighLow', {}).get('max', 0)),
                        low=float(price_info.get('intraDayHighLow', {}).get('min', 0)),
                        open=float(price_info.get('open', 0)),
                        previous_close=float(price_info.get('previousClose', 0)),
                        market_cap=None,  # Not available in NSE API
                        timestamp=datetime.now(),
                        source="nse_emerge_api"
                    )
                    
        except Exception as e:
            logger.error(f"Error fetching live price from NSE API: {e}")
            
        return None
    
    def get_ohlc_data(self, timeframe: TimeFrame = TimeFrame.ONE_DAY, 
                      period: Period = Period.ONE_MONTH) -> List[OHLCData]:
        """Get OHLC data using yfinance"""
        try:
            # Download data
            df = yf.download(
                self.symbol_nse, 
                interval=timeframe.value, 
                period=period.value, 
                progress=False
            )
            
            if df.empty:
                logger.warning(f"No OHLC data returned for {self.symbol_nse}")
                return []
            
            # Reset index to get timestamp as column
            df = df.reset_index()
            
            ohlc_data = []
            for _, row in df.iterrows():
                ohlc_data.append(OHLCData(
                    timestamp=row['Datetime'] if 'Datetime' in row else row['Date'],
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']),
                    adj_close=float(row['Adj Close']) if 'Adj Close' in row else None
                ))
            
            return ohlc_data
            
        except Exception as e:
            logger.error(f"Error fetching OHLC data: {e}")
            return []
    
    def get_intraday_ohlc(self, timeframe: TimeFrame = TimeFrame.ONE_MINUTE) -> List[OHLCData]:
        """Get intraday OHLC data for current trading day"""
        return self.get_ohlc_data(timeframe=timeframe, period=Period.ONE_DAY)
    
    def get_historical_ohlc(self, days: int = 30) -> List[OHLCData]:
        """Get historical daily OHLC data"""
        if days <= 5:
            period = Period.FIVE_DAYS
        elif days <= 30:
            period = Period.ONE_MONTH
        elif days <= 90:
            period = Period.THREE_MONTHS
        elif days <= 180:
            period = Period.SIX_MONTHS
        elif days <= 365:
            period = Period.ONE_YEAR
        else:
            period = Period.TWO_YEARS
            
        return self.get_ohlc_data(timeframe=TimeFrame.ONE_DAY, period=period)
    
    async def get_comprehensive_data(self) -> Dict[str, Any]:
        """Get comprehensive ZTECH data including live price and OHLC"""
        result = {
            "symbol": self.symbol_base,
            "company_name": "Zentech Systems Limited",
            "exchange": "NSE_EMERGE",
            "live_price": None,
            "intraday_ohlc": [],
            "daily_ohlc": [],
            "technical_indicators": {},
            "timestamp": datetime.now().isoformat(),
            "data_sources": []
        }
        
        try:
            # Get live price from multiple sources
            live_price_yf = self.get_live_price_yfinance()
            live_price_nse = await self.get_live_price_nse_api()
            
            # Use NSE API if available, otherwise yfinance
            if live_price_nse:
                result["live_price"] = live_price_nse.to_dict()
                result["data_sources"].append("nse_emerge_api")
            elif live_price_yf:
                result["live_price"] = live_price_yf.to_dict()
                result["data_sources"].append("yfinance")
            
            # Get OHLC data
            intraday_data = self.get_intraday_ohlc(TimeFrame.ONE_MINUTE)
            daily_data = self.get_historical_ohlc(30)
            
            result["intraday_ohlc"] = [data.to_dict() for data in intraday_data[-50:]]  # Last 50 minutes
            result["daily_ohlc"] = [data.to_dict() for data in daily_data]
            
            if intraday_data or daily_data:
                result["data_sources"].append("yfinance_ohlc")
            
            # Calculate technical indicators
            if daily_data:
                result["technical_indicators"] = self.calculate_technical_indicators(daily_data)
            
        except Exception as e:
            logger.error(f"Error getting comprehensive data: {e}")
            result["error"] = str(e)
        
        return result
    
    def calculate_technical_indicators(self, ohlc_data: List[OHLCData]) -> Dict[str, Any]:
        """Calculate basic technical indicators"""
        if len(ohlc_data) < 20:
            return {}
        
        try:
            # Convert to pandas for calculations
            df = pd.DataFrame([data.to_dict() for data in ohlc_data])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            indicators = {}
            
            # Simple Moving Averages
            if len(df) >= 20:
                indicators['sma_20'] = df['close'].rolling(window=20).mean().iloc[-1]
            if len(df) >= 50:
                indicators['sma_50'] = df['close'].rolling(window=50).mean().iloc[-1]
            
            # RSI (Relative Strength Index)
            if len(df) >= 14:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # Bollinger Bands
            if len(df) >= 20:
                sma_20 = df['close'].rolling(window=20).mean()
                std_20 = df['close'].rolling(window=20).std()
                indicators['bollinger_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
                indicators['bollinger_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
                indicators['bollinger_middle'] = sma_20.iloc[-1]
            
            # Volume indicators
            indicators['avg_volume_20'] = df['volume'].rolling(window=20).mean().iloc[-1]
            indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['avg_volume_20']
            
            # Price levels
            indicators['52_week_high'] = df['high'].max()
            indicators['52_week_low'] = df['low'].min()
            indicators['current_price'] = df['close'].iloc[-1]
            
            # Convert numpy types to Python types for JSON serialization
            for key, value in indicators.items():
                if isinstance(value, (np.integer, np.floating)):
                    indicators[key] = float(value)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}

class ZTechAPIService:
    """API service for ZTECH data endpoints"""
    
    def __init__(self):
        self.data_service = ZTechDataService()
    
    async def get_live_price(self) -> Dict[str, Any]:
        """API endpoint for live price"""
        async with self.data_service as service:
            # Try NSE API first
            live_price = await service.get_live_price_nse_api()
            
            # Fallback to yfinance
            if not live_price:
                live_price = service.get_live_price_yfinance()
            
            if live_price:
                return {
                    "status": "success",
                    "data": live_price.to_dict(),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "message": "Unable to fetch live price data",
                    "timestamp": datetime.now().isoformat()
                }
    
    async def get_ohlc(self, timeframe: str = "1d", period: str = "1mo") -> Dict[str, Any]:
        """API endpoint for OHLC data"""
        try:
            tf = TimeFrame(timeframe)
            pd = Period(period)
            
            async with self.data_service as service:
                ohlc_data = service.get_ohlc_data(timeframe=tf, period=pd)
                
                return {
                    "status": "success",
                    "data": {
                        "symbol": "ZTECH",
                        "timeframe": timeframe,
                        "period": period,
                        "count": len(ohlc_data),
                        "ohlc": [data.to_dict() for data in ohlc_data]
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
        except ValueError as e:
            return {
                "status": "error",
                "message": f"Invalid timeframe or period: {e}",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_comprehensive_data(self) -> Dict[str, Any]:
        """API endpoint for comprehensive ZTECH data"""
        async with self.data_service as service:
            data = await service.get_comprehensive_data()
            
            return {
                "status": "success",
                "data": data,
                "timestamp": datetime.now().isoformat()
            }

# Global instance
ztech_api_service = ZTechAPIService()

# Convenience functions for direct usage
async def get_ztech_live_price() -> Dict[str, Any]:
    """Get ZTECH live price"""
    return await ztech_api_service.get_live_price()

async def get_ztech_ohlc(timeframe: str = "1d", period: str = "1mo") -> Dict[str, Any]:
    """Get ZTECH OHLC data"""
    return await ztech_api_service.get_ohlc(timeframe, period)

async def get_ztech_comprehensive() -> Dict[str, Any]:
    """Get comprehensive ZTECH data"""
    return await ztech_api_service.get_comprehensive_data()

# Test function
async def test_ztech_data():
    """Test ZTECH data fetching"""
    print("=== Testing ZTECH Data Service ===")
    
    # Test live price
    print("\n1. Testing Live Price...")
    live_data = await get_ztech_live_price()
    print(f"Live Price Status: {live_data['status']}")
    if live_data['status'] == 'success':
        price_data = live_data['data']
        print(f"Current Price: ₹{price_data['price']}")
        print(f"Change: ₹{price_data['change']} ({price_data['change_percent']:.2f}%)")
        print(f"Volume: {price_data['volume']:,}")
        print(f"Day Range: ₹{price_data['low']} - ₹{price_data['high']}")
    
    # Test OHLC data
    print("\n2. Testing OHLC Data...")
    ohlc_data = await get_ztech_ohlc("1d", "1mo")
    print(f"OHLC Status: {ohlc_data['status']}")
    if ohlc_data['status'] == 'success':
        ohlc_info = ohlc_data['data']
        print(f"OHLC Records: {ohlc_info['count']}")
        if ohlc_info['ohlc']:
            latest = ohlc_info['ohlc'][-1]
            print(f"Latest OHLC: O:{latest['open']} H:{latest['high']} L:{latest['low']} C:{latest['close']}")
    
    # Test comprehensive data
    print("\n3. Testing Comprehensive Data...")
    comp_data = await get_ztech_comprehensive()
    print(f"Comprehensive Status: {comp_data['status']}")
    if comp_data['status'] == 'success':
        data = comp_data['data']
        print(f"Data Sources: {data['data_sources']}")
        print(f"Intraday Records: {len(data['intraday_ohlc'])}")
        print(f"Daily Records: {len(data['daily_ohlc'])}")
        
        if data['technical_indicators']:
            indicators = data['technical_indicators']
            print(f"Technical Indicators:")
            for key, value in indicators.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_ztech_data())
