"""
Universal Financial Search Engine
Google-level real-time search for ANY company, mutual fund, or financial instrument
Extends ZTECH capabilities to entire Indian financial market
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re

# Import existing components
from enhanced_indian_companies_database import EnhancedIndianCompaniesDatabase, ExchangeType, MarketCapCategory
from json_database_manager import JSONDatabaseManager, SearchResult
from dual_ztech_data_service import DualZTechAPIService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialInstrumentType(Enum):
    """Types of financial instruments"""
    STOCK = "stock"
    MUTUAL_FUND = "mutual_fund"
    ETF = "etf"
    BOND = "bond"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    INDEX = "index"
    DERIVATIVE = "derivative"

class SearchCategory(Enum):
    """Search categories for suggestions"""
    COMPANY = "company"
    PRICE = "price"
    ANALYSIS = "analysis"
    NEWS = "news"
    COMPARISON = "comparison"
    TECHNICAL = "technical"
    MUTUAL_FUND = "mutual_fund"
    PORTFOLIO = "portfolio"
    SECTOR = "sector"

@dataclass
class UniversalSearchSuggestion:
    """Universal search suggestion with financial data"""
    query: str
    category: SearchCategory
    instrument_type: FinancialInstrumentType
    symbol: str
    name: str
    description: str
    instant_data: Dict[str, Any]
    confidence: float
    exchange: str
    sector: Optional[str] = None
    market_cap: Optional[str] = None

@dataclass
class UniversalSearchResult:
    """Universal search result with comprehensive data"""
    query: str
    suggestions: List[UniversalSearchSuggestion]
    live_data: Dict[str, Any]
    total_time_ms: float
    timestamp: datetime
    instruments_found: int
    categories_covered: List[str]

class UniversalFinancialSearchEngine:
    """
    Universal Google-level search engine for entire Indian financial market
    Supports stocks, mutual funds, ETFs, bonds, commodities, and more
    """
    
    def __init__(self):
        """Initialize the universal search engine"""
        self.enhanced_db = enhanced_indian_companies_db
        
        # Cache for performance
        self.cache = {}
        self.cache_ttl = 30  # 30 seconds
        
        # Background data storage
        self.background_data = {}
        self.last_update = {}
        self.update_interval = 15  # 15 seconds for broader coverage
        
        # Popular mutual funds and ETFs
        self.mutual_funds = {
            "sbi bluechip": {"symbol": "SBI_BLUECHIP", "name": "SBI Blue Chip Fund", "type": "Large Cap"},
            "hdfc top 100": {"symbol": "HDFC_TOP100", "name": "HDFC Top 100 Fund", "type": "Large Cap"},
            "icici pru bluechip": {"symbol": "ICICI_BLUECHIP", "name": "ICICI Prudential Blue Chip Fund", "type": "Large Cap"},
            "axis bluechip": {"symbol": "AXIS_BLUECHIP", "name": "Axis Blue Chip Fund", "type": "Large Cap"},
            "mirae asset large cap": {"symbol": "MIRAE_LARGECAP", "name": "Mirae Asset Large Cap Fund", "type": "Large Cap"},
            "parag parikh flexi cap": {"symbol": "PPFAS_FLEXICAP", "name": "Parag Parikh Flexi Cap Fund", "type": "Flexi Cap"},
            "sbi small cap": {"symbol": "SBI_SMALLCAP", "name": "SBI Small Cap Fund", "type": "Small Cap"},
            "hdfc mid cap": {"symbol": "HDFC_MIDCAP", "name": "HDFC Mid-Cap Opportunities Fund", "type": "Mid Cap"},
            "kotak emerging equity": {"symbol": "KOTAK_EMERGING", "name": "Kotak Emerging Equity Fund", "type": "Mid Cap"},
            "nippon india small cap": {"symbol": "NIPPON_SMALLCAP", "name": "Nippon India Small Cap Fund", "type": "Small Cap"}
        }
        
        # Popular ETFs
        self.etfs = {
            "nifty bees": {"symbol": "NIFTYBEES.NS", "name": "Nippon India ETF Nifty BeES", "index": "NIFTY 50"},
            "bank bees": {"symbol": "BANKBEES.NS", "name": "Nippon India ETF Bank BeES", "index": "NIFTY Bank"},
            "junior bees": {"symbol": "JUNIORBEES.NS", "name": "Nippon India ETF Junior BeES", "index": "NIFTY Next 50"},
            "gold bees": {"symbol": "GOLDBEES.NS", "name": "Nippon India ETF Gold BeES", "index": "Gold"},
            "liquid bees": {"symbol": "LIQUIDBEES.NS", "name": "Nippon India ETF Liquid BeES", "index": "Liquid"},
            "icicipru nifty": {"symbol": "ICICIPRU.NS", "name": "ICICI Prudential Nifty ETF", "index": "NIFTY 50"},
            "hdfcnifty": {"symbol": "HDFCNIFTY.NS", "name": "HDFC Nifty 50 ETF", "index": "NIFTY 50"},
            "sbietf": {"symbol": "SBIETF.NS", "name": "SBI ETF Nifty 50", "index": "NIFTY 50"}
        }
        
        # Major indices
        self.indices = {
            "nifty": {"symbol": "^NSEI", "name": "NIFTY 50", "description": "NSE benchmark index"},
            "sensex": {"symbol": "^BSESN", "name": "S&P BSE SENSEX", "description": "BSE benchmark index"},
            "bank nifty": {"symbol": "^NSEBANK", "name": "NIFTY Bank", "description": "Banking sector index"},
            "nifty it": {"symbol": "^NSEIT", "name": "NIFTY IT", "description": "IT sector index"},
            "nifty auto": {"symbol": "^NSEAUTO", "name": "NIFTY Auto", "description": "Auto sector index"},
            "nifty pharma": {"symbol": "^NSEPHARMA", "name": "NIFTY Pharma", "description": "Pharma sector index"},
            "nifty fmcg": {"symbol": "^NSEFMCG", "name": "NIFTY FMCG", "description": "FMCG sector index"},
            "nifty metal": {"symbol": "^NSEMETAL", "name": "NIFTY Metal", "description": "Metal sector index"}
        }
        
        # Sector mappings
        self.sector_keywords = {
            "banking": ["bank", "banking", "financial", "finance"],
            "it": ["it", "technology", "tech", "software", "infotech", "it sector"],
            "pharma": ["pharma", "pharmaceutical", "medicine", "drug", "healthcare"],
            "auto": ["auto", "automobile", "car", "vehicle", "automotive", "auto sector"],
            "fmcg": ["fmcg", "consumer", "goods", "food", "beverage", "fmcg companies", "consumer goods"],
            "metal": ["metal", "steel", "iron", "mining", "aluminum", "metal stocks", "metals"],
            "energy": ["energy", "oil", "gas", "power", "electricity", "energy sector"],
            "telecom": ["telecom", "telecommunication", "mobile", "network"]
        }
        
        # Sector to company mappings for better search
        self.sector_companies = {
            "fmcg": ["ITC", "HINDUNILVR", "NESTLEIND"],
            "metal": ["COALINDIA", "POWERGRID", "NTPC"],
            "energy": ["ONGC", "COALINDIA", "NTPC", "POWERGRID"]
        }
        
        # Background task
        self.background_task = None
        
    async def start_background_updates(self):
        """Start background data collection for universal coverage"""
        logger.info("🚀 Starting universal financial data collection...")
        
        async def update_background_data():
            while True:
                try:
                    # Update popular stocks
                    await self._update_popular_stocks()
                    
                    # Update indices
                    await self._update_indices()
                    
                    # Update ETFs
                    await self._update_etfs()
                    
                    # Update mutual fund NAVs (simulated)
                    await self._update_mutual_funds()
                    
                    logger.info(f"📊 Universal data updated at {datetime.now().strftime('%H:%M:%S')}")
                    
                except Exception as e:
                    logger.error(f"Background update error: {e}")
                
                await asyncio.sleep(self.update_interval)
        
        self.background_task = asyncio.create_task(update_background_data())
    
    async def _update_popular_stocks(self):
        """Update data for popular stocks"""
        try:
            popular_symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS", 
                             "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS"]
            
            for symbol in popular_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    if info and 'currentPrice' in info:
                        self.background_data[symbol] = {
                            'current_price': info.get('currentPrice', 0),
                            'change': info.get('regularMarketChange', 0),
                            'change_percent': info.get('regularMarketChangePercent', 0),
                            'volume': info.get('regularMarketVolume', 0),
                            'market_cap': info.get('marketCap'),
                            'company_name': info.get('longName', symbol),
                            'sector': info.get('sector', 'Unknown'),
                            'last_updated': datetime.now()
                        }
                except Exception as e:
                    logger.warning(f"Failed to update {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Error updating popular stocks: {e}")
    
    async def _update_indices(self):
        """Update major indices data"""
        try:
            for index_key, index_info in self.indices.items():
                try:
                    ticker = yf.Ticker(index_info['symbol'])
                    hist = ticker.history(period="1d")
                    
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        self.background_data[f"INDEX_{index_key.upper()}"] = {
                            'current_price': latest['Close'],
                            'change': latest['Close'] - hist.iloc[-2]['Close'] if len(hist) > 1 else 0,
                            'change_percent': ((latest['Close'] - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close'] * 100) if len(hist) > 1 else 0,
                            'volume': latest['Volume'],
                            'name': index_info['name'],
                            'description': index_info['description'],
                            'last_updated': datetime.now()
                        }
                except Exception as e:
                    logger.warning(f"Failed to update index {index_key}: {e}")
                    
        except Exception as e:
            logger.error(f"Error updating indices: {e}")
    
    async def _update_etfs(self):
        """Update ETF data"""
        try:
            for etf_key, etf_info in self.etfs.items():
                try:
                    ticker = yf.Ticker(etf_info['symbol'])
                    info = ticker.info
                    
                    if info:
                        self.background_data[f"ETF_{etf_key.upper()}"] = {
                            'current_price': info.get('currentPrice', 0),
                            'change': info.get('regularMarketChange', 0),
                            'change_percent': info.get('regularMarketChangePercent', 0),
                            'volume': info.get('regularMarketVolume', 0),
                            'name': etf_info['name'],
                            'index': etf_info['index'],
                            'last_updated': datetime.now()
                        }
                except Exception as e:
                    logger.warning(f"Failed to update ETF {etf_key}: {e}")
                    
        except Exception as e:
            logger.error(f"Error updating ETFs: {e}")
    
    async def _update_mutual_funds(self):
        """Update mutual fund NAVs (simulated data)"""
        try:
            for mf_key, mf_info in self.mutual_funds.items():
                # Simulate NAV data (in real implementation, use AMFI API)
                base_nav = 50 + hash(mf_key) % 100  # Deterministic but varied
                change = (hash(mf_key) % 200 - 100) / 100  # -1 to +1
                
                self.background_data[f"MF_{mf_key.upper()}"] = {
                    'nav': base_nav + change,
                    'change': change,
                    'change_percent': (change / base_nav) * 100,
                    'name': mf_info['name'],
                    'type': mf_info['type'],
                    'last_updated': datetime.now()
                }
                
        except Exception as e:
            logger.error(f"Error updating mutual funds: {e}")
    
    def stop_background_updates(self):
        """Stop background data collection"""
        if self.background_task:
            self.background_task.cancel()
    
    async def search_universal(self, query: str, max_suggestions: int = 10) -> UniversalSearchResult:
        """
        Universal search across all financial instruments
        
        Args:
            query: User's search query
            max_suggestions: Maximum number of suggestions
            
        Returns:
            UniversalSearchResult with comprehensive data
        """
        start_time = time.time()
        query_lower = query.lower().strip()
        
        try:
            # Generate suggestions from all sources
            suggestions = []
            
            # Search stocks
            stock_suggestions = await self._search_stocks(query_lower)
            suggestions.extend(stock_suggestions)
            
            # Search mutual funds
            mf_suggestions = await self._search_mutual_funds(query_lower)
            suggestions.extend(mf_suggestions)
            
            # Search ETFs
            etf_suggestions = await self._search_etfs(query_lower)
            suggestions.extend(etf_suggestions)
            
            # Search indices
            index_suggestions = await self._search_indices(query_lower)
            suggestions.extend(index_suggestions)
            
            # Search by sector
            sector_suggestions = await self._search_by_sector(query_lower)
            suggestions.extend(sector_suggestions)
            
            # Sort by confidence and limit results
            suggestions.sort(key=lambda x: x.confidence, reverse=True)
            suggestions = suggestions[:max_suggestions]
            
            # Get comprehensive live data
            live_data = await self._get_universal_live_data(query_lower, suggestions)
            
            total_time = (time.time() - start_time) * 1000
            
            # Get unique categories
            categories = list(set(s.category.value for s in suggestions))
            
            return UniversalSearchResult(
                query=query,
                suggestions=suggestions,
                live_data=live_data,
                total_time_ms=total_time,
                timestamp=datetime.now(),
                instruments_found=len(suggestions),
                categories_covered=categories
            )
            
        except Exception as e:
            logger.error(f"Universal search error for '{query}': {e}")
            return UniversalSearchResult(
                query=query,
                suggestions=[],
                live_data={"error": str(e)},
                total_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                instruments_found=0,
                categories_covered=[]
            )
    
    async def _search_stocks(self, query: str) -> List[UniversalSearchSuggestion]:
        """Search for stocks in the enhanced database"""
        suggestions = []
        
        try:
            for symbol, company_info in self.enhanced_db.companies.items():
                # Check if query matches symbol, name, or aliases
                matches = []
                
                # Symbol match
                if query in symbol.lower():
                    matches.append(("symbol", 1.0))
                
                # Company name match
                if query in company_info.name.lower():
                    matches.append(("name", 0.9))
                
                # Aliases match
                for alias in company_info.aliases:
                    if query in alias.lower():
                        matches.append(("alias", 0.8))
                
                # Keywords match
                for keyword in company_info.keywords:
                    if query in keyword.lower():
                        matches.append(("keyword", 0.6))
                
                if matches:
                    # Calculate confidence based on best match
                    best_match = max(matches, key=lambda x: x[1])
                    confidence = best_match[1]
                    
                    # Adjust confidence based on query length and match quality
                    if query == symbol.lower():
                        confidence = 1.0
                    elif query == company_info.name.lower():
                        confidence = 0.95
                    elif len(query) >= 3:
                        confidence *= 0.9
                    
                    # Get instant data
                    instant_data = self._get_stock_instant_data(symbol, company_info)
                    
                    suggestion = UniversalSearchSuggestion(
                        query=f"{company_info.name} ({symbol})",
                        category=SearchCategory.COMPANY,
                        instrument_type=FinancialInstrumentType.STOCK,
                        symbol=symbol,
                        name=company_info.name,
                        description=f"{company_info.name} - {company_info.sector} ({company_info.exchange.value})",
                        instant_data=instant_data,
                        confidence=confidence,
                        exchange=company_info.exchange.value,
                        sector=company_info.sector,
                        market_cap=company_info.market_cap_category.value
                    )
                    
                    suggestions.append(suggestion)
                    
        except Exception as e:
            logger.error(f"Error searching stocks: {e}")
        
        return suggestions
    
    async def _search_mutual_funds(self, query: str) -> List[UniversalSearchSuggestion]:
        """Search for mutual funds"""
        suggestions = []
        
        try:
            for mf_key, mf_info in self.mutual_funds.items():
                if query in mf_key.lower() or query in mf_info['name'].lower():
                    confidence = 1.0 if query == mf_key.lower() else 0.8
                    
                    # Get instant data
                    instant_data = self.background_data.get(f"MF_{mf_key.upper()}", {})
                    
                    suggestion = UniversalSearchSuggestion(
                        query=mf_info['name'],
                        category=SearchCategory.MUTUAL_FUND,
                        instrument_type=FinancialInstrumentType.MUTUAL_FUND,
                        symbol=mf_info['symbol'],
                        name=mf_info['name'],
                        description=f"{mf_info['name']} - {mf_info['type']} Fund",
                        instant_data=instant_data,
                        confidence=confidence,
                        exchange="AMFI",
                        sector="Asset Management"
                    )
                    
                    suggestions.append(suggestion)
                    
        except Exception as e:
            logger.error(f"Error searching mutual funds: {e}")
        
        return suggestions
    
    async def _search_etfs(self, query: str) -> List[UniversalSearchSuggestion]:
        """Search for ETFs"""
        suggestions = []
        
        try:
            for etf_key, etf_info in self.etfs.items():
                if query in etf_key.lower() or query in etf_info['name'].lower():
                    confidence = 1.0 if query == etf_key.lower() else 0.8
                    
                    # Get instant data
                    instant_data = self.background_data.get(f"ETF_{etf_key.upper()}", {})
                    
                    suggestion = UniversalSearchSuggestion(
                        query=etf_info['name'],
                        category=SearchCategory.COMPANY,
                        instrument_type=FinancialInstrumentType.ETF,
                        symbol=etf_info['symbol'],
                        name=etf_info['name'],
                        description=f"{etf_info['name']} - Tracks {etf_info['index']}",
                        instant_data=instant_data,
                        confidence=confidence,
                        exchange="NSE",
                        sector="ETF"
                    )
                    
                    suggestions.append(suggestion)
                    
        except Exception as e:
            logger.error(f"Error searching ETFs: {e}")
        
        return suggestions
    
    async def _search_indices(self, query: str) -> List[UniversalSearchSuggestion]:
        """Search for indices"""
        suggestions = []
        
        try:
            for index_key, index_info in self.indices.items():
                if query in index_key.lower() or query in index_info['name'].lower():
                    confidence = 1.0 if query == index_key.lower() else 0.8
                    
                    # Get instant data
                    instant_data = self.background_data.get(f"INDEX_{index_key.upper()}", {})
                    
                    suggestion = UniversalSearchSuggestion(
                        query=index_info['name'],
                        category=SearchCategory.COMPANY,
                        instrument_type=FinancialInstrumentType.INDEX,
                        symbol=index_info['symbol'],
                        name=index_info['name'],
                        description=index_info['description'],
                        instant_data=instant_data,
                        confidence=confidence,
                        exchange="NSE/BSE",
                        sector="Index"
                    )
                    
                    suggestions.append(suggestion)
                    
        except Exception as e:
            logger.error(f"Error searching indices: {e}")
        
        return suggestions
    
    async def _search_by_sector(self, query: str) -> List[UniversalSearchSuggestion]:
        """Search by sector keywords"""
        suggestions = []
        
        try:
            for sector, keywords in self.sector_keywords.items():
                for keyword in keywords:
                    if query.lower() == keyword.lower() or keyword.lower() in query.lower():
                        # Find companies in this sector from enhanced database
                        sector_companies = []
                        for symbol, company_info in self.enhanced_db.companies.items():
                            if sector.lower() in company_info.sector.lower():
                                sector_companies.append((symbol, company_info))
                        
                        # Also add predefined sector companies if available
                        if sector in self.sector_companies:
                            for symbol in self.sector_companies[sector]:
                                if symbol in self.enhanced_db.companies:
                                    company_info = self.enhanced_db.companies[symbol]
                                    sector_companies.append((symbol, company_info))
                        
                        if sector_companies:
                            # Create sector suggestion
                            suggestion = UniversalSearchSuggestion(
                                query=f"{sector.title()} Sector",
                                category=SearchCategory.SECTOR,
                                instrument_type=FinancialInstrumentType.STOCK,
                                symbol=f"SECTOR_{sector.upper()}",
                                name=f"{sector.title()} Sector",
                                description=f"{len(sector_companies)} companies in {sector.title()} sector",
                                instant_data={"companies_count": len(sector_companies), "top_companies": [c[0] for c in sector_companies[:5]]},
                                confidence=0.8 if query.lower() == keyword.lower() else 0.7,
                                exchange="Multiple",
                                sector=sector.title()
                            )
                            
                            suggestions.append(suggestion)
                            
                            # Also add individual companies from this sector
                            for symbol, company_info in sector_companies[:3]:  # Top 3 companies
                                company_suggestion = UniversalSearchSuggestion(
                                    query=f"{company_info.name} ({sector.title()} Sector)",
                                    category=SearchCategory.COMPANY,
                                    instrument_type=FinancialInstrumentType.STOCK,
                                    symbol=symbol,
                                    name=company_info.name,
                                    description=f"{company_info.name} - {sector.title()} sector company",
                                    instant_data=self._get_stock_instant_data(symbol, company_info),
                                    confidence=0.6,
                                    exchange=company_info.exchange.value,
                                    sector=company_info.sector,
                                    market_cap=company_info.market_cap_category.value
                                )
                                suggestions.append(company_suggestion)
                            
                            break
                            
        except Exception as e:
            logger.error(f"Error searching by sector: {e}")
        
        return suggestions
    
    def _get_stock_instant_data(self, symbol: str, company_info) -> Dict[str, Any]:
        """Get instant data for a stock"""
        try:
            # Check if we have background data
            yahoo_symbol = f"{symbol}.NS"  # Assume NSE for now
            if yahoo_symbol in self.background_data:
                return self.background_data[yahoo_symbol]
            
            # Return basic company info
            return {
                "symbol": symbol,
                "name": company_info.name,
                "sector": company_info.sector,
                "exchange": company_info.exchange.value,
                "market_cap_category": company_info.market_cap_category.value
            }
            
        except Exception as e:
            logger.error(f"Error getting stock instant data: {e}")
            return {}
    
    async def _get_universal_live_data(self, query: str, suggestions: List[UniversalSearchSuggestion]) -> Dict[str, Any]:
        """Get comprehensive live data for the search"""
        try:
            live_data = {
                "query": query,
                "total_suggestions": len(suggestions),
                "instruments_by_type": {},
                "top_matches": []
            }
            
            # Group by instrument type
            for suggestion in suggestions:
                inst_type = suggestion.instrument_type.value
                if inst_type not in live_data["instruments_by_type"]:
                    live_data["instruments_by_type"][inst_type] = 0
                live_data["instruments_by_type"][inst_type] += 1
            
            # Add top matches with instant data
            for suggestion in suggestions[:5]:
                if suggestion.instant_data:
                    live_data["top_matches"].append({
                        "name": suggestion.name,
                        "symbol": suggestion.symbol,
                        "type": suggestion.instrument_type.value,
                        "instant_data": suggestion.instant_data
                    })
            
            return live_data
            
        except Exception as e:
            logger.error(f"Error getting universal live data: {e}")
            return {}
    
    async def get_autocomplete(self, query: str, limit: int = 8) -> List[str]:
        """Get autocomplete suggestions for any financial instrument"""
        try:
            result = await self.search_universal(query, limit)
            return [suggestion.query for suggestion in result.suggestions]
        except Exception as e:
            logger.error(f"Autocomplete error: {e}")
            return []

# Test function
async def test_universal_search():
    """Test the universal financial search engine"""
    print("🚀 Testing Universal Financial Search Engine")
    print("=" * 70)
    
    engine = UniversalFinancialSearchEngine()
    await engine.start_background_updates()
    
    # Wait for initial data collection
    print("⏳ Collecting universal financial data...")
    await asyncio.sleep(5)
    
    # Test various types of queries
    test_queries = [
        # Stocks
        "reliance",
        "tcs",
        "hdfc bank",
        
        # Mutual funds
        "sbi bluechip",
        "hdfc top 100",
        "parag parikh",
        
        # ETFs
        "nifty bees",
        "gold bees",
        
        # Indices
        "nifty",
        "sensex",
        "bank nifty",
        
        # Sectors
        "banking",
        "it sector",
        "pharma"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 40)
        
        result = await engine.search_universal(query)
        
        print(f"⚡ Response Time: {result.total_time_ms:.2f}ms")
        print(f"📊 Instruments Found: {result.instruments_found}")
        print(f"🏷️  Categories: {', '.join(result.categories_covered)}")
        
        for i, suggestion in enumerate(result.suggestions[:3], 1):
            print(f"   {i}. {suggestion.name}")
            print(f"      Type: {suggestion.instrument_type.value.title()}")
            print(f"      Exchange: {suggestion.exchange}")
            print(f"      Confidence: {suggestion.confidence:.2f}")
            
            if suggestion.instant_data:
                if 'current_price' in suggestion.instant_data:
                    price = suggestion.instant_data['current_price']
                    print(f"      Price: ₹{price}")
                elif 'nav' in suggestion.instant_data:
                    nav = suggestion.instant_data['nav']
                    print(f"      NAV: ₹{nav:.2f}")
    
    engine.stop_background_updates()
    print("\n✅ Universal search engine test completed!")

if __name__ == "__main__":
    asyncio.run(test_universal_search())
