"""
ZTECH Real-Time Search Engine
Google-level autocomplete and instant data capture for ZTECH queries
Provides instant results as users type "ztech"
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import redis
from concurrent.futures import ThreadPoolExecutor
import threading
from dual_ztech_data_service import DualZTechAPIService, ZTechCompanyType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchType(Enum):
    """Types of search suggestions"""
    COMPANY = "company"
    PRICE = "price"
    ANALYSIS = "analysis"
    NEWS = "news"
    COMPARISON = "comparison"
    TECHNICAL = "technical"

@dataclass
class SearchSuggestion:
    """Search suggestion with instant data"""
    query: str
    type: SearchType
    company: str
    description: str
    instant_data: Dict[str, Any]
    confidence: float
    response_time_ms: float

@dataclass
class InstantResult:
    """Instant search result with live data"""
    query: str
    suggestions: List[SearchSuggestion]
    live_data: Dict[str, Any]
    total_time_ms: float
    timestamp: datetime

class ZTechRealtimeSearchEngine:
    """
    Google-level real-time search engine for ZTECH
    Provides instant autocomplete and data capture
    """
    
    def __init__(self):
        """Initialize the real-time search engine"""
        self.dual_ztech_service = DualZTechAPIService()
        
        # Redis for caching (optional, fallback to in-memory)
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_available = True
        except:
            self.redis_client = None
            self.redis_available = False
            logger.warning("Redis not available, using in-memory cache")
        
        # In-memory cache
        self.cache = {}
        self.cache_ttl = 30  # 30 seconds TTL
        
        # Pre-defined search patterns
        self.search_patterns = {
            # Company queries
            "ztech": [
                ("ztech india", SearchType.COMPANY, "Z-Tech (India) Limited"),
                ("ztech india price", SearchType.PRICE, "Z-Tech (India) Limited"),
                ("ztech india live", SearchType.PRICE, "Z-Tech (India) Limited"),
                ("ztech india analysis", SearchType.ANALYSIS, "Z-Tech (India) Limited"),
                ("ztech india technical", SearchType.TECHNICAL, "Z-Tech (India) Limited"),
            ],
            "zt": [
                ("ztech", SearchType.COMPANY, "Both Companies"),
                ("ztech india", SearchType.COMPANY, "Z-Tech (India) Limited"),
            ],
            "zen": [
                ("zentech", SearchType.COMPANY, "Zentech Systems Limited"),
                ("zentech systems", SearchType.COMPANY, "Zentech Systems Limited"),
                ("zentech price", SearchType.PRICE, "Zentech Systems Limited"),
                ("zentech emerge", SearchType.COMPANY, "Zentech Systems Limited"),
            ],
            "emerge": [
                ("emerge ztech", SearchType.COMPANY, "Zentech Systems Limited"),
                ("emerge zentech", SearchType.COMPANY, "Zentech Systems Limited"),
            ],
            "compare": [
                ("compare ztech", SearchType.COMPARISON, "Both Companies"),
                ("compare ztech companies", SearchType.COMPARISON, "Both Companies"),
                ("ztech comparison", SearchType.COMPARISON, "Both Companies"),
            ]
        }
        
        # Background data updater
        self.background_data = {}
        self.last_update = {}
        self.update_interval = 10  # 10 seconds
        
        # Start background data collection
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.background_task = None
        
    async def start_background_updates(self):
        """Start background data collection for instant responses"""
        logger.info("🚀 Starting background ZTECH data collection...")
        
        async def update_background_data():
            while True:
                try:
                    # Update Z-Tech India data
                    ztech_india_data = await self.dual_ztech_service.data_service.get_live_data(
                        ZTechCompanyType.ZTECH_INDIA
                    )
                    if ztech_india_data:
                        self.background_data['ztech_india'] = asdict(ztech_india_data)
                        self.last_update['ztech_india'] = datetime.now()
                    
                    # Update Zentech Systems data
                    zentech_data = await self.dual_ztech_service.data_service.get_live_data(
                        ZTechCompanyType.ZENTECH_SYSTEMS
                    )
                    if zentech_data:
                        self.background_data['zentech_systems'] = asdict(zentech_data)
                        self.last_update['zentech_systems'] = datetime.now()
                    
                    # Update comparison data
                    comparison_data = await self.dual_ztech_service.data_service.get_all_companies_data()
                    if comparison_data:
                        self.background_data['comparison'] = comparison_data
                        self.last_update['comparison'] = datetime.now()
                    
                    logger.info(f"📊 Background data updated at {datetime.now().strftime('%H:%M:%S')}")
                    
                except Exception as e:
                    logger.error(f"Background update error: {e}")
                
                await asyncio.sleep(self.update_interval)
        
        self.background_task = asyncio.create_task(update_background_data())
    
    def stop_background_updates(self):
        """Stop background data collection"""
        if self.background_task:
            self.background_task.cancel()
    
    async def search_realtime(self, query: str, max_suggestions: int = 8) -> InstantResult:
        """
        Real-time search with instant results
        
        Args:
            query: User's search query (as they type)
            max_suggestions: Maximum number of suggestions
            
        Returns:
            InstantResult with suggestions and live data
        """
        start_time = time.time()
        query_lower = query.lower().strip()
        
        try:
            # Generate suggestions
            suggestions = await self._generate_suggestions(query_lower, max_suggestions)
            
            # Get instant live data
            live_data = await self._get_instant_live_data(query_lower)
            
            total_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return InstantResult(
                query=query,
                suggestions=suggestions,
                live_data=live_data,
                total_time_ms=total_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Real-time search error for '{query}': {e}")
            return InstantResult(
                query=query,
                suggestions=[],
                live_data={"error": str(e)},
                total_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now()
            )
    
    async def _generate_suggestions(self, query: str, max_suggestions: int) -> List[SearchSuggestion]:
        """Generate search suggestions based on query"""
        suggestions = []
        
        # Find matching patterns
        for prefix, patterns in self.search_patterns.items():
            if query.startswith(prefix) or prefix.startswith(query):
                for pattern_query, search_type, company in patterns:
                    if pattern_query.startswith(query) or query in pattern_query:
                        # Calculate confidence based on match quality
                        confidence = self._calculate_confidence(query, pattern_query)
                        
                        # Get instant data for this suggestion
                        instant_data = await self._get_suggestion_data(pattern_query, search_type, company)
                        
                        suggestion = SearchSuggestion(
                            query=pattern_query,
                            type=search_type,
                            company=company,
                            description=self._generate_description(pattern_query, search_type, company, instant_data),
                            instant_data=instant_data,
                            confidence=confidence,
                            response_time_ms=0  # Will be calculated
                        )
                        
                        suggestions.append(suggestion)
        
        # Sort by confidence and limit results
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return suggestions[:max_suggestions]
    
    def _calculate_confidence(self, query: str, pattern: str) -> float:
        """Calculate confidence score for suggestion"""
        if query == pattern:
            return 1.0
        elif pattern.startswith(query):
            return 0.9 - (len(pattern) - len(query)) * 0.1
        elif query in pattern:
            return 0.7
        else:
            # Fuzzy matching
            common_chars = sum(1 for a, b in zip(query, pattern) if a == b)
            return max(0.3, common_chars / max(len(query), len(pattern)))
    
    async def _get_suggestion_data(self, query: str, search_type: SearchType, company: str) -> Dict[str, Any]:
        """Get instant data for suggestion"""
        try:
            if search_type == SearchType.PRICE:
                if "india" in query.lower():
                    return self.background_data.get('ztech_india', {})
                elif "zentech" in query.lower() or "emerge" in query.lower():
                    return self.background_data.get('zentech_systems', {})
                else:
                    # Return both prices
                    return {
                        "ztech_india": self.background_data.get('ztech_india', {}),
                        "zentech_systems": self.background_data.get('zentech_systems', {})
                    }
            
            elif search_type == SearchType.COMPARISON:
                return self.background_data.get('comparison', {})
            
            elif search_type == SearchType.COMPANY:
                if "india" in company.lower():
                    return self.background_data.get('ztech_india', {})
                elif "zentech" in company.lower():
                    return self.background_data.get('zentech_systems', {})
                else:
                    return self.background_data.get('comparison', {})
            
            else:
                # Default to company data
                if "india" in query.lower():
                    return self.background_data.get('ztech_india', {})
                else:
                    return self.background_data.get('zentech_systems', {})
                    
        except Exception as e:
            logger.error(f"Error getting suggestion data: {e}")
            return {}
    
    def _generate_description(self, query: str, search_type: SearchType, company: str, data: Dict) -> str:
        """Generate description for suggestion"""
        try:
            if search_type == SearchType.PRICE and data:
                if 'current_price' in data:
                    price = data['current_price']
                    change = data.get('change', 0)
                    change_symbol = "+" if change >= 0 else ""
                    return f"{company} - ₹{price} ({change_symbol}{change:.2f})"
                elif 'ztech_india' in data and 'zentech_systems' in data:
                    zi_price = data['ztech_india'].get('current_price', 0)
                    zs_price = data['zentech_systems'].get('current_price', 0)
                    return f"Z-Tech India: ₹{zi_price} | Zentech Systems: ₹{zs_price}"
            
            elif search_type == SearchType.COMPARISON and data:
                return "Compare Z-Tech India vs Zentech Systems - Live analysis"
            
            elif search_type == SearchType.COMPANY and data:
                if 'current_price' in data:
                    price = data['current_price']
                    exchange = data.get('exchange', 'NSE')
                    return f"{company} - ₹{price} ({exchange})"
                elif 'comparison' in data:
                    return "Both ZTECH companies - Complete analysis"
            
            # Default descriptions
            descriptions = {
                SearchType.COMPANY: f"{company} - Company information",
                SearchType.PRICE: f"{company} - Live price",
                SearchType.ANALYSIS: f"{company} - Technical analysis",
                SearchType.NEWS: f"{company} - Latest news",
                SearchType.COMPARISON: "Compare ZTECH companies",
                SearchType.TECHNICAL: f"{company} - Technical indicators"
            }
            
            return descriptions.get(search_type, f"{company} - {query}")
            
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return f"{company} - {query}"
    
    async def _get_instant_live_data(self, query: str) -> Dict[str, Any]:
        """Get instant live data for query"""
        try:
            # Determine which data to return based on query
            if "compare" in query or "both" in query:
                return self.background_data.get('comparison', {})
            elif "india" in query:
                return self.background_data.get('ztech_india', {})
            elif "zentech" in query or "emerge" in query:
                return self.background_data.get('zentech_systems', {})
            else:
                # Return summary of both
                return {
                    "ztech_india": self.background_data.get('ztech_india', {}),
                    "zentech_systems": self.background_data.get('zentech_systems', {}),
                    "last_updated": max(
                        self.last_update.get('ztech_india', datetime.min),
                        self.last_update.get('zentech_systems', datetime.min)
                    ).isoformat()
                }
        except Exception as e:
            logger.error(f"Error getting instant live data: {e}")
            return {}
    
    async def get_autocomplete(self, query: str, limit: int = 5) -> List[str]:
        """Get autocomplete suggestions (just the query strings)"""
        try:
            result = await self.search_realtime(query, limit)
            return [suggestion.query for suggestion in result.suggestions]
        except Exception as e:
            logger.error(f"Autocomplete error: {e}")
            return []
    
    async def get_instant_data(self, query: str) -> Dict[str, Any]:
        """Get instant data for a specific query"""
        try:
            result = await self.search_realtime(query, 1)
            return result.live_data
        except Exception as e:
            logger.error(f"Instant data error: {e}")
            return {"error": str(e)}

# FastAPI Integration
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# Global search engine instance
search_engine = None

async def initialize_search_engine():
    """Initialize the global search engine"""
    global search_engine
    if search_engine is None:
        search_engine = ZTechRealtimeSearchEngine()
        await search_engine.start_background_updates()
        logger.info("🚀 ZTECH Real-time Search Engine initialized")

# API Endpoints for integration
async def realtime_search_endpoint(query: str = Query(..., description="Search query")):
    """Real-time search endpoint"""
    if search_engine is None:
        await initialize_search_engine()
    
    result = await search_engine.search_realtime(query)
    return {
        "query": result.query,
        "suggestions": [asdict(s) for s in result.suggestions],
        "live_data": result.live_data,
        "response_time_ms": result.total_time_ms,
        "timestamp": result.timestamp.isoformat()
    }

async def autocomplete_endpoint(q: str = Query(..., description="Query for autocomplete")):
    """Autocomplete endpoint"""
    if search_engine is None:
        await initialize_search_engine()
    
    suggestions = await search_engine.get_autocomplete(q)
    return {
        "query": q,
        "suggestions": suggestions,
        "timestamp": datetime.now().isoformat()
    }

async def instant_data_endpoint(query: str = Query(..., description="Query for instant data")):
    """Instant data endpoint"""
    if search_engine is None:
        await initialize_search_engine()
    
    data = await search_engine.get_instant_data(query)
    return {
        "query": query,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }

# Test function
async def test_realtime_search():
    """Test the real-time search engine"""
    print("🚀 Testing ZTECH Real-time Search Engine")
    print("=" * 60)
    
    engine = ZTechRealtimeSearchEngine()
    await engine.start_background_updates()
    
    # Wait for initial data collection
    print("⏳ Collecting initial data...")
    await asyncio.sleep(3)
    
    # Test queries
    test_queries = [
        "z",
        "zt", 
        "ztech",
        "ztech i",
        "ztech india",
        "ztech india p",
        "ztech india price",
        "zen",
        "zentech",
        "compare",
        "compare ztech"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 30)
        
        result = await engine.search_realtime(query)
        
        print(f"⚡ Response Time: {result.total_time_ms:.2f}ms")
        print(f"📊 Suggestions: {len(result.suggestions)}")
        
        for i, suggestion in enumerate(result.suggestions[:3], 1):
            print(f"   {i}. {suggestion.query}")
            print(f"      {suggestion.description}")
            print(f"      Confidence: {suggestion.confidence:.2f}")
        
        if result.live_data:
            print(f"📈 Live Data Available: {len(result.live_data)} fields")
    
    engine.stop_background_updates()
    print("\n✅ Real-time search engine test completed!")

if __name__ == "__main__":
    asyncio.run(test_realtime_search())
