"""
JSON Database Manager
Manages comprehensive market data in JSON format for universal search engine
Provides fast access, search capabilities, and auto-sync with live data
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict
import asyncio
import aiofiles

from comprehensive_market_data_fetcher import CompanyData, MutualFundData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result structure"""
    symbol_or_code: str
    name: str
    instrument_type: str  # STOCK, MUTUAL_FUND, ETF, INDEX
    exchange: str
    category: str
    confidence: float
    live_data: Optional[Dict[str, Any]] = None

class JSONDatabaseManager:
    """Manages comprehensive market data in JSON format"""
    
    def __init__(self, data_dir: str = "market_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Database files
        self.companies_file = self.data_dir / "all_companies.json"
        self.mutual_funds_file = self.data_dir / "all_mutual_funds.json"
        self.search_index_file = self.data_dir / "search_index.json"
        self.metadata_file = self.data_dir / "metadata.json"
        
        # In-memory cache
        self.companies_cache: Dict[str, CompanyData] = {}
        self.funds_cache: Dict[str, MutualFundData] = {}
        self.search_index: Dict[str, List[str]] = {}
        self.last_loaded = None
        
        # Search optimization
        self.keyword_index: Dict[str, Set[str]] = {}
        self.sector_index: Dict[str, List[str]] = {}
        self.category_index: Dict[str, List[str]] = {}

    async def load_database(self) -> bool:
        """Load database from JSON files"""
        try:
            logger.info("📚 Loading comprehensive market database...")
            
            # Load companies
            if self.companies_file.exists():
                async with aiofiles.open(self.companies_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    companies_data = json.loads(content)
                    
                    for symbol, company_dict in companies_data.get("companies", {}).items():
                        self.companies_cache[symbol] = CompanyData(**company_dict)
            
            # Load mutual funds
            if self.mutual_funds_file.exists():
                async with aiofiles.open(self.mutual_funds_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    funds_data = json.loads(content)
                    
                    for code, fund_dict in funds_data.get("mutual_funds", {}).items():
                        self.funds_cache[code] = MutualFundData(**fund_dict)
            
            # Build search indices
            await self.build_search_indices()
            
            self.last_loaded = datetime.now()
            
            logger.info(f"✅ Database loaded: {len(self.companies_cache)} companies, {len(self.funds_cache)} funds")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading database: {e}")
            return False

    async def build_search_indices(self):
        """Build search indices for fast lookups"""
        try:
            logger.info("🔍 Building search indices...")
            
            self.keyword_index.clear()
            self.sector_index.clear()
            self.category_index.clear()
            
            # Index companies
            for symbol, company in self.companies_cache.items():
                # Keyword index
                keywords = [
                    company.name.lower(),
                    company.symbol.lower(),
                    company.sector.lower(),
                    company.industry.lower()
                ]
                
                # Add name words
                name_words = company.name.lower().split()
                keywords.extend(name_words)
                
                # Add abbreviations
                if len(name_words) > 1:
                    abbreviation = ''.join([word[0] for word in name_words if word])
                    keywords.append(abbreviation)
                
                for keyword in keywords:
                    if keyword and len(keyword) > 1:
                        if keyword not in self.keyword_index:
                            self.keyword_index[keyword] = set()
                        self.keyword_index[keyword].add(f"COMPANY:{symbol}")
                
                # Sector index
                sector = company.sector.lower()
                if sector not in self.sector_index:
                    self.sector_index[sector] = []
                self.sector_index[sector].append(symbol)
            
            # Index mutual funds
            for code, fund in self.funds_cache.items():
                # Keyword index
                keywords = [
                    fund.scheme_name.lower(),
                    fund.amc_name.lower(),
                    fund.category.lower(),
                    fund.sub_category.lower()
                ]
                
                # Add scheme name words
                name_words = fund.scheme_name.lower().split()
                keywords.extend(name_words)
                
                # Add AMC abbreviation
                amc_words = fund.amc_name.lower().split()
                if len(amc_words) > 1:
                    amc_abbr = ''.join([word[0] for word in amc_words if word])
                    keywords.append(amc_abbr)
                
                for keyword in keywords:
                    if keyword and len(keyword) > 1:
                        if keyword not in self.keyword_index:
                            self.keyword_index[keyword] = set()
                        self.keyword_index[keyword].add(f"FUND:{code}")
                
                # Category index
                category = fund.category.lower()
                if category not in self.category_index:
                    self.category_index[category] = []
                self.category_index[category].append(code)
            
            # Save search index
            search_index_data = {
                "last_built": datetime.now().isoformat(),
                "companies_count": len(self.companies_cache),
                "funds_count": len(self.funds_cache),
                "keywords_count": len(self.keyword_index),
                "sectors_count": len(self.sector_index),
                "categories_count": len(self.category_index)
            }
            
            async with aiofiles.open(self.search_index_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(search_index_data, indent=2))
            
            logger.info(f"✅ Search indices built: {len(self.keyword_index)} keywords indexed")
            
        except Exception as e:
            logger.error(f"❌ Error building search indices: {e}")

    async def search_instruments(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search for instruments using the query"""
        try:
            if not self.companies_cache and not self.funds_cache:
                await self.load_database()
            
            query_lower = query.lower().strip()
            results = []
            
            # Direct symbol/code match (highest priority)
            if query_lower.upper() in self.companies_cache:
                company = self.companies_cache[query_lower.upper()]
                results.append(SearchResult(
                    symbol_or_code=company.symbol,
                    name=company.name,
                    instrument_type="STOCK",
                    exchange=company.exchange,
                    category=company.sector,
                    confidence=1.0
                ))
            
            if query_lower in self.funds_cache:
                fund = self.funds_cache[query_lower]
                results.append(SearchResult(
                    symbol_or_code=fund.scheme_code,
                    name=fund.scheme_name,
                    instrument_type="MUTUAL_FUND",
                    exchange="AMFI",
                    category=fund.category,
                    confidence=1.0
                ))
            
            # Keyword search
            matched_instruments = set()
            
            # Exact keyword match
            if query_lower in self.keyword_index:
                matched_instruments.update(self.keyword_index[query_lower])
            
            # Partial keyword match
            for keyword, instruments in self.keyword_index.items():
                if query_lower in keyword or keyword in query_lower:
                    matched_instruments.update(instruments)
            
            # Process matched instruments
            for instrument_id in matched_instruments:
                instrument_type, identifier = instrument_id.split(":", 1)
                
                if instrument_type == "COMPANY" and identifier in self.companies_cache:
                    company = self.companies_cache[identifier]
                    
                    # Calculate confidence based on match quality
                    confidence = self.calculate_match_confidence(query_lower, [
                        company.name.lower(),
                        company.symbol.lower(),
                        company.sector.lower()
                    ])
                    
                    # Avoid duplicates
                    if not any(r.symbol_or_code == company.symbol for r in results):
                        results.append(SearchResult(
                            symbol_or_code=company.symbol,
                            name=company.name,
                            instrument_type="STOCK",
                            exchange=company.exchange,
                            category=company.sector,
                            confidence=confidence
                        ))
                
                elif instrument_type == "FUND" and identifier in self.funds_cache:
                    fund = self.funds_cache[identifier]
                    
                    # Calculate confidence
                    confidence = self.calculate_match_confidence(query_lower, [
                        fund.scheme_name.lower(),
                        fund.amc_name.lower(),
                        fund.category.lower()
                    ])
                    
                    # Avoid duplicates
                    if not any(r.symbol_or_code == fund.scheme_code for r in results):
                        results.append(SearchResult(
                            symbol_or_code=fund.scheme_code,
                            name=fund.scheme_name,
                            instrument_type="MUTUAL_FUND",
                            exchange="AMFI",
                            category=fund.category,
                            confidence=confidence
                        ))
            
            # Sort by confidence and limit results
            results.sort(key=lambda x: x.confidence, reverse=True)
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []

    def calculate_match_confidence(self, query: str, target_strings: List[str]) -> float:
        """Calculate match confidence score"""
        max_confidence = 0.0
        
        for target in target_strings:
            if not target:
                continue
                
            # Exact match
            if query == target:
                return 1.0
            
            # Starts with
            if target.startswith(query):
                confidence = 0.9
            # Contains
            elif query in target:
                confidence = 0.7
            # Word match
            elif any(word.startswith(query) for word in target.split()):
                confidence = 0.6
            # Partial word match
            elif any(query in word for word in target.split()):
                confidence = 0.5
            else:
                confidence = 0.0
            
            max_confidence = max(max_confidence, confidence)
        
        return max_confidence

    async def get_companies_by_sector(self, sector: str) -> List[CompanyData]:
        """Get all companies in a specific sector"""
        try:
            if not self.companies_cache:
                await self.load_database()
            
            sector_lower = sector.lower()
            companies = []
            
            if sector_lower in self.sector_index:
                for symbol in self.sector_index[sector_lower]:
                    if symbol in self.companies_cache:
                        companies.append(self.companies_cache[symbol])
            
            return companies
            
        except Exception as e:
            logger.error(f"❌ Error getting companies by sector: {e}")
            return []

    async def get_funds_by_category(self, category: str) -> List[MutualFundData]:
        """Get all funds in a specific category"""
        try:
            if not self.funds_cache:
                await self.load_database()
            
            category_lower = category.lower()
            funds = []
            
            if category_lower in self.category_index:
                for code in self.category_index[category_lower]:
                    if code in self.funds_cache:
                        funds.append(self.funds_cache[code])
            
            return funds
            
        except Exception as e:
            logger.error(f"❌ Error getting funds by category: {e}")
            return []

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            if not self.companies_cache and not self.funds_cache:
                await self.load_database()
            
            # Company statistics
            exchange_counts = {}
            sector_counts = {}
            
            for company in self.companies_cache.values():
                exchange = company.exchange
                sector = company.sector
                
                exchange_counts[exchange] = exchange_counts.get(exchange, 0) + 1
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            # Fund statistics
            amc_counts = {}
            category_counts = {}
            
            for fund in self.funds_cache.values():
                amc = fund.amc_name
                category = fund.category
                
                amc_counts[amc] = amc_counts.get(amc, 0) + 1
                category_counts[category] = category_counts.get(category, 0) + 1
            
            return {
                "last_loaded": self.last_loaded.isoformat() if self.last_loaded else None,
                "companies": {
                    "total": len(self.companies_cache),
                    "by_exchange": exchange_counts,
                    "by_sector": dict(sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:10])
                },
                "mutual_funds": {
                    "total": len(self.funds_cache),
                    "by_amc": dict(sorted(amc_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                    "by_category": category_counts
                },
                "search_index": {
                    "keywords": len(self.keyword_index),
                    "sectors": len(self.sector_index),
                    "categories": len(self.category_index)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {}

    async def refresh_database(self) -> bool:
        """Refresh database from JSON files"""
        try:
            logger.info("🔄 Refreshing database...")
            
            # Clear cache
            self.companies_cache.clear()
            self.funds_cache.clear()
            
            # Reload
            success = await self.load_database()
            
            if success:
                logger.info("✅ Database refreshed successfully")
            else:
                logger.error("❌ Database refresh failed")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error refreshing database: {e}")
            return False

    def is_data_stale(self, max_age_hours: int = 24) -> bool:
        """Check if database is stale"""
        if not self.last_loaded:
            return True
        
        age = datetime.now() - self.last_loaded
        return age > timedelta(hours=max_age_hours)

    async def get_company_by_symbol(self, symbol: str) -> Optional[CompanyData]:
        """Get company by symbol"""
        if not self.companies_cache:
            await self.load_database()
        
        return self.companies_cache.get(symbol.upper())

    async def get_fund_by_code(self, code: str) -> Optional[MutualFundData]:
        """Get mutual fund by scheme code"""
        if not self.funds_cache:
            await self.load_database()
        
        return self.funds_cache.get(code)

    async def search_suggestions(self, query: str, max_suggestions: int = 5) -> List[str]:
        """Get search suggestions for autocomplete"""
        try:
            if not self.keyword_index:
                await self.load_database()
            
            query_lower = query.lower()
            suggestions = set()
            
            # Find matching keywords
            for keyword in self.keyword_index.keys():
                if keyword.startswith(query_lower):
                    suggestions.add(keyword)
                elif query_lower in keyword:
                    suggestions.add(keyword)
            
            # Sort by relevance (starts with query first)
            sorted_suggestions = sorted(suggestions, key=lambda x: (
                not x.startswith(query_lower),  # Starts with query first
                len(x),  # Shorter suggestions first
                x  # Alphabetical
            ))
            
            return sorted_suggestions[:max_suggestions]
            
        except Exception as e:
            logger.error(f"❌ Error getting suggestions: {e}")
            return []

# Global database instance
db_manager = JSONDatabaseManager()

async def initialize_database():
    """Initialize the global database"""
    await db_manager.load_database()

async def main():
    """Test the database manager"""
    db = JSONDatabaseManager()
    await db.load_database()
    
    # Test search
    results = await db.search_instruments("reliance")
    print(f"Search results for 'reliance': {len(results)}")
    for result in results:
        print(f"  {result.symbol_or_code}: {result.name} ({result.confidence:.2f})")
    
    # Test stats
    stats = await db.get_database_stats()
    print(f"\nDatabase stats: {stats}")

if __name__ == "__main__":
    asyncio.run(main())
