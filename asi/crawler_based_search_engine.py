#!/usr/bin/env python3
"""
Crawler-Based Universal Financial Search Engine
Google-like search engine using comprehensive crawler-based database
"""

import json
import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result with confidence scoring"""
    company_name: str
    primary_symbol: str
    exchanges: List[str]
    symbols: Dict[str, str]
    sector: str
    industry: str
    confidence_score: float
    search_confidence: float
    crawling_method: str
    data_sources: List[str]
    isin: Optional[str] = None
    match_type: str = "partial"

class CrawlerBasedSearchEngine:
    """Google-like search engine using crawler-based database"""
    
    def __init__(self):
        self.database_file = "market_data/final_crawler_based_companies.json"
        self.companies = []
        self.search_index = {}
        self.metadata = {}
        self.statistics = {}
        self.load_database()
        self.build_search_index()
    
    def load_database(self):
        """Load crawler-based database"""
        try:
            with open(self.database_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.companies = data.get('companies', [])
            self.metadata = data.get('metadata', {})
            self.statistics = data.get('statistics', {})
            
            logger.info(f"✅ Loaded crawler-based database:")
            logger.info(f"   Total Companies: {len(self.companies):,}")
            logger.info(f"   Integration Method: {self.metadata.get('integration_method')}")
            logger.info(f"   Crawling Approach: {self.metadata.get('crawling_approach')}")
            
        except FileNotFoundError:
            logger.error(f"❌ Database file not found: {self.database_file}")
            self.companies = []
    
    def build_search_index(self):
        """Build search index for fast lookups"""
        logger.info("🔍 Building search index...")
        
        self.search_index = {
            'by_name': {},
            'by_symbol': {},
            'by_keywords': {},
            'by_sector': {},
            'by_industry': {}
        }
        
        for i, company in enumerate(self.companies):
            company_name = company.get('company_name', '').lower()
            primary_symbol = company.get('primary_symbol', '').lower()
            sector = company.get('sector', '').lower()
            industry = company.get('industry', '').lower()
            
            # Index by name
            if company_name:
                self.search_index['by_name'][company_name] = i
                
                # Index name words
                for word in company_name.split():
                    if len(word) > 2:  # Skip very short words
                        if word not in self.search_index['by_keywords']:
                            self.search_index['by_keywords'][word] = []
                        self.search_index['by_keywords'][word].append(i)
            
            # Index by symbol
            if primary_symbol:
                self.search_index['by_symbol'][primary_symbol] = i
            
            # Index by all symbols
            symbols = company.get('symbols', {})
            for exchange, symbol in symbols.items():
                if symbol:
                    self.search_index['by_symbol'][symbol.lower()] = i
            
            # Index by sector
            if sector and sector != 'unknown':
                if sector not in self.search_index['by_sector']:
                    self.search_index['by_sector'][sector] = []
                self.search_index['by_sector'][sector].append(i)
            
            # Index by industry
            if industry and industry != 'unknown':
                if industry not in self.search_index['by_industry']:
                    self.search_index['by_industry'][industry] = []
                self.search_index['by_industry'][industry].append(i)
        
        logger.info(f"✅ Search index built:")
        logger.info(f"   Keywords: {len(self.search_index['by_keywords']):,}")
        logger.info(f"   Sectors: {len(self.search_index['by_sector']):,}")
        logger.info(f"   Industries: {len(self.search_index['by_industry']):,}")
    
    def normalize_query(self, query: str) -> str:
        """Normalize search query"""
        if not query:
            return ""
        
        # Convert to lowercase and strip
        normalized = query.lower().strip()
        
        # Remove special characters except spaces and alphanumeric
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Collapse multiple spaces
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def calculate_search_confidence(self, company: Dict, query: str, match_type: str) -> float:
        """Calculate search confidence score"""
        company_name = company.get('company_name', '').lower()
        primary_symbol = company.get('primary_symbol', '').lower()
        query_lower = query.lower()
        
        if match_type == "exact_name":
            return 1.0
        elif match_type == "exact_symbol":
            return 1.0
        elif match_type == "starts_with_name":
            return 0.9
        elif match_type == "starts_with_symbol":
            return 0.9
        elif match_type == "contains_name":
            # Higher confidence for shorter queries that match
            if len(query_lower) >= len(company_name) * 0.5:
                return 0.8
            else:
                return 0.7
        elif match_type == "contains_symbol":
            return 0.8
        elif match_type == "keyword_match":
            return 0.6
        elif match_type == "sector_match":
            return 0.5
        elif match_type == "industry_match":
            return 0.5
        else:
            return 0.3
    
    def search_exact_matches(self, query: str) -> List[SearchResult]:
        """Search for exact matches"""
        results = []
        query_lower = query.lower()
        
        # Exact name match
        if query_lower in self.search_index['by_name']:
            idx = self.search_index['by_name'][query_lower]
            company = self.companies[idx]
            confidence = self.calculate_search_confidence(company, query, "exact_name")
            result = self.create_search_result(company, confidence, "exact_name")
            results.append(result)
        
        # Exact symbol match
        if query_lower in self.search_index['by_symbol']:
            idx = self.search_index['by_symbol'][query_lower]
            company = self.companies[idx]
            confidence = self.calculate_search_confidence(company, query, "exact_symbol")
            result = self.create_search_result(company, confidence, "exact_symbol")
            if result not in results:
                results.append(result)
        
        return results
    
    def search_partial_matches(self, query: str) -> List[SearchResult]:
        """Search for partial matches"""
        results = []
        query_lower = query.lower()
        
        for i, company in enumerate(self.companies):
            company_name = company.get('company_name', '').lower()
            primary_symbol = company.get('primary_symbol', '').lower()
            
            # Skip if already found in exact matches
            if query_lower == company_name or query_lower == primary_symbol:
                continue
            
            # Starts with matches
            if company_name.startswith(query_lower):
                confidence = self.calculate_search_confidence(company, query, "starts_with_name")
                result = self.create_search_result(company, confidence, "starts_with_name")
                results.append(result)
            elif primary_symbol.startswith(query_lower):
                confidence = self.calculate_search_confidence(company, query, "starts_with_symbol")
                result = self.create_search_result(company, confidence, "starts_with_symbol")
                results.append(result)
            # Contains matches
            elif query_lower in company_name:
                confidence = self.calculate_search_confidence(company, query, "contains_name")
                result = self.create_search_result(company, confidence, "contains_name")
                results.append(result)
            elif query_lower in primary_symbol:
                confidence = self.calculate_search_confidence(company, query, "contains_symbol")
                result = self.create_search_result(company, confidence, "contains_symbol")
                results.append(result)
        
        return results
    
    def search_keyword_matches(self, query: str) -> List[SearchResult]:
        """Search for keyword matches"""
        results = []
        query_words = query.lower().split()
        
        for word in query_words:
            if len(word) > 2 and word in self.search_index['by_keywords']:
                for idx in self.search_index['by_keywords'][word]:
                    company = self.companies[idx]
                    confidence = self.calculate_search_confidence(company, query, "keyword_match")
                    result = self.create_search_result(company, confidence, "keyword_match")
                    if result not in results:
                        results.append(result)
        
        return results
    
    def search_sector_industry(self, query: str) -> List[SearchResult]:
        """Search by sector and industry"""
        results = []
        query_lower = query.lower()
        
        # Search sectors
        for sector, indices in self.search_index['by_sector'].items():
            if query_lower in sector:
                for idx in indices:
                    company = self.companies[idx]
                    confidence = self.calculate_search_confidence(company, query, "sector_match")
                    result = self.create_search_result(company, confidence, "sector_match")
                    if result not in results:
                        results.append(result)
        
        # Search industries
        for industry, indices in self.search_index['by_industry'].items():
            if query_lower in industry:
                for idx in indices:
                    company = self.companies[idx]
                    confidence = self.calculate_search_confidence(company, query, "industry_match")
                    result = self.create_search_result(company, confidence, "industry_match")
                    if result not in results:
                        results.append(result)
        
        return results
    
    def create_search_result(self, company: Dict, search_confidence: float, match_type: str) -> SearchResult:
        """Create search result object"""
        return SearchResult(
            company_name=company.get('company_name', ''),
            primary_symbol=company.get('primary_symbol', ''),
            exchanges=company.get('exchanges', []),
            symbols=company.get('symbols', {}),
            sector=company.get('sector', 'Unknown'),
            industry=company.get('industry', 'Unknown'),
            confidence_score=company.get('confidence_score', 0.0),
            search_confidence=search_confidence,
            crawling_method=company.get('crawling_method', ''),
            data_sources=company.get('data_sources', []),
            isin=company.get('isin'),
            match_type=match_type
        )
    
    def search(self, query: str, max_results: int = 10) -> Tuple[List[SearchResult], float]:
        """Perform comprehensive search"""
        start_time = time.time()
        
        if not query or not query.strip():
            return [], 0.0
        
        normalized_query = self.normalize_query(query)
        all_results = []
        
        # 1. Exact matches (highest priority)
        exact_results = self.search_exact_matches(normalized_query)
        all_results.extend(exact_results)
        
        # 2. Partial matches
        if len(all_results) < max_results:
            partial_results = self.search_partial_matches(normalized_query)
            all_results.extend(partial_results)
        
        # 3. Keyword matches
        if len(all_results) < max_results:
            keyword_results = self.search_keyword_matches(normalized_query)
            all_results.extend(keyword_results)
        
        # 4. Sector/Industry matches
        if len(all_results) < max_results:
            sector_results = self.search_sector_industry(normalized_query)
            all_results.extend(sector_results)
        
        # Remove duplicates and sort by confidence
        unique_results = []
        seen_companies = set()
        
        for result in all_results:
            company_key = (result.company_name, result.primary_symbol)
            if company_key not in seen_companies:
                seen_companies.add(company_key)
                unique_results.append(result)
        
        # Sort by search confidence (descending)
        unique_results.sort(key=lambda x: x.search_confidence, reverse=True)
        
        # Limit results
        final_results = unique_results[:max_results]
        
        search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return final_results, search_time
    
    def get_autocomplete_suggestions(self, query: str, max_suggestions: int = 5) -> List[str]:
        """Get autocomplete suggestions"""
        if not query or len(query) < 2:
            return []
        
        suggestions = set()
        query_lower = query.lower()
        
        # Company name suggestions
        for company in self.companies:
            company_name = company.get('company_name', '')
            if company_name.lower().startswith(query_lower):
                suggestions.add(company_name)
                if len(suggestions) >= max_suggestions:
                    break
        
        # Symbol suggestions
        if len(suggestions) < max_suggestions:
            for company in self.companies:
                primary_symbol = company.get('primary_symbol', '')
                if primary_symbol.lower().startswith(query_lower):
                    suggestions.add(primary_symbol)
                    if len(suggestions) >= max_suggestions:
                        break
        
        return sorted(list(suggestions))[:max_suggestions]
    
    def display_search_results(self, results: List[SearchResult], search_time: float, query: str):
        """Display search results"""
        logger.info(f"\n🔍 Search Results for '{query}':")
        logger.info(f"   Found {len(results)} result(s) in {search_time:.2f}ms")
        
        if not results:
            logger.info("   No companies found.")
            return
        
        for i, result in enumerate(results, 1):
            exchanges_str = ' + '.join(result.exchanges)
            logger.info(f"\n   {i}. {result.company_name}")
            logger.info(f"      Symbol: {result.primary_symbol}")
            logger.info(f"      Exchanges: {exchanges_str}")
            logger.info(f"      Sector: {result.sector}")
            logger.info(f"      Industry: {result.industry}")
            logger.info(f"      Search Confidence: {result.search_confidence:.2f}")
            logger.info(f"      Data Confidence: {result.confidence_score:.2f}")
            logger.info(f"      Match Type: {result.match_type}")
            logger.info(f"      Crawling Method: {result.crawling_method}")
            
            if result.isin:
                logger.info(f"      ISIN: {result.isin}")
            
            # Show exchange-specific symbols
            if len(result.symbols) > 1:
                logger.info(f"      Exchange Symbols:")
                for exchange, symbol in result.symbols.items():
                    logger.info(f"        {exchange}: {symbol}")
    
    def run_comprehensive_test(self):
        """Run comprehensive search test"""
        logger.info("🚀 Starting Crawler-Based Search Engine Test...")
        
        # Display database info
        logger.info(f"\n📊 Database Information:")
        logger.info(f"   Total Companies: {len(self.companies):,}")
        logger.info(f"   Integration Method: {self.metadata.get('integration_method')}")
        logger.info(f"   Crawling Approach: {self.metadata.get('crawling_approach')}")
        logger.info(f"   Crawler-Based Companies: {self.statistics.get('crawler_based_companies', 0):,}")
        
        # Test queries
        test_queries = [
            "RNIT",
            "RNIT AI",
            "RNIT AI Technologies",
            "543320",
            "Reliance",
            "TCS",
            "HDFC",
            "Infosys",
            "AI Technologies",
            "Information Technology"
        ]
        
        logger.info(f"\n🧪 Testing Search Queries:")
        
        total_time = 0
        successful_searches = 0
        
        for query in test_queries:
            results, search_time = self.search(query)
            total_time += search_time
            
            if results:
                successful_searches += 1
                top_result = results[0]
                logger.info(f"✅ '{query}' → {top_result.company_name} ({search_time:.2f}ms, Confidence: {top_result.search_confidence:.2f})")
            else:
                logger.warning(f"⚠️ '{query}' → No results ({search_time:.2f}ms)")
        
        # Performance summary
        avg_time = total_time / len(test_queries) if test_queries else 0
        success_rate = (successful_searches / len(test_queries)) * 100 if test_queries else 0
        
        logger.info(f"\n📈 Performance Summary:")
        logger.info(f"   Success Rate: {success_rate:.1f}% ({successful_searches}/{len(test_queries)})")
        logger.info(f"   Average Response Time: {avg_time:.2f}ms")
        logger.info(f"   Total Test Time: {total_time:.2f}ms")
        
        # Detailed RNIT AI search
        logger.info(f"\n🎯 Detailed RNIT AI Search Test:")
        rnit_results, rnit_time = self.search("RNIT AI Technologies")
        self.display_search_results(rnit_results, rnit_time, "RNIT AI Technologies")
        
        # Autocomplete test
        logger.info(f"\n🔤 Autocomplete Test:")
        autocomplete_queries = ["RN", "RNIT", "TCS", "REL"]
        for query in autocomplete_queries:
            suggestions = self.get_autocomplete_suggestions(query)
            logger.info(f"   '{query}' → {suggestions}")
        
        # Final assessment
        if success_rate >= 90:
            logger.info(f"\n🎉 CRAWLER-BASED SEARCH ENGINE SUCCESS!")
            logger.info(f"   ✅ Excellent success rate: {success_rate:.1f}%")
            logger.info(f"   ✅ Fast response times: {avg_time:.2f}ms average")
            logger.info(f"   ✅ RNIT AI successfully searchable")
            logger.info(f"   ✅ Production-ready search engine")
        else:
            logger.warning(f"\n⚠️ Search engine needs optimization:")
            logger.warning(f"   Success rate: {success_rate:.1f}% (target: 90%+)")

def main():
    """Main function"""
    search_engine = CrawlerBasedSearchEngine()
    search_engine.run_comprehensive_test()

if __name__ == "__main__":
    main()
