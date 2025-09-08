#!/usr/bin/env python3
"""
Test Enhanced Search Engine with BSE SME + NSE Data
"""

import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class SearchResult:
    symbol: str
    name: str
    exchange: str
    exchanges: List[str]
    confidence_score: float
    isin: Optional[str] = None
    market_cap_category: str = "Unknown"

class EnhancedSearchEngine:
    """Enhanced search engine with BSE SME + NSE support"""
    
    def __init__(self):
        self.companies_data = {}
        self.load_enhanced_data()
    
    def load_enhanced_data(self):
        """Load enhanced unified companies data"""
        try:
            with open("market_data/enhanced_unified_companies.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.companies_data = data.get("companies", {})
            print(f"✅ Loaded {len(self.companies_data)} companies from enhanced database")
        except Exception as e:
            print(f"❌ Error loading enhanced data: {e}")
    
    def normalize_query(self, query: str) -> str:
        """Normalize search query"""
        if not query:
            return ""
        
        normalized = query.lower().strip()
        # Remove special characters but keep spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def calculate_confidence(self, query: str, company: Dict) -> float:
        """Calculate confidence score for search match"""
        query_norm = self.normalize_query(query)
        name_norm = self.normalize_query(company.get("company_name", ""))
        symbol_norm = company.get("primary_symbol", "").lower()
        
        # Exact symbol match
        if query_norm == symbol_norm:
            return 1.0
        
        # Symbol contains query
        if query_norm in symbol_norm:
            return 0.9
        
        # Exact name match
        if query_norm == name_norm:
            return 0.95
        
        # Name contains all query words
        query_words = query_norm.split()
        name_words = name_norm.split()
        
        if all(any(word in name_word for name_word in name_words) for word in query_words):
            return 0.8
        
        # Partial name match
        if any(word in name_norm for word in query_words):
            return 0.6
        
        return 0.0
    
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search for companies"""
        results = []
        
        print(f"Debug: Searching for '{query}' in {len(self.companies_data)} companies")
        
        for symbol, company in self.companies_data.items():
            confidence = self.calculate_confidence(query, company)
            
            # Debug for RNIT AI specifically
            if "rnit" in company.get("company_name", "").lower():
                print(f"Debug: Found RNIT company: {company.get('company_name')} - Confidence: {confidence}")
            
            if confidence > 0.3:  # Lower threshold for testing
                result = SearchResult(
                    symbol=company.get("primary_symbol", symbol),
                    name=company.get("company_name", ""),
                    exchange=company.get("exchanges", ["Unknown"])[0],  # Primary exchange
                    exchanges=company.get("exchanges", []),
                    confidence_score=confidence,
                    isin=company.get("isin"),
                    market_cap_category=company.get("market_cap_category", "Unknown")
                )
                results.append(result)
        
        print(f"Debug: Found {len(results)} results")
        
        # Sort by confidence score
        results.sort(key=lambda x: x.confidence_score, reverse=True)
        return results[:limit]

def test_search_queries():
    """Test various search queries"""
    engine = EnhancedSearchEngine()
    
    test_queries = [
        "RNIT AI",
        "RNIT",
        "AI Technologies",
        "Reliance",
        "TCS",
        "Infosys",
        "BSE SME",
        "Manorama",
        "InfoBeans",
        "KPI Green"
    ]
    
    print("\n🔍 TESTING ENHANCED SEARCH ENGINE WITH BSE SME + NSE DATA\n")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n🔎 Search Query: '{query}'")
        print("-" * 50)
        
        results = engine.search(query, limit=5)
        
        if results:
            for i, result in enumerate(results, 1):
                exchanges_str = " + ".join(result.exchanges) if len(result.exchanges) > 1 else result.exchange
                print(f"{i}. {result.name}")
                print(f"   Symbol: {result.symbol} | Exchange: {exchanges_str}")
                print(f"   Category: {result.market_cap_category} | Confidence: {result.confidence_score:.2f}")
                if result.isin:
                    print(f"   ISIN: {result.isin}")
                print()
        else:
            print("❌ No results found")
            print()
    
    print("=" * 80)
    print("✅ Enhanced search engine test completed!")

if __name__ == "__main__":
    test_search_queries()
