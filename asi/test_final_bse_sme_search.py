#!/usr/bin/env python3
"""
Test Final BSE SME Search with RNIT AI
"""

import json
import sys
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SearchResult:
    name: str
    symbol: str
    exchange: str
    exchanges: List[str]
    category: str
    confidence_score: float
    isin: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None

class FinalBSESMESearchEngine:
    """Final search engine with complete BSE SME data"""
    
    def __init__(self):
        self.companies = []
        self.load_enhanced_data()
    
    def load_enhanced_data(self):
        """Load enhanced BSE SME data"""
        try:
            with open('market_data/enhanced_bse_sme_complete.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ Loaded {data['metadata']['total_companies']} enhanced BSE SME companies")
            print(f"   RNIT AI Included: {data['metadata']['includes_rnit_ai']}")
            print(f"   Source: {data['metadata']['source']}")
            print()
            
            self.companies = data['companies']
            
        except FileNotFoundError:
            print("❌ Enhanced BSE SME data file not found")
            sys.exit(1)
    
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search for companies"""
        query = query.lower().strip()
        results = []
        
        print(f"🔎 Searching for '{query}' in {len(self.companies)} BSE SME companies...")
        
        for company in self.companies:
            confidence = 0.0
            
            # Exact symbol match
            for exchange, symbol in company['symbols'].items():
                if query == symbol.lower():
                    confidence = 1.0
                    break
                elif query in symbol.lower():
                    confidence = max(confidence, 0.9)
            
            # Company name matching
            company_name = company['company_name'].lower()
            if query == company_name:
                confidence = max(confidence, 0.95)
            elif query in company_name:
                # Check if all words in query are in company name
                query_words = query.split()
                if all(word in company_name for word in query_words):
                    confidence = max(confidence, 0.8)
                else:
                    confidence = max(confidence, 0.6)
            
            # Primary symbol matching
            if query == company['primary_symbol'].lower():
                confidence = max(confidence, 1.0)
            elif query in company['primary_symbol'].lower():
                confidence = max(confidence, 0.9)
            
            if confidence > 0.5:  # Threshold for inclusion
                result = SearchResult(
                    name=company['company_name'],
                    symbol=company['primary_symbol'],
                    exchange=company['exchanges'][0],
                    exchanges=company['exchanges'],
                    category=company['market_cap_category'],
                    confidence_score=confidence,
                    isin=company.get('isin'),
                    sector=company.get('sector'),
                    industry=company.get('industry')
                )
                results.append(result)
        
        # Sort by confidence score
        results.sort(key=lambda x: x.confidence_score, reverse=True)
        
        print(f"   Found {len(results)} results")
        return results[:limit]

def main():
    """Test the final BSE SME search engine"""
    print("🚀 Testing Final BSE SME Search Engine with RNIT AI")
    print("=" * 60)
    print()
    
    engine = FinalBSESMESearchEngine()
    
    # Test searches
    test_queries = [
        "RNIT AI",
        "RNIT",
        "AI Technologies",
        "InfoBeans",
        "KPI Green",
        "Valiant",
        "Premier",
        "3BFILMS",
        "AASHKA",
        "BONDADA"
    ]
    
    for query in test_queries:
        print(f"🔍 Testing query: '{query}'")
        print("-" * 40)
        
        results = engine.search(query, limit=3)
        
        if results:
            for i, result in enumerate(results, 1):
                exchanges_str = ' + '.join(result.exchanges) if len(result.exchanges) > 1 else result.exchange
                print(f"{i}. {result.name}")
                print(f"   Symbol: {result.symbol} | Exchange: {exchanges_str}")
                print(f"   Category: {result.category} | Confidence: {result.confidence_score:.2f}")
                if result.sector and result.sector != "Unknown":
                    print(f"   Sector: {result.sector}")
                if result.isin:
                    print(f"   ISIN: {result.isin}")
                print()
        else:
            print("   ❌ No results found")
            print()
        
        print()
    
    # Special RNIT AI verification
    print("🎯 SPECIAL VERIFICATION: RNIT AI Technologies")
    print("=" * 50)
    
    rnit_results = engine.search("RNIT AI", limit=1)
    if rnit_results and rnit_results[0].confidence_score >= 0.8:
        result = rnit_results[0]
        print("✅ SUCCESS: RNIT AI Technologies found with high confidence!")
        print(f"   Company: {result.name}")
        print(f"   Symbol: {result.symbol}")
        print(f"   Exchange: {result.exchange}")
        print(f"   Confidence: {result.confidence_score:.2f}")
        print(f"   Sector: {result.sector}")
        print(f"   Industry: {result.industry}")
        if result.isin:
            print(f"   ISIN: {result.isin}")
    else:
        print("❌ FAILED: RNIT AI Technologies not found or low confidence")
    
    print()
    print("🎉 BSE SME Search Engine Test Completed!")

if __name__ == "__main__":
    main()
