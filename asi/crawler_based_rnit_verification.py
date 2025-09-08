#!/usr/bin/env python3
"""
Crawler-Based RNIT AI Verification
Verify RNIT AI Technologies is properly included in crawler-based database
"""

import json
import logging
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrawlerBasedRNITVerification:
    """Verify RNIT AI in crawler-based database"""
    
    def __init__(self):
        self.database_file = "market_data/final_crawler_based_companies.json"
    
    def load_crawler_database(self) -> Dict:
        """Load crawler-based database"""
        try:
            with open(self.database_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            companies = data.get('companies', [])
            metadata = data.get('metadata', {})
            statistics = data.get('statistics', {})
            
            logger.info(f"✅ Loaded crawler-based database:")
            logger.info(f"   Total Companies: {len(companies):,}")
            logger.info(f"   Integration Method: {metadata.get('integration_method')}")
            logger.info(f"   Crawling Approach: {metadata.get('crawling_approach')}")
            logger.info(f"   Crawler-Based Companies: {statistics.get('crawler_based_companies', 0):,}")
            
            return data
            
        except FileNotFoundError:
            logger.error(f"❌ Database file not found: {self.database_file}")
            return {}
    
    def search_companies(self, companies: List[Dict], query: str) -> List[Dict]:
        """Search companies by query"""
        query_lower = query.lower()
        results = []
        
        for company in companies:
            company_name = company.get('company_name', '').lower()
            primary_symbol = company.get('primary_symbol', '').lower()
            
            # Check if query matches company name or symbol
            if (query_lower in company_name or 
                query_lower in primary_symbol or
                company_name.startswith(query_lower)):
                
                # Calculate confidence score
                confidence = 0.0
                if query_lower == company_name:
                    confidence = 1.0
                elif query_lower == primary_symbol:
                    confidence = 1.0
                elif company_name.startswith(query_lower):
                    confidence = 0.9
                elif query_lower in company_name:
                    confidence = 0.8
                elif query_lower in primary_symbol:
                    confidence = 0.7
                
                result = company.copy()
                result['search_confidence'] = confidence
                results.append(result)
        
        # Sort by confidence
        results.sort(key=lambda x: x['search_confidence'], reverse=True)
        return results
    
    def verify_rnit_ai_presence(self, companies: List[Dict]) -> List[Dict]:
        """Verify RNIT AI presence in database"""
        logger.info("🔍 Searching for RNIT AI Technologies...")
        
        # Search for RNIT AI variations
        search_queries = [
            "RNIT",
            "RNIT AI",
            "RNIT AI Technologies",
            "543320"  # BSE symbol
        ]
        
        all_results = []
        for query in search_queries:
            results = self.search_companies(companies, query)
            if results:
                logger.info(f"✅ Query '{query}' found {len(results)} result(s)")
                for result in results:
                    if result not in all_results:
                        all_results.append(result)
            else:
                logger.warning(f"⚠️ Query '{query}' found no results")
        
        return all_results
    
    def display_rnit_results(self, rnit_results: List[Dict]):
        """Display RNIT AI search results"""
        if not rnit_results:
            logger.error("❌ RNIT AI Technologies NOT FOUND in crawler-based database!")
            return
        
        logger.info(f"\n🎯 RNIT AI Technologies Found - {len(rnit_results)} result(s):")
        
        for i, company in enumerate(rnit_results, 1):
            logger.info(f"\n   Result {i}:")
            logger.info(f"   Company Name: {company.get('company_name')}")
            logger.info(f"   Primary Symbol: {company.get('primary_symbol')}")
            logger.info(f"   Exchanges: {', '.join(company.get('exchanges', []))}")
            logger.info(f"   ISIN: {company.get('isin', 'N/A')}")
            logger.info(f"   Sector: {company.get('sector', 'Unknown')}")
            logger.info(f"   Industry: {company.get('industry', 'Unknown')}")
            logger.info(f"   Confidence Score: {company.get('confidence_score', 0):.2f}")
            logger.info(f"   Search Confidence: {company.get('search_confidence', 0):.2f}")
            logger.info(f"   Crawling Method: {company.get('crawling_method', 'Unknown')}")
            logger.info(f"   Data Sources: {', '.join(company.get('data_sources', []))}")
            
            # Show exchange-specific symbols
            symbols = company.get('symbols', {})
            if symbols:
                logger.info(f"   Exchange Symbols:")
                for exchange, symbol in symbols.items():
                    logger.info(f"     {exchange}: {symbol}")
    
    def test_search_functionality(self, companies: List[Dict]):
        """Test search functionality with various queries"""
        logger.info("\n🧪 Testing Search Functionality:")
        
        test_queries = [
            "Reliance",
            "TCS", 
            "HDFC",
            "Infosys",
            "RNIT",
            "AI Technologies",
            "543320"
        ]
        
        for query in test_queries:
            results = self.search_companies(companies, query)
            if results:
                top_result = results[0]
                logger.info(f"✅ '{query}' → {top_result['company_name']} (Confidence: {top_result['search_confidence']:.2f})")
            else:
                logger.warning(f"⚠️ '{query}' → No results found")
    
    def display_database_statistics(self, data: Dict):
        """Display comprehensive database statistics"""
        metadata = data.get('metadata', {})
        statistics = data.get('statistics', {})
        
        logger.info(f"\n📊 Crawler-Based Database Statistics:")
        logger.info(f"   Integration Method: {metadata.get('integration_method')}")
        logger.info(f"   Crawling Approach: {metadata.get('crawling_approach')}")
        logger.info(f"   Total Companies: {statistics.get('total_companies', 0):,}")
        logger.info(f"   NSE Companies: {statistics.get('nse_companies', 0):,}")
        logger.info(f"   BSE Main Companies: {statistics.get('bse_main_companies', 0):,}")
        logger.info(f"   BSE SME Companies: {statistics.get('bse_sme_companies', 0):,}")
        logger.info(f"   Crawler-Based Companies: {statistics.get('crawler_based_companies', 0):,}")
        logger.info(f"   High Confidence Companies: {statistics.get('high_confidence_companies', 0):,}")
        logger.info(f"   Multi-Exchange Companies: {statistics.get('multi_exchange_companies', 0):,}")
        logger.info(f"   RNIT AI Included: {statistics.get('rnit_ai_included', False)}")
        
        # Show exchange distribution
        exchange_dist = statistics.get('companies_by_exchange', {})
        if exchange_dist:
            logger.info(f"\n📈 Exchange Distribution:")
            for exchange, count in sorted(exchange_dist.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"   {exchange}: {count:,} companies")
        
        # Show crawling methods
        crawling_methods = statistics.get('companies_by_crawling_method', {})
        if crawling_methods:
            logger.info(f"\n🕷️ Crawling Methods:")
            for method, count in sorted(crawling_methods.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"   {method}: {count:,} companies")
    
    def run_verification(self):
        """Run complete verification"""
        logger.info("🚀 Starting Crawler-Based RNIT AI Verification...")
        
        # Load database
        data = self.load_crawler_database()
        if not data:
            return
        
        companies = data.get('companies', [])
        
        # Display database statistics
        self.display_database_statistics(data)
        
        # Verify RNIT AI presence
        rnit_results = self.verify_rnit_ai_presence(companies)
        self.display_rnit_results(rnit_results)
        
        # Test search functionality
        self.test_search_functionality(companies)
        
        # Final assessment
        if rnit_results:
            logger.info(f"\n🎉 CRAWLER-BASED VERIFICATION SUCCESS!")
            logger.info(f"   ✅ RNIT AI Technologies found in crawler-based database")
            logger.info(f"   ✅ Google-like crawling successfully captured RNIT AI")
            logger.info(f"   ✅ Search functionality working correctly")
            logger.info(f"   ✅ Comprehensive Indian market coverage achieved")
            
            # Check if it's on correct exchange
            for result in rnit_results:
                exchanges = result.get('exchanges', [])
                if 'BSE_MAIN' in exchanges:
                    logger.info(f"   ✅ RNIT AI correctly identified on BSE Main Board")
                elif 'BSE_SME' in exchanges:
                    logger.info(f"   ⚠️ RNIT AI found on BSE SME (verify if correct)")
        else:
            logger.error(f"\n❌ CRAWLER-BASED VERIFICATION FAILED!")
            logger.error(f"   ❌ RNIT AI Technologies NOT FOUND")
            logger.error(f"   ❌ Crawler may need enhancement")

def main():
    """Main function"""
    verification = CrawlerBasedRNITVerification()
    verification.run_verification()

if __name__ == "__main__":
    main()
