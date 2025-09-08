#!/usr/bin/env python3
"""
RNIT AI Search Test - Verify RNIT AI is correctly found in BSE Main Board
"""

import json
import logging
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RNITAISearchTest:
    """Test class to verify RNIT AI is correctly found"""
    
    def __init__(self):
        self.data_file = "market_data/corrected_unified_companies.json"
        self.companies = []
        self.metadata = {}
    
    def load_data(self):
        """Load the corrected unified companies data"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.companies = data['companies']
            self.metadata = data['metadata']
            
            logger.info(f"✅ Loaded {len(self.companies)} companies")
            logger.info(f"   NSE Companies: {self.metadata['nse_companies']}")
            logger.info(f"   BSE Main Companies: {self.metadata['bse_main_companies']}")
            logger.info(f"   BSE SME Companies: {self.metadata['bse_sme_companies']}")
            logger.info(f"   Multi-Exchange Companies: {self.metadata['multi_exchange_companies']}")
            logger.info(f"   RNIT AI Included: {self.metadata['includes_rnit_ai']}")
            
        except FileNotFoundError:
            logger.error(f"❌ Data file not found: {self.data_file}")
            return False
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
        
        return True
    
    def search_company(self, query: str) -> List[Tuple[Dict, float]]:
        """Search for companies with confidence scoring"""
        query = query.lower()
        results = []
        
        for company in self.companies:
            confidence = 0.0
            
            # Exact symbol match (highest confidence)
            if query == company['primary_symbol'].lower():
                confidence = 1.0
            elif query in company['primary_symbol'].lower():
                confidence = 0.9
            
            # Exact company name match
            elif query == company['company_name'].lower():
                confidence = 1.0
            elif query in company['company_name'].lower():
                confidence = 0.8
            
            # Symbol matches in exchanges
            elif any(query == symbol.lower() for symbol in company['symbols'].values()):
                confidence = 0.9
            elif any(query in symbol.lower() for symbol in company['symbols'].values()):
                confidence = 0.7
            
            # Partial matches
            elif any(word in company['company_name'].lower() for word in query.split()):
                confidence = 0.6
            
            if confidence > 0:
                results.append((company, confidence))
        
        # Sort by confidence (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def test_rnit_ai_search(self):
        """Test RNIT AI search specifically"""
        logger.info("🎯 Testing RNIT AI Search...")
        
        test_queries = [
            "RNIT",
            "rnit",
            "RNIT AI",
            "rnit ai",
            "RNIT AI Technologies",
            "rnit ai technologies",
            "543320"
        ]
        
        for query in test_queries:
            logger.info(f"\n🔍 Query: '{query}'")
            results = self.search_company(query)
            
            if results:
                logger.info(f"   Found {len(results)} results:")
                for i, (company, confidence) in enumerate(results[:3], 1):  # Show top 3
                    exchanges_str = ' + '.join(company['exchanges'])
                    logger.info(f"   {i}. {company['company_name']} ({company['primary_symbol']})")
                    logger.info(f"      Confidence: {confidence:.2f}")
                    logger.info(f"      Exchanges: {exchanges_str}")
                    logger.info(f"      Sector: {company['sector']}")
                    logger.info(f"      ISIN: {company['isin']}")
                    
                    # Check if this is RNIT AI
                    if "RNIT" in company['company_name'] and "Technologies" in company['company_name']:
                        logger.info(f"      ✅ RNIT AI FOUND!")
                        
                        # Verify it's on BSE Main Board
                        if "BSE_MAIN" in company['exchanges']:
                            logger.info(f"      ✅ Correctly listed on BSE Main Board")
                        else:
                            logger.warning(f"      ⚠️ Not on BSE Main Board: {company['exchanges']}")
            else:
                logger.warning(f"   ❌ No results found for '{query}'")
    
    def get_exchange_statistics(self):
        """Get statistics by exchange"""
        exchange_stats = {}
        
        for company in self.companies:
            for exchange in company['exchanges']:
                if exchange not in exchange_stats:
                    exchange_stats[exchange] = []
                exchange_stats[exchange].append(company['company_name'])
        
        logger.info(f"\n📊 Exchange Statistics:")
        for exchange, companies in exchange_stats.items():
            logger.info(f"   {exchange}: {len(companies)} companies")
            
            # Show RNIT companies in each exchange
            rnit_companies = [c for c in companies if "RNIT" in c.upper()]
            if rnit_companies:
                logger.info(f"      RNIT companies: {rnit_companies}")
    
    def verify_bse_main_board_companies(self):
        """Verify BSE Main Board companies"""
        logger.info(f"\n🏢 BSE Main Board Companies:")
        
        bse_main_companies = [c for c in self.companies if "BSE_MAIN" in c['exchanges']]
        logger.info(f"   Total BSE Main Board Companies: {len(bse_main_companies)}")
        
        for company in bse_main_companies:
            logger.info(f"   • {company['company_name']} ({company['primary_symbol']}) - {company['sector']}")
            
            # Highlight RNIT AI
            if "RNIT" in company['company_name']:
                logger.info(f"     🎯 *** RNIT AI FOUND ON BSE MAIN BOARD! ***")
    
    def run_comprehensive_test(self):
        """Run comprehensive RNIT AI test"""
        logger.info("🚀 Starting Comprehensive RNIT AI Search Test...")
        print()
        
        # Load data
        if not self.load_data():
            return False
        
        # Test RNIT AI search
        self.test_rnit_ai_search()
        
        # Get exchange statistics
        self.get_exchange_statistics()
        
        # Verify BSE Main Board companies
        self.verify_bse_main_board_companies()
        
        # Final verification
        logger.info(f"\n🎉 TEST SUMMARY:")
        logger.info(f"   Total Companies: {len(self.companies)}")
        logger.info(f"   RNIT AI Included: {self.metadata['includes_rnit_ai']}")
        
        # Check if RNIT AI is found
        rnit_companies = [c for c in self.companies if "RNIT" in c['company_name'].upper()]
        if rnit_companies:
            logger.info(f"   ✅ RNIT AI COMPANIES FOUND: {len(rnit_companies)}")
            for company in rnit_companies:
                exchanges_str = ' + '.join(company['exchanges'])
                logger.info(f"      • {company['company_name']} on {exchanges_str}")
        else:
            logger.error(f"   ❌ RNIT AI NOT FOUND!")
        
        logger.info(f"\n✅ Comprehensive RNIT AI test completed!")
        return True

def main():
    """Main function"""
    test = RNITAISearchTest()
    test.run_comprehensive_test()

if __name__ == "__main__":
    main()
