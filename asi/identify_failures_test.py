"""
Identify specific failures to achieve 100% success rate
"""

import asyncio
from universal_financial_search_engine import UniversalFinancialSearchEngine

async def identify_failures():
    """Identify which specific queries are failing"""
    print('🔍 IDENTIFYING FAILED QUERIES FOR 100% SUCCESS')
    print('=' * 60)
    
    engine = UniversalFinancialSearchEngine()
    await engine.start_background_updates()
    
    # Wait for data collection
    await asyncio.sleep(3)
    
    # All test queries from the comprehensive test
    all_queries = [
        # Stocks
        "reliance", "tcs", "hdfc bank", "infosys", "icici bank", "wipro", "bharti airtel", "itc",
        
        # Mutual funds
        "sbi bluechip", "hdfc top 100", "parag parikh flexi cap", "axis bluechip", 
        "mirae asset large cap", "icici pru bluechip", "kotak emerging equity", "nippon india small cap",
        
        # ETFs
        "nifty bees", "gold bees", "bank bees", "junior bees", "liquid bees", "icicipru nifty", "hdfcnifty",
        
        # Indices
        "nifty", "sensex", "bank nifty", "nifty it", "nifty auto", "nifty pharma", "nifty fmcg",
        
        # Sectors
        "banking", "it sector", "pharma", "auto sector", "fmcg companies", "metal stocks", "energy sector"
    ]
    
    failed_queries = []
    successful_queries = []
    
    print('🧪 TESTING ALL QUERIES:')
    print('-' * 40)
    
    for query in all_queries:
        try:
            result = await engine.search_universal(query, max_suggestions=5)
            
            if result.instruments_found > 0:
                successful_queries.append(query)
                print(f'✅ "{query}" → {result.suggestions[0].name}')
            else:
                failed_queries.append(query)
                print(f'❌ "{query}" → NO RESULTS')
                
        except Exception as e:
            failed_queries.append(query)
            print(f'❌ "{query}" → ERROR: {e}')
    
    print(f'\n📊 RESULTS SUMMARY:')
    print(f'✅ Successful: {len(successful_queries)}/{len(all_queries)}')
    print(f'❌ Failed: {len(failed_queries)}/{len(all_queries)}')
    print(f'📈 Success Rate: {(len(successful_queries)/len(all_queries))*100:.1f}%')
    
    if failed_queries:
        print(f'\n❌ FAILED QUERIES TO FIX:')
        for i, query in enumerate(failed_queries, 1):
            print(f'   {i}. "{query}"')
            
        print(f'\n🔧 RECOMMENDATIONS:')
        for query in failed_queries:
            if "sector" in query or "companies" in query or "stocks" in query:
                print(f'   • "{query}": Add sector-specific search logic')
            elif any(word in query for word in ["nifty", "sensex", "bank"]):
                print(f'   • "{query}": Add index to indices database')
            elif any(word in query for word in ["bees", "etf"]):
                print(f'   • "{query}": Add ETF to ETFs database')
            else:
                print(f'   • "{query}": Add to enhanced companies database')
    else:
        print(f'\n🎉 ALL QUERIES SUCCESSFUL! 100% SUCCESS RATE ACHIEVED!')
    
    engine.stop_background_updates()

if __name__ == "__main__":
    asyncio.run(identify_failures())
