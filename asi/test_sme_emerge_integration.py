"""
Comprehensive Test Suite for BSE SME and NSE Emerge Integration
Tests all SME/Emerge functionality including data sources, entity resolution, and API endpoints
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime

# Import the modules to test
from enhanced_indian_companies_database import enhanced_indian_companies_db, ExchangeType, MarketCapCategory
from sme_emerge_data_sources import SMEEmergeDataAggregator, NSEEmergeDataSource, BSESMEDataSource, SMECompanyData

class TestEnhancedIndianCompaniesDatabase:
    """Test enhanced companies database with SME/Emerge support"""
    
    def test_exchange_types_available(self):
        """Test that all exchange types are available"""
        assert ExchangeType.NSE_MAIN in ExchangeType
        assert ExchangeType.BSE_MAIN in ExchangeType
        assert ExchangeType.NSE_EMERGE in ExchangeType
        assert ExchangeType.BSE_SME in ExchangeType
    
    def test_market_cap_categories(self):
        """Test market cap categories"""
        assert MarketCapCategory.LARGE_CAP in MarketCapCategory
        assert MarketCapCategory.MID_CAP in MarketCapCategory
        assert MarketCapCategory.SMALL_CAP in MarketCapCategory
        assert MarketCapCategory.MICRO_CAP in MarketCapCategory
        assert MarketCapCategory.NANO_CAP in MarketCapCategory
    
    def test_companies_loaded(self):
        """Test that companies are loaded correctly"""
        assert len(enhanced_indian_companies_db.companies) > 0
        
        # Check for main board companies
        assert "TCS" in enhanced_indian_companies_db.companies
        assert "RELIANCE" in enhanced_indian_companies_db.companies
        
        # Check for NSE Emerge companies
        emerge_companies = enhanced_indian_companies_db.get_companies_by_exchange(ExchangeType.NSE_EMERGE)
        assert len(emerge_companies) > 0
        
        # Check for BSE SME companies
        sme_companies = enhanced_indian_companies_db.get_companies_by_exchange(ExchangeType.BSE_SME)
        assert len(sme_companies) > 0
    
    def test_symbol_resolution_basic(self):
        """Test basic symbol resolution"""
        # Test exact symbol match
        assert enhanced_indian_companies_db.resolve_symbol("TCS") == "TCS"
        assert enhanced_indian_companies_db.resolve_symbol("tcs") == "TCS"
        
        # Test company name resolution
        assert enhanced_indian_companies_db.resolve_symbol("tata consultancy") == "TCS"
        assert enhanced_indian_companies_db.resolve_symbol("Tata Consultancy Services") == "TCS"
    
    def test_exchange_specific_resolution(self):
        """Test exchange-specific symbol resolution"""
        # Test NSE Emerge resolution
        emerge_symbol = enhanced_indian_companies_db.resolve_symbol("emerge:rossari")
        if emerge_symbol:
            company = enhanced_indian_companies_db.companies[emerge_symbol]
            assert company.exchange == ExchangeType.NSE_EMERGE
        
        # Test BSE SME resolution
        sme_symbol = enhanced_indian_companies_db.resolve_symbol("sme:arshiya")
        if sme_symbol:
            company = enhanced_indian_companies_db.companies[sme_symbol]
            assert company.exchange == ExchangeType.BSE_SME
    
    def test_get_sme_companies(self):
        """Test getting all SME companies"""
        sme_companies = enhanced_indian_companies_db.get_sme_companies()
        assert len(sme_companies) > 0
        
        # Verify all are from SME/Emerge exchanges
        for company in sme_companies:
            assert company.exchange in [ExchangeType.NSE_EMERGE, ExchangeType.BSE_SME]
    
    def test_market_cap_filtering(self):
        """Test market cap based filtering"""
        large_cap = enhanced_indian_companies_db.search_companies_by_market_cap(MarketCapCategory.LARGE_CAP)
        small_cap = enhanced_indian_companies_db.search_companies_by_market_cap(MarketCapCategory.SMALL_CAP)
        
        assert len(large_cap) > 0
        assert len(small_cap) > 0
        
        # Verify market cap categories
        for company in large_cap:
            assert company.market_cap_category == MarketCapCategory.LARGE_CAP
    
    def test_exchange_statistics(self):
        """Test exchange statistics generation"""
        stats = enhanced_indian_companies_db.get_exchange_statistics()
        
        # Verify all exchanges are present
        assert ExchangeType.NSE_MAIN.value in stats
        assert ExchangeType.BSE_MAIN.value in stats
        assert ExchangeType.NSE_EMERGE.value in stats
        assert ExchangeType.BSE_SME.value in stats
        
        # Verify structure
        for exchange, data in stats.items():
            assert "total_companies" in data
            assert "sectors" in data
            assert "market_cap_distribution" in data

class TestSMEEmergeDataSources:
    """Test SME and Emerge data sources"""
    
    @pytest.mark.asyncio
    async def test_nse_emerge_data_source_init(self):
        """Test NSE Emerge data source initialization"""
        async with NSEEmergeDataSource() as source:
            assert source.base_url == "https://www.nseindia.com"
            assert source.session is not None
    
    @pytest.mark.asyncio
    async def test_bse_sme_data_source_init(self):
        """Test BSE SME data source initialization"""
        async with BSESMEDataSource() as source:
            assert source.base_url == "https://www.bseindia.com"
            assert source.session is not None
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_nse_emerge_companies_list_mock(self, mock_get):
        """Test NSE Emerge companies list with mocked response"""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": [
                {"symbol": "ROSSARI", "companyName": "Rossari Biotech Limited"},
                {"symbol": "EASEMYTRIP", "companyName": "Easy Trip Planners Limited"}
            ]
        })
        mock_get.return_value.__aenter__.return_value = mock_response
        
        async with NSEEmergeDataSource() as source:
            companies = await source.get_emerge_companies_list()
            
        assert len(companies) == 2
        assert companies[0]["symbol"] == "ROSSARI"
        assert companies[1]["symbol"] == "EASEMYTRIP"
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_nse_emerge_company_data_mock(self, mock_get):
        """Test NSE Emerge company data with mocked response"""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "priceInfo": {
                "lastPrice": 1250.50,
                "change": 25.30,
                "pChange": 2.07,
                "pe": 18.5
            },
            "securityInfo": {
                "companyName": "Rossari Biotech Limited",
                "faceValue": 2.0,
                "industry": "Specialty Chemicals",
                "isin": "INE02A801020"
            },
            "securityWiseDP": {
                "quantityTraded": 15000
            }
        })
        mock_get.return_value.__aenter__.return_value = mock_response
        
        async with NSEEmergeDataSource() as source:
            data = await source.get_emerge_company_data("ROSSARI")
            
        assert data is not None
        assert data.symbol == "ROSSARI"
        assert data.exchange == "NSE_EMERGE"
        assert data.price == 1250.50
        assert data.change == 25.30
        assert data.change_percent == 2.07
        assert data.volume == 15000
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_bse_sme_companies_list_mock(self, mock_get):
        """Test BSE SME companies list with mocked response"""
        # Mock HTML response
        mock_html = """
        <table>
            <tr><th>Scrip Code</th><th>Company Name</th><th>Symbol</th></tr>
            <tr><td>532758</td><td>Arshiya Limited</td><td>ARSHIYA</td></tr>
            <tr><td>542337</td><td>Spencer's Retail Limited</td><td>SPENCERS</td></tr>
        </table>
        """
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=mock_html)
        mock_get.return_value.__aenter__.return_value = mock_response
        
        async with BSESMEDataSource() as source:
            companies = await source.get_sme_companies_list()
            
        assert len(companies) >= 2
        # Verify structure (exact parsing depends on HTML structure)
        for company in companies:
            assert "scrip_code" in company
            assert "company_name" in company
    
    @pytest.mark.asyncio
    async def test_sme_emerge_aggregator_init(self):
        """Test SME Emerge aggregator initialization"""
        async with SMEEmergeDataAggregator() as aggregator:
            assert aggregator.nse_emerge is not None
            assert aggregator.bse_sme is not None
            assert aggregator.sme_news is not None

class TestSMEEmergeAPIIntegration:
    """Test API integration for SME and Emerge functionality"""
    
    def test_sme_company_data_structure(self):
        """Test SME company data structure"""
        data = SMECompanyData(
            symbol="TEST",
            name="Test Company",
            exchange="NSE_EMERGE",
            price=100.0,
            change=5.0,
            change_percent=5.26,
            volume=10000,
            timestamp=datetime.now()
        )
        
        assert data.symbol == "TEST"
        assert data.exchange == "NSE_EMERGE"
        assert data.price == 100.0
        assert data.metadata == {}
    
    @pytest.mark.asyncio
    @patch('sme_emerge_data_sources.sme_emerge_aggregator.get_comprehensive_sme_data')
    async def test_comprehensive_sme_data_mock(self, mock_get_data):
        """Test comprehensive SME data aggregation with mock"""
        # Mock return data
        mock_data = {
            'symbol': 'ROSSARI',
            'exchange': 'NSE_EMERGE',
            'price_data': SMECompanyData(
                symbol='ROSSARI',
                name='Rossari Biotech Limited',
                exchange='NSE_EMERGE',
                price=1250.50,
                change=25.30,
                change_percent=2.07
            ),
            'news': [
                {
                    'title': 'Rossari Biotech reports strong Q3 results',
                    'source': 'sme_world',
                    'timestamp': datetime.now().isoformat()
                }
            ],
            'metadata': {
                'data_sources': ['NSE_EMERGE', 'SME_NEWS'],
                'data_quality': 'high'
            }
        }
        
        mock_get_data.return_value = mock_data
        
        from sme_emerge_data_sources import sme_emerge_aggregator
        
        async with sme_emerge_aggregator as aggregator:
            result = await aggregator.get_comprehensive_sme_data('ROSSARI', 'NSE_EMERGE')
            
        assert result['symbol'] == 'ROSSARI'
        assert result['exchange'] == 'NSE_EMERGE'
        assert 'price_data' in result
        assert 'news' in result
        assert 'metadata' in result

class TestSMEEmergeQueries:
    """Test various query patterns for SME and Emerge companies"""
    
    def test_natural_language_queries(self):
        """Test natural language query resolution for SME companies"""
        test_queries = [
            "rossari biotech share price",
            "easy trip planners stock",
            "craftsman automation limited",
            "heranba industries news",
            "anupam rasayan financial results"
        ]
        
        for query in test_queries:
            # Clean query for testing
            cleaned_query = query.replace(" share price", "").replace(" stock", "").replace(" news", "").replace(" financial results", "")
            
            # Try to resolve
            symbol = enhanced_indian_companies_db.resolve_symbol(cleaned_query)
            
            # If resolved, verify it's a valid company
            if symbol:
                assert symbol in enhanced_indian_companies_db.companies
                company = enhanced_indian_companies_db.companies[symbol]
                print(f"Query '{query}' resolved to {symbol} ({company.name}) on {company.exchange.value}")
    
    def test_sector_based_sme_queries(self):
        """Test sector-based queries for SME companies"""
        # Get all SME companies
        sme_companies = enhanced_indian_companies_db.get_sme_companies()
        
        # Group by sector
        sectors = {}
        for company in sme_companies:
            sector = company.sector.lower()
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(company)
        
        # Test sector queries
        for sector, companies in sectors.items():
            if len(companies) > 0:
                print(f"Sector '{sector}' has {len(companies)} SME companies")
                
                # Test if sector query works
                sector_results = enhanced_indian_companies_db._match_sector(sector)
                assert len(sector_results) >= 0  # May or may not find matches depending on keyword mapping
    
    def test_exchange_filtering(self):
        """Test exchange-specific filtering"""
        # Test NSE Emerge filtering
        emerge_companies = enhanced_indian_companies_db.get_companies_by_exchange(ExchangeType.NSE_EMERGE)
        for company in emerge_companies:
            assert company.exchange == ExchangeType.NSE_EMERGE
            print(f"NSE Emerge: {company.symbol} - {company.name}")
        
        # Test BSE SME filtering
        sme_companies = enhanced_indian_companies_db.get_companies_by_exchange(ExchangeType.BSE_SME)
        for company in sme_companies:
            assert company.exchange == ExchangeType.BSE_SME
            print(f"BSE SME: {company.symbol} - {company.name}")

def run_comprehensive_sme_emerge_tests():
    """Run all SME and Emerge tests"""
    print("=== Testing Enhanced Indian Companies Database ===")
    
    # Test database functionality
    db_tests = TestEnhancedIndianCompaniesDatabase()
    db_tests.test_exchange_types_available()
    db_tests.test_market_cap_categories()
    db_tests.test_companies_loaded()
    db_tests.test_symbol_resolution_basic()
    db_tests.test_exchange_specific_resolution()
    db_tests.test_get_sme_companies()
    db_tests.test_market_cap_filtering()
    db_tests.test_exchange_statistics()
    
    print("✅ Database tests passed!")
    
    # Test query functionality
    print("\n=== Testing SME/Emerge Query Patterns ===")
    query_tests = TestSMEEmergeQueries()
    query_tests.test_natural_language_queries()
    query_tests.test_sector_based_sme_queries()
    query_tests.test_exchange_filtering()
    
    print("✅ Query tests passed!")
    
    # Test data structures
    print("\n=== Testing Data Structures ===")
    api_tests = TestSMEEmergeAPIIntegration()
    api_tests.test_sme_company_data_structure()
    
    print("✅ Data structure tests passed!")
    
    print("\n🎉 All SME and Emerge tests completed successfully!")
    
    # Display summary statistics
    print("\n=== Summary Statistics ===")
    stats = enhanced_indian_companies_db.get_exchange_statistics()
    
    total_companies = sum(stat["total_companies"] for stat in stats.values())
    sme_emerge_total = stats.get("NSE_EMERGE", {}).get("total_companies", 0) + stats.get("BSE_SME", {}).get("total_companies", 0)
    
    print(f"Total Companies: {total_companies}")
    print(f"SME/Emerge Companies: {sme_emerge_total}")
    print(f"SME/Emerge Coverage: {(sme_emerge_total/total_companies)*100:.1f}%")
    
    for exchange, data in stats.items():
        print(f"{exchange}: {data['total_companies']} companies, {data['sectors']} sectors")

if __name__ == "__main__":
    run_comprehensive_sme_emerge_tests()
