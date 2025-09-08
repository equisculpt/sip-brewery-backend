"""
Comprehensive Test Suite for ASI Finance Search Engine
Tests all components: Entity Resolution, Crawling, Semantic Search, ASI Integration
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime
import json

# Import all components
from asi.enhanced_finance_engine_api import app
from asi.indian_companies_database import indian_companies_db
from asi.advanced_crawling_engine import AdvancedCrawlingEngine, CrawlTask, CrawlResult
from asi.semantic_search_layer import semantic_search_engine, SearchQuery, SearchResult
from asi.asi_integration_layer import MarketInsight, PredictiveSignal, InsightType, AlertPriority

# Test client
client = TestClient(app)

class TestEntityResolution:
    """Test entity resolution functionality"""
    
    def test_exact_symbol_match(self):
        """Test exact symbol matching"""
        assert indian_companies_db.resolve_symbol("TCS") == "TCS"
        assert indian_companies_db.resolve_symbol("tcs") == "TCS"
        assert indian_companies_db.resolve_symbol("INFY") == "INFY"
    
    def test_company_name_resolution(self):
        """Test company name to symbol resolution"""
        assert indian_companies_db.resolve_symbol("Tata Consultancy Services") == "TCS"
        assert indian_companies_db.resolve_symbol("tata consultancy") == "TCS"
        assert indian_companies_db.resolve_symbol("Infosys") == "INFY"
        assert indian_companies_db.resolve_symbol("infosys limited") == "INFY"
    
    def test_query_with_suffixes(self):
        """Test queries with price/stock suffixes"""
        assert indian_companies_db.resolve_symbol("TCS share price") == "TCS"
        assert indian_companies_db.resolve_symbol("tcs stock price") == "TCS"
        assert indian_companies_db.resolve_symbol("infosys price") == "INFY"
    
    def test_fuzzy_matching(self):
        """Test fuzzy matching for typos"""
        assert indian_companies_db.resolve_symbol("tata consultency") == "TCS"  # typo
        assert indian_companies_db.resolve_symbol("infosis") == "INFY"  # typo
    
    def test_sector_matching(self):
        """Test sector-based matching"""
        it_companies = indian_companies_db._match_sector("it companies")
        assert "TCS" in it_companies
        assert "INFY" in it_companies
        
        banking_companies = indian_companies_db._match_sector("banking stocks")
        assert "HDFCBANK" in banking_companies
        assert "ICICIBANK" in banking_companies
    
    def test_company_info_retrieval(self):
        """Test company information retrieval"""
        tcs_info = indian_companies_db.get_company_info("TCS")
        assert tcs_info is not None
        assert tcs_info.name == "Tata Consultancy Services"
        assert tcs_info.sector == "Information Technology"
        assert tcs_info.symbol == "TCS"
    
    def test_search_companies(self):
        """Test company search with ranking"""
        results = indian_companies_db.search_companies("tata")
        assert len(results) > 0
        
        # Check that TCS is in results (Tata group company)
        symbols = [result[0] for result in results]
        assert "TCS" in symbols or "TATASTEEL" in symbols or "TATAMOTORS" in symbols

class TestAdvancedCrawling:
    """Test advanced crawling engine"""
    
    @pytest.mark.asyncio
    async def test_crawl_task_creation(self):
        """Test crawl task creation and queuing"""
        async with AdvancedCrawlingEngine(max_concurrent=2) as engine:
            # Add test tasks
            engine.add_crawl_task(
                url="https://httpbin.org/json",
                symbol="TEST",
                source_type="test",
                priority=1
            )
            
            assert len(engine.task_queue) == 1
            task = engine.task_queue[0]
            assert task.symbol == "TEST"
            assert task.priority == 1
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting functionality"""
        async with AdvancedCrawlingEngine(max_concurrent=2) as engine:
            start_time = asyncio.get_event_loop().time()
            
            # Test rate limiting for same domain
            await engine.rate_limiter.acquire("https://example.com/page1")
            await engine.rate_limiter.acquire("https://example.com/page2")
            
            end_time = asyncio.get_event_loop().time()
            
            # Should have some delay due to rate limiting
            assert end_time - start_time >= 0.5  # At least 500ms delay
    
    @pytest.mark.asyncio
    async def test_user_agent_rotation(self):
        """Test user agent rotation"""
        async with AdvancedCrawlingEngine(max_concurrent=2) as engine:
            headers1 = engine.user_agent_rotator.get_headers("https://example.com")
            headers2 = engine.user_agent_rotator.get_headers("https://example.com")
            
            # User agents should be different (rotation)
            assert headers1["User-Agent"] != headers2["User-Agent"]
    
    @pytest.mark.asyncio
    async def test_content_deduplication(self):
        """Test content deduplication"""
        async with AdvancedCrawlingEngine(max_concurrent=2) as engine:
            content1 = "This is test content for deduplication testing."
            content2 = "This is test content for deduplication testing."  # Exact duplicate
            content3 = "This is different content for testing."
            
            # First content should not be duplicate
            assert not engine.content_deduplicator.is_duplicate(content1, "url1")
            
            # Second content should be duplicate
            assert engine.content_deduplicator.is_duplicate(content2, "url2")
            
            # Third content should not be duplicate
            assert not engine.content_deduplicator.is_duplicate(content3, "url3")
    
    @pytest.mark.asyncio
    async def test_quality_scoring(self):
        """Test content quality scoring"""
        async with AdvancedCrawlingEngine(max_concurrent=2) as engine:
            # High quality financial content
            good_content = "TCS reported revenue of ₹50,000 crore with 15% growth in Q4 earnings."
            good_score = engine.quality_scorer.score_content(good_content, "url", "news")
            
            # Low quality spam content
            spam_content = "Click here to subscribe now! Advertisement sponsored content."
            spam_score = engine.quality_scorer.score_content(spam_content, "url", "news")
            
            assert good_score > spam_score
            assert good_score > 0.6  # Should be reasonably high
            assert spam_score < 0.4  # Should be low due to spam indicators

class TestSemanticSearch:
    """Test semantic search functionality"""
    
    def test_query_processing(self):
        """Test natural language query processing"""
        # Test price intent
        query1 = semantic_search_engine.query_processor.process_query("TCS share price today")
        assert query1.intent == "price"
        assert "TCS" in query1.entities or "tcs" in query1.entities
        assert query1.time_filter == "today"
        
        # Test news intent
        query2 = semantic_search_engine.query_processor.process_query("latest news about Infosys")
        assert query2.intent == "news"
        assert any("infosys" in entity.lower() for entity in query2.entities)
        
        # Test comparison intent
        query3 = semantic_search_engine.query_processor.process_query("TCS vs Infosys performance")
        assert query3.intent == "comparison"
    
    def test_entity_extraction(self):
        """Test entity extraction from queries"""
        processor = semantic_search_engine.query_processor
        
        entities1 = processor._extract_entities("TCS and Infosys quarterly results")
        assert any("TCS" in entity for entity in entities1)
        
        entities2 = processor._extract_entities("banking sector performance")
        # Should extract some entities or keywords
        assert isinstance(entities2, list)
    
    def test_time_filter_extraction(self):
        """Test time filter extraction"""
        processor = semantic_search_engine.query_processor
        
        assert processor._extract_time_filter("TCS price today") == "today"
        assert processor._extract_time_filter("last week performance") == "this_week"
        assert processor._extract_time_filter("quarterly results") == "this_quarter"
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis"""
        processor = semantic_search_engine.query_processor
        
        assert processor._analyze_sentiment("good performance bullish trend") == "positive"
        assert processor._analyze_sentiment("poor results bearish decline") == "negative"
        assert processor._analyze_sentiment("TCS price information") == "neutral"
    
    @pytest.mark.asyncio
    async def test_search_ranking(self):
        """Test search result ranking"""
        # Create mock search results
        results = [
            SearchResult(
                content="TCS reported strong quarterly earnings with 20% growth",
                title="TCS Q4 Results",
                url="https://example.com/tcs-results",
                symbol="TCS",
                source="news",
                timestamp=datetime.now(),
                relevance_score=0.0,
                quality_score=0.8,
                sentiment_score=0.7,
                entities=["TCS"],
                summary="TCS earnings report"
            ),
            SearchResult(
                content="General market update with various stocks",
                title="Market Update",
                url="https://example.com/market",
                symbol="MARKET",
                source="news",
                timestamp=datetime.now(),
                relevance_score=0.0,
                quality_score=0.6,
                sentiment_score=0.5,
                entities=[],
                summary="Market summary"
            )
        ]
        
        # Test search and ranking
        processed_query, ranked_results = await semantic_search_engine.search("TCS earnings", results)
        
        assert len(ranked_results) == 2
        # First result should be more relevant to TCS
        assert ranked_results[0].symbol == "TCS"
        assert ranked_results[0].relevance_score > ranked_results[1].relevance_score

class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_entity_resolution_endpoint(self):
        """Test entity resolution API endpoint"""
        response = client.post("/api/v2/entity-resolution", json={"query": "tata consultancy"})
        assert response.status_code == 200
        
        data = response.json()
        assert data["resolved_symbol"] == "TCS"
        assert data["confidence"] > 0.5
        assert data["company_info"] is not None
        assert data["company_info"]["name"] == "Tata Consultancy Services"
    
    def test_entity_resolution_with_alternatives(self):
        """Test entity resolution with alternatives"""
        response = client.post("/api/v2/entity-resolution", json={"query": "tata"})
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["alternatives"]) > 0
        # Should include multiple Tata group companies
        symbols = [alt["symbol"] for alt in data["alternatives"]]
        tata_companies = ["TCS", "TATASTEEL", "TATAMOTORS"]
        assert any(symbol in tata_companies for symbol in symbols)
    
    @patch('asi.enhanced_finance_engine_api.crawling_engine')
    def test_crawl_endpoint(self, mock_engine):
        """Test enhanced crawl endpoint"""
        mock_engine.add_crawl_task = MagicMock()
        
        response = client.post("/api/v2/crawl", json={
            "query": "TCS share price",
            "priority": 3,
            "source_types": ["price", "news"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "crawl_initiated"
        assert data["resolved_symbol"] == "TCS"
        assert data["tasks_created"] > 0
    
    @patch('asi.enhanced_finance_engine_api.legacy_engine')
    def test_search_endpoint(self, mock_engine):
        """Test semantic search endpoint"""
        # Mock legacy engine response
        mock_engine.search.return_value = [
            {
                "symbol": "TCS",
                "title": "TCS Results",
                "content": "TCS quarterly results",
                "url": "https://example.com",
                "source": "news",
                "timestamp": datetime.now().isoformat(),
                "summary": "TCS earnings"
            }
        ]
        
        response = client.post("/api/v2/search", json={
            "query": "TCS earnings report",
            "limit": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "processed_query" in data
        assert "results" in data
        assert data["total_results"] >= 0
        assert "search_time_ms" in data
    
    def test_status_endpoint(self):
        """Test system status endpoint"""
        response = client.get("/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "operational"
        assert data["version"] == "2.0.0"
        assert "components" in data
        assert "timestamp" in data
    
    def test_sources_endpoint(self):
        """Test data sources endpoint"""
        response = client.get("/sources")
        assert response.status_code == 200
        
        data = response.json()
        assert "data_sources" in data
        assert "source_types" in data
        assert "total_companies" in data
        assert "supported_sectors" in data
        
        # Check that we have major Indian exchanges
        sources = data["data_sources"]
        assert any("NSE" in source for source in sources)
        assert any("BSE" in source for source in sources)
    
    def test_legacy_search_compatibility(self):
        """Test backward compatibility with legacy search"""
        response = client.get("/search?q=TCS&mode=full_text")
        assert response.status_code == 200
        # Should return some form of results (list or dict)
        data = response.json()
        assert isinstance(data, (list, dict))

class TestIntegration:
    """Integration tests for complete system"""
    
    def test_end_to_end_query_flow(self):
        """Test complete query flow from entity resolution to search"""
        # Step 1: Entity resolution
        entity_response = client.post("/api/v2/entity-resolution", json={
            "query": "tata consultancy share price"
        })
        assert entity_response.status_code == 200
        
        resolved_symbol = entity_response.json()["resolved_symbol"]
        assert resolved_symbol == "TCS"
        
        # Step 2: Search with resolved entity
        search_response = client.post("/api/v2/search", json={
            "query": f"{resolved_symbol} price analysis",
            "limit": 5
        })
        assert search_response.status_code == 200
        
        search_data = search_response.json()
        assert search_data["processed_query"]["entities"] is not None
    
    @patch('asi.enhanced_finance_engine_api.crawling_engine')
    def test_crawl_to_search_flow(self, mock_engine):
        """Test flow from crawling to search results"""
        mock_engine.add_crawl_task = MagicMock()
        
        # Step 1: Initiate crawl
        crawl_response = client.post("/api/v2/crawl", json={
            "query": "Infosys",
            "source_types": ["news"]
        })
        assert crawl_response.status_code == 200
        
        resolved_symbol = crawl_response.json()["resolved_symbol"]
        assert resolved_symbol == "INFY"
        
        # Step 2: Search should work with the resolved symbol
        search_response = client.post("/api/v2/search", json={
            "query": resolved_symbol,
            "limit": 10
        })
        assert search_response.status_code == 200

class TestPerformance:
    """Performance and load tests"""
    
    def test_entity_resolution_performance(self):
        """Test entity resolution performance"""
        import time
        
        queries = ["TCS", "Infosys", "HDFC Bank", "Reliance Industries", "tata consultancy"]
        
        start_time = time.time()
        for query in queries:
            indian_companies_db.resolve_symbol(query)
        end_time = time.time()
        
        # Should resolve all queries quickly
        avg_time = (end_time - start_time) / len(queries)
        assert avg_time < 0.1  # Less than 100ms per query
    
    def test_concurrent_api_requests(self):
        """Test concurrent API requests"""
        import concurrent.futures
        import time
        
        def make_request():
            response = client.post("/api/v2/entity-resolution", json={"query": "TCS"})
            return response.status_code == 200
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]
        end_time = time.time()
        
        # All requests should succeed
        assert all(results)
        # Should handle concurrent requests efficiently
        assert end_time - start_time < 5.0  # Less than 5 seconds for 10 requests

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
