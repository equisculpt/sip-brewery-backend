"""
Unit test for symbol resolution logic (Google-like search)
Tests entity resolution without making actual network calls
"""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from asi.asi_finance_engine_api import app

client = TestClient(app)

def resolve_symbol(query: str) -> str:
    """
    Simple symbol resolution function for testing
    In production, this would use a database or API
    """
    # Normalize query
    query = query.lower().strip()
    
    # Remove common suffixes
    query = query.replace(" share price", "").replace(" stock", "").replace(" price", "")
    
    # Symbol mapping dictionary (expand this in production)
    symbol_map = {
        "tcs": "TCS",
        "tata consultancy": "TCS", 
        "tata consultancy services": "TCS",
        "infosys": "INFY",
        "infy": "INFY",
        "reliance": "RELIANCE",
        "reliance industries": "RELIANCE",
        "hdfc": "HDFCBANK",
        "hdfc bank": "HDFCBANK",
        "icici": "ICICIBANK",
        "icici bank": "ICICIBANK"
    }
    
    # Try exact match first
    if query in symbol_map:
        return symbol_map[query]
    
    # Try fuzzy matching (simple version)
    for key, symbol in symbol_map.items():
        if query in key or key in query:
            return symbol
    
    # If no match found, return uppercase version of query
    return query.upper()

def test_symbol_resolution_logic():
    """Test the symbol resolution function directly"""
    # Test exact matches
    assert resolve_symbol("TCS") == "TCS"
    assert resolve_symbol("tcs") == "TCS"
    
    # Test company name variations
    assert resolve_symbol("Tata Consultancy") == "TCS"
    assert resolve_symbol("Tata Consultancy Services") == "TCS"
    assert resolve_symbol("tata consultancy") == "TCS"
    
    # Test with price/stock suffixes
    assert resolve_symbol("tcs share price") == "TCS"
    assert resolve_symbol("TCS stock") == "TCS"
    assert resolve_symbol("Tata Consultancy price") == "TCS"
    
    # Test other companies
    assert resolve_symbol("infosys") == "INFY"
    assert resolve_symbol("Infosys") == "INFY"
    assert resolve_symbol("reliance") == "RELIANCE"
    assert resolve_symbol("Reliance Industries") == "RELIANCE"

@patch('asi.asi_finance_engine_api.engine.crawl_and_index')
def test_crawl_endpoint_with_mocked_engine(mock_crawl):
    """Test /crawl endpoint with mocked engine to avoid network calls"""
    # Mock the async crawl_and_index method
    mock_crawl.return_value = AsyncMock()
    
    # Test various queries
    test_queries = [
        "TCS",
        "tcs", 
        "Tata Consultancy",
        "tcs share price"
    ]
    
    for query in test_queries:
        response = client.post("/crawl", json={"symbol": query})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "crawled"
        assert "symbol" in data
        # The symbol should be the original query (until we implement resolution)
        assert data["symbol"] == query

def test_crawl_endpoint_validation():
    """Test /crawl endpoint input validation"""
    # Test missing symbol
    response = client.post("/crawl", json={})
    assert response.status_code == 422  # Validation error
    
    # Test empty symbol
    response = client.post("/crawl", json={"symbol": ""})
    assert response.status_code == 200  # Should work but might return error in response
    
    # Test valid symbol
    with patch('asi.asi_finance_engine_api.engine.crawl_and_index') as mock_crawl:
        mock_crawl.return_value = AsyncMock()
        response = client.post("/crawl", json={"symbol": "TCS"})
        assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
