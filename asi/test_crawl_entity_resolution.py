import pytest
from fastapi.testclient import TestClient
from asi.asi_finance_engine_api import app

client = TestClient(app)

def test_crawl_tcs_variants():
    # All these should resolve to the same symbol (e.g., 'TCS')
    queries = [
        "TCS",
        "tcs",
        "Tata Consultancy",
        "Tata Consultancy Services",
        "tcs share price",
        "tata consultancy share price"
    ]
    expected_symbol = "TCS"  # This should match your symbol resolver
    for q in queries:
        response = client.post("/crawl", json={"symbol": q})
        assert response.status_code == 200, f"Failed for query: {q}"
        data = response.json()
        assert data["symbol"].upper() == expected_symbol, f"Resolver failed for query: {q}"
        assert data["status"] == "crawled"

if __name__ == "__main__":
    pytest.main([__file__])
