import asyncio
from typing import List, Dict, Any
from datetime import datetime

# --- 1. Data Source Plugins ---

class DataSource:
    name: str
    async def fetch(self, query: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

from real_market_data_asi import RealMarketDataASI

class NSESource(DataSource):
    name = "NSE"
    async def fetch(self, query: str) -> List[Dict[str, Any]]:
        asi = RealMarketDataASI()
        await asi.initialize()
        try:
            data = await asi.get_real_time_data(query)
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
            if data and isinstance(data, dict):
                return [data]
            return []
        finally:
            await asi.close()

class MoneyControlSource(DataSource):
    name = "MoneyControl"
    async def fetch(self, query: str) -> List[Dict[str, Any]]:
        # TODO: Integrate your MoneyControl crawler logic here
        print(f"[MoneyControl] Fetching data for {query}")
        return []

# Add more sources as needed...

# --- 2. Data Normalization ---

def normalize(data: Dict[str, Any], source: str) -> Dict[str, Any]:
    # Convert all fields to a unified schema
    return {
        "symbol": data.get("symbol"),
        "price": data.get("price"),
        "timestamp": data.get("timestamp", datetime.now()),
        "source": source,
        # ... add more normalized fields
    }

# --- 3. Indexing (In-Memory for Private Use) ---

class FinanceIndex:
    def __init__(self):
        self.data = []  # Replace with actual DB or search index for scale

    def add(self, record: Dict[str, Any]):
        self.data.append(record)

    def search(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Simple filter, replace with Elasticsearch/Weaviate for scale
        return [r for r in self.data if r["symbol"] == query.get("symbol")]

# --- 4. Search Engine Orchestrator ---

class FinanceSearchEngine:
    def __init__(self, sources: List[DataSource]):
        self.sources = sources
        self.index = FinanceIndex()

    async def crawl_and_index(self, query: str):
        results = []
        for src in self.sources:
            try:
                data = await src.fetch(query)
                for d in data:
                    norm = normalize(d, src.name)
                    self.index.add(norm)
                    results.append(norm)
            except Exception as e:
                print(f"[{src.name}] Error: {e}")
        return results

    def search(self, symbol: str) -> List[Dict[str, Any]]:
        return self.index.search({"symbol": symbol})

# --- 5. Example Usage ---

async def main():
    sources = [NSESource(), MoneyControlSource()]
    engine = FinanceSearchEngine(sources)
    await engine.crawl_and_index("RELIANCE")
    await engine.crawl_and_index("TCS")
    print(engine.search("RELIANCE"))

if __name__ == "__main__":
    asyncio.run(main())
