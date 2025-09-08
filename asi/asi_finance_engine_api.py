"""
REST API wrapper for Institutional-Grade Finance Engine
- /search: query by text/entity/symbol
- /crawl: trigger new crawl/index
- /status: engine status
- /sources: list available sources
"""
import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from asi.asi_finance_engine_fullstack import (
    FinanceEngineFullstack, NSESource, MoneyControlSource, YahooFinanceSource, EconomicTimesNewsSource, BSEFilingsSource,
    ReutersNewsSource, YahooFinanceNewsSource, SEBIFilingsSource, BloombergNewsSource, CNBCNewsSource, RBICircularsSource, SEBICircularsSource, MintNewsSource
)

app = FastAPI()

# Instantiate engine with all sources
sources = [
    NSESource(), MoneyControlSource(), YahooFinanceSource(), EconomicTimesNewsSource(), BSEFilingsSource(),
    ReutersNewsSource(), YahooFinanceNewsSource(), SEBIFilingsSource(),
    BloombergNewsSource(), CNBCNewsSource(), RBICircularsSource(), SEBICircularsSource(), MintNewsSource()
]
engine = FinanceEngineFullstack(sources)

class SearchResult(BaseModel):
    symbol: Optional[str]
    price: Optional[str]
    change: Optional[str]
    change_percent: Optional[str]
    volume: Optional[str]
    timestamp: Optional[str]
    source: Optional[str]
    title: Optional[str]
    summary: Optional[str]
    url: Optional[str]
    headline: Optional[str]
    date: Optional[str]
    content: Optional[str]

@app.get("/search", response_model=List[SearchResult])
def search(q: str = Query(..., description="Search query (symbol, text, entity)"), mode: str = Query("full_text", enum=["full_text","entity","symbol"])):
    """Search indexed data (full text, entity, or symbol)"""
    results = engine.search(q, mode=mode)
    return results

from pydantic import BaseModel

class CrawlRequest(BaseModel):
    symbol: str

@app.post("/crawl")
async def crawl(req: CrawlRequest):
    """Trigger new crawl/index for a symbol or keyword"""
    await engine.crawl_and_index(req.symbol)
    return {"status": "crawled", "symbol": req.symbol}

@app.get("/status")
def status():
    """Get engine status (doc count, sources)"""
    return {"indexed_docs": len(engine.index.docs), "sources": [s.name for s in sources]}

@app.get("/sources")
def get_sources():
    """List available data sources"""
    return {"sources": [s.name for s in sources]}

# --- Analytics/LLM Endpoints ---
from asi.asi_finance_engine_fullstack import summarize_text, analyze_sentiment, answer_question, detect_trend

class TextRequest(BaseModel):
    text: str

class QARequest(BaseModel):
    question: str
    context: str

class TrendRequest(BaseModel):
    field: Optional[str] = "volume"
    query: Optional[str] = None
    mode: Optional[str] = "full_text"

@app.post("/summarize")
def summarize(req: TextRequest):
    """Summarize a given text using transformer model"""
    return {"summary": summarize_text(req.text)}

@app.post("/sentiment")
def sentiment(req: TextRequest):
    """Analyze sentiment of given text"""
    return analyze_sentiment(req.text)

@app.post("/ask")
def ask(req: QARequest):
    """Answer a question over provided context using transformer QA"""
    return {"answer": answer_question(req.question, req.context)}

@app.post("/trend")
def trend(req: TrendRequest):
    """Detect trend/anomaly for a query or field (e.g., volume spike)"""
    docs = []
    if req.query:
        docs = engine.search(req.query, mode=req.mode)
    return detect_trend(docs, field=req.field)

if __name__ == "__main__":
    uvicorn.run("asi_finance_engine_api:app", host="0.0.0.0", port=8081, reload=True)
