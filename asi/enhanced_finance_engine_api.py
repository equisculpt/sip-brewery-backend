"""
Enhanced ASI Finance Engine API
Integrating all advanced components: Entity Resolution, Advanced Crawling, Semantic Search, ASI Integration
"""
import asyncio
import logging
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

# Import all our advanced components
from indian_companies_database import indian_companies_db
from enhanced_indian_companies_database import enhanced_indian_companies_db, ExchangeType, MarketCapCategory
from sme_emerge_data_sources import SMEEmergeAggregator
from ztech_ohlc_live_service import ZTechAPIService
from dual_ztech_data_service import DualZTechAPIService, ZTechCompanyType
from ztech_realtime_search_engine import ZTechRealtimeSearchEngine
from universal_financial_search_engine import UniversalFinancialSearchEngine, initialize_search_engine
from advanced_crawling_engine import AdvancedCrawlingEngine
from semantic_search_layer import SemanticSearchLayer
from asi_integration_layer import ASIIntegrationLayer, MarketInsight, PredictiveSignal
from asi.asi_finance_engine_fullstack import FinanceEngineFullstack

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="ASI Finance Search Engine",
    description="Institutional-Grade Indian Finance Search Engine for ASI",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CrawlRequest(BaseModel):
    query: str
    priority: Optional[int] = 5
    source_types: Optional[List[str]] = ["all"]

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 20
    filters: Optional[Dict[str, Any]] = {}

class EntityResolutionRequest(BaseModel):
    query: str

class InsightRequest(BaseModel):
    symbol: str
    insight_types: Optional[List[str]] = ["all"]

class PredictionRequest(BaseModel):
    symbol: str
    time_horizon: Optional[str] = "short"

# Response models
class EntityResolutionResponse(BaseModel):
    original_query: str
    resolved_symbol: Optional[str]
    confidence: float
    company_info: Optional[Dict[str, Any]]
    alternatives: List[Dict[str, Any]]

class CrawlResponse(BaseModel):
    status: str
    query: str
    resolved_symbol: Optional[str]
    tasks_created: int
    estimated_completion: str

class SearchResponse(BaseModel):
    query: str
    processed_query: Dict[str, Any]
    results: List[Dict[str, Any]]
    total_results: int
    search_time_ms: float

class InsightResponse(BaseModel):
    symbol: str
    insights: List[Dict[str, Any]]
    total_insights: int

class PredictionResponse(BaseModel):
    symbol: str
    prediction: Optional[Dict[str, Any]]
    confidence: float
    generated_at: str

# Global instances
crawling_engine: Optional[AdvancedCrawlingEngine] = None
legacy_engine: Optional[FinanceEngineFullstack] = None

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global crawling_engine, legacy_engine
    
    logger.info("Initializing ASI Finance Search Engine...")
    
    try:
        # Initialize ASI integration engine
        await asi_integration_engine.initialize()
        logger.info("ASI Integration Engine initialized")
        
        # Initialize advanced crawling engine
        crawling_engine = AdvancedCrawlingEngine(max_concurrent=15)
        await crawling_engine.__aenter__()
        logger.info("Advanced Crawling Engine initialized")
        
        # Initialize legacy engine for backward compatibility
        from asi.asi_finance_engine_fullstack import (
            NSESource, MoneyControlSource, YahooFinanceSource, EconomicTimesNewsSource,
            BSEFilingsSource, ReutersNewsSource, YahooFinanceNewsSource, SEBIFilingsSource,
            BloombergNewsSource, CNBCNewsSource, RBICircularsSource, SEBICircularsSource, MintNewsSource
        )
        
        sources = [
            NSESource(), MoneyControlSource(), YahooFinanceSource(), EconomicTimesNewsSource(),
            BSEFilingsSource(), ReutersNewsSource(), YahooFinanceNewsSource(), SEBIFilingsSource(),
            BloombergNewsSource(), CNBCNewsSource(), RBICircularsSource(), SEBICircularsSource(), MintNewsSource()
        ]
        legacy_engine = FinanceEngineFullstack(sources)
        logger.info("Legacy engine initialized for backward compatibility")
        
        logger.info("🚀 ASI Finance Search Engine fully initialized!")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global crawling_engine
    
    logger.info("Shutting down ASI Finance Search Engine...")
    
    if crawling_engine:
        await crawling_engine.__aexit__(None, None, None)
        
    await asi_integration_engine.data_streamer.stop_streaming()
    
    logger.info("ASI Finance Search Engine shut down complete")

# === ENHANCED API ENDPOINTS ===

@app.post("/api/v2/entity-resolution", response_model=EntityResolutionResponse)
async def resolve_entity(request: EntityResolutionRequest):
    """
    Resolve natural language query to company symbol with confidence scoring
    Examples: "tata consultancy", "tcs share price", "infosys"
    """
    try:
        # Resolve using our advanced database
        resolved_symbol = indian_companies_db.resolve_symbol(request.query)
        
        # Get detailed search results with confidence scores
        search_results = indian_companies_db.search_companies(request.query)
        
        # Prepare response
        company_info = None
        confidence = 0.0
        alternatives = []
        
        if resolved_symbol:
            company_info_obj = indian_companies_db.get_company_info(resolved_symbol)
            if company_info_obj:
                company_info = {
                    "symbol": company_info_obj.symbol,
                    "name": company_info_obj.name,
                    "sector": company_info_obj.sector,
                    "industry": company_info_obj.industry,
                    "market_cap_category": company_info_obj.market_cap_category,
                    "exchange": company_info_obj.exchange
                }
                confidence = search_results[0][2] if search_results else 0.8
        
        # Prepare alternatives
        for symbol, company, score in search_results[:5]:
            if symbol != resolved_symbol:
                alternatives.append({
                    "symbol": symbol,
                    "name": company.name,
                    "sector": company.sector,
                    "confidence": score
                })
        
        return EntityResolutionResponse(
            original_query=request.query,
            resolved_symbol=resolved_symbol,
            confidence=confidence,
            company_info=company_info,
            alternatives=alternatives
        )
        
    except Exception as e:
        logger.error(f"Entity resolution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/crawl", response_model=CrawlResponse)
async def enhanced_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    """
    Enhanced crawling with entity resolution and intelligent task scheduling
    """
    try:
        # First, resolve the entity
        resolved_symbol = indian_companies_db.resolve_symbol(request.query)
        
        if not resolved_symbol:
            # If we can't resolve, try partial matching
            search_results = indian_companies_db.search_companies(request.query)
            if search_results:
                resolved_symbol = search_results[0][0]  # Take best match
            else:
                resolved_symbol = request.query.upper()  # Fallback to original query
        
        # Create crawl tasks for different source types
        tasks_created = 0
        
        if "all" in request.source_types or "price" in request.source_types:
            # NSE price data
            crawling_engine.add_crawl_task(
                url=f"https://www.nseindia.com/api/quote-equity?symbol={resolved_symbol}",
                symbol=resolved_symbol,
                source_type="price",
                priority=request.priority,
                metadata={"original_query": request.query}
            )
            tasks_created += 1
            
            # BSE price data
            crawling_engine.add_crawl_task(
                url=f"https://api.bseindia.com/BseIndiaAPI/api/StockReachGraph/w?scripcode={resolved_symbol}&flag=0",
                symbol=resolved_symbol,
                source_type="price",
                priority=request.priority
            )
            tasks_created += 1
        
        if "all" in request.source_types or "news" in request.source_types:
            # News sources
            news_sources = [
                f"https://www.moneycontrol.com/news/tags/{resolved_symbol.lower()}.html",
                f"https://economictimes.indiatimes.com/topic/{resolved_symbol}",
                f"https://www.reuters.com/companies/{resolved_symbol}.NS/news"
            ]
            
            for url in news_sources:
                crawling_engine.add_crawl_task(
                    url=url,
                    symbol=resolved_symbol,
                    source_type="news",
                    priority=request.priority + 1  # Lower priority for news
                )
                tasks_created += 1
        
        if "all" in request.source_types or "regulatory" in request.source_types:
            # Regulatory filings
            crawling_engine.add_crawl_task(
                url=f"https://www.bseindia.com/corporates/ann.aspx?scrip={resolved_symbol}",
                symbol=resolved_symbol,
                source_type="regulatory",
                priority=request.priority + 2
            )
            tasks_created += 1
        
        # Execute crawling in background
        background_tasks.add_task(execute_crawl_tasks, resolved_symbol)
        
        return CrawlResponse(
            status="crawl_initiated",
            query=request.query,
            resolved_symbol=resolved_symbol,
            tasks_created=tasks_created,
            estimated_completion="2-5 minutes"
        )
        
    except Exception as e:
        logger.error(f"Enhanced crawl error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def execute_crawl_tasks(symbol: str):
    """Execute crawl tasks in background"""
    try:
        results = await crawling_engine.crawl_all()
        logger.info(f"Crawling completed for {symbol}: {len(results.get(symbol, []))} results")
        
        # Process results and update search index
        if symbol in results:
            await process_crawl_results(symbol, results[symbol])
            
    except Exception as e:
        logger.error(f"Background crawl execution error: {e}")

async def process_crawl_results(symbol: str, results: List[CrawlResult]):
    """Process crawl results and update search index"""
    try:
        # Convert crawl results to search results
        search_results = []
        
        for result in results:
            search_result = SearchResult(
                content=result.content,
                title=f"{symbol} - {result.source_type}",
                url=result.url,
                symbol=symbol,
                source=result.source_type,
                timestamp=result.timestamp,
                relevance_score=0.8,
                quality_score=result.metadata.get('quality_score', 0.5),
                sentiment_score=0.5,  # Would be calculated by sentiment analyzer
                entities=[symbol],
                summary=result.content[:200] + "..." if len(result.content) > 200 else result.content
            )
            search_results.append(search_result)
        
        # Here you would update your search index
        # For now, we'll just log the results
        logger.info(f"Processed {len(search_results)} results for {symbol}")
        
    except Exception as e:
        logger.error(f"Error processing crawl results: {e}")

@app.post("/api/v2/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """
    Advanced semantic search with NLP processing and context-aware ranking
    """
    start_time = datetime.now()
    
    try:
        # First, try to get results from legacy engine for backward compatibility
        legacy_results = []
        if legacy_engine:
            try:
                legacy_data = legacy_engine.search(request.query, mode="full_text")
                # Convert legacy results to SearchResult format
                for item in legacy_data:
                    search_result = SearchResult(
                        content=item.get('content', ''),
                        title=item.get('title', ''),
                        url=item.get('url', ''),
                        symbol=item.get('symbol', ''),
                        source=item.get('source', ''),
                        timestamp=datetime.fromisoformat(item.get('timestamp', datetime.now().isoformat())),
                        relevance_score=0.5,
                        quality_score=0.5,
                        sentiment_score=0.5,
                        entities=[],
                        summary=item.get('summary', '')
                    )
                    legacy_results.append(search_result)
            except Exception as e:
                logger.warning(f"Legacy search failed: {e}")
        
        # Process query with semantic search engine
        processed_query, ranked_results = await semantic_search_engine.search(request.query, legacy_results)
        
        # Apply limit
        limited_results = ranked_results[:request.limit]
        
        # Convert to response format
        result_dicts = []
        for result in limited_results:
            result_dict = {
                "title": result.title,
                "content": result.content,
                "url": result.url,
                "symbol": result.symbol,
                "source": result.source,
                "timestamp": result.timestamp.isoformat(),
                "relevance_score": result.relevance_score,
                "quality_score": result.quality_score,
                "sentiment_score": result.sentiment_score,
                "entities": result.entities,
                "summary": result.summary
            }
            result_dicts.append(result_dict)
        
        # Calculate search time
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SearchResponse(
            query=request.query,
            processed_query={
                "intent": processed_query.intent,
                "entities": processed_query.entities,
                "time_filter": processed_query.time_filter,
                "sector_filter": processed_query.sector_filter,
                "sentiment": processed_query.sentiment,
                "confidence": processed_query.confidence
            },
            results=result_dicts,
            total_results=len(ranked_results),
            search_time_ms=search_time
        )
        
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/insights/{symbol}", response_model=InsightResponse)
async def get_insights(symbol: str, insight_types: Optional[str] = Query("all")):
    """
    Get real-time insights for a symbol from ASI analysis
    """
    try:
        # Resolve symbol first
        resolved_symbol = indian_companies_db.resolve_symbol(symbol)
        if not resolved_symbol:
            resolved_symbol = symbol.upper()
        
        # Get insights from ASI integration engine
        insights = await asi_integration_engine.get_insights_for_symbol(resolved_symbol)
        
        # Filter by insight types if specified
        if insight_types != "all":
            requested_types = insight_types.split(",")
            insights = [i for i in insights if i.insight_type.value in requested_types]
        
        # Convert to response format
        insight_dicts = []
        for insight in insights:
            insight_dict = {
                "insight_id": insight.insight_id,
                "insight_type": insight.insight_type.value,
                "priority": insight.priority.value,
                "confidence": insight.confidence,
                "title": insight.title,
                "description": insight.description,
                "data": insight.data,
                "timestamp": insight.timestamp.isoformat(),
                "actions": insight.actions
            }
            insight_dicts.append(insight_dict)
        
        return InsightResponse(
            symbol=resolved_symbol,
            insights=insight_dicts,
            total_insights=len(insight_dicts)
        )
        
    except Exception as e:
        logger.error(f"Insights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/prediction/{symbol}", response_model=PredictionResponse)
async def get_prediction(symbol: str, time_horizon: Optional[str] = Query("short")):
    """
    Get AI-powered price prediction for a symbol
    """
    try:
        # Resolve symbol first
        resolved_symbol = indian_companies_db.resolve_symbol(symbol)
        if not resolved_symbol:
            resolved_symbol = symbol.upper()
        
        # Get prediction from ASI integration engine
        prediction = await asi_integration_engine.get_prediction_for_symbol(resolved_symbol)
        
        prediction_dict = None
        confidence = 0.0
        
        if prediction:
            prediction_dict = {
                "signal_type": prediction.signal_type,
                "confidence": prediction.confidence,
                "target_price": prediction.target_price,
                "time_horizon": prediction.time_horizon,
                "reasoning": prediction.reasoning,
                "risk_factors": prediction.risk_factors,
                "supporting_data": prediction.supporting_data,
                "timestamp": prediction.timestamp.isoformat()
            }
            confidence = prediction.confidence
        
        return PredictionResponse(
            symbol=resolved_symbol,
            prediction=prediction_dict,
            confidence=confidence,
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === LEGACY ENDPOINTS FOR BACKWARD COMPATIBILITY ===

@app.get("/search")
async def legacy_search(q: str = Query(...), mode: str = Query("full_text")):
    """Legacy search endpoint for backward compatibility"""
    try:
        if legacy_engine:
            results = legacy_engine.search(q, mode=mode)
            return results
        else:
            # Fallback to new search
            request = SearchRequest(query=q, limit=20)
            response = await semantic_search(request)
            return response.results
    except Exception as e:
        logger.error(f"Legacy search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crawl")
async def legacy_crawl(req: CrawlRequest):
    """Legacy crawl endpoint for backward compatibility"""
    try:
        if legacy_engine:
            await legacy_engine.crawl_and_index(req.query)
            return {"status": "crawled", "symbol": req.query}
        else:
            # Use new enhanced crawl
            background_tasks = BackgroundTasks()
            response = await enhanced_crawl(req, background_tasks)
            return {"status": "crawled", "symbol": response.resolved_symbol}
    except Exception as e:
        logger.error(f"Legacy crawl error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get system status"""
    try:
        stats = {}
        
        if crawling_engine:
            stats.update(crawling_engine.get_statistics())
        
        return {
            "status": "operational",
            "version": "2.0.0",
            "components": {
                "entity_resolution": "active",
                "advanced_crawling": "active" if crawling_engine else "inactive",
                "semantic_search": "active",
                "asi_integration": "active",
                "legacy_engine": "active" if legacy_engine else "inactive"
            },
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching exchange statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ZTECH Specific Endpoints
@app.get("/api/v2/ztech/live-price")
async def get_ztech_live_price_endpoint():
    """Get ZTECH live price data"""
    try:
        data = await get_ztech_live_price()
        return data
    except Exception as e:
        logger.error(f"Error fetching ZTECH live price: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/ztech/ohlc")
async def get_ztech_ohlc_endpoint(timeframe: str = "1d", period: str = "1mo"):
    """Get ZTECH OHLC data"""
    try:
        # Validate parameters
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
        
        if timeframe not in valid_timeframes:
            raise HTTPException(status_code=400, detail=f"Invalid timeframe. Valid options: {valid_timeframes}")
        
        if period not in valid_periods:
            raise HTTPException(status_code=400, detail=f"Invalid period. Valid options: {valid_periods}")
        
        data = await get_ztech_ohlc(timeframe, period)
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching ZTECH OHLC: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/ztech/comprehensive")
async def get_ztech_comprehensive_endpoint():
    """Get comprehensive ZTECH data including live price, OHLC, and technical indicators"""
    try:
        data = await get_ztech_comprehensive()
        return data
    except Exception as e:
        logger.error(f"Error fetching comprehensive ZTECH data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/ztech/intraday")
async def get_ztech_intraday_endpoint(timeframe: str = "1m"):
    """Get ZTECH intraday data for current trading session"""
    try:
        # Validate timeframe for intraday
        valid_intraday_timeframes = ["1m", "5m", "15m", "30m", "1h"]
        
        if timeframe not in valid_intraday_timeframes:
            raise HTTPException(status_code=400, detail=f"Invalid intraday timeframe. Valid options: {valid_intraday_timeframes}")
        
        data = await get_ztech_ohlc(timeframe, "1d")
        
        # Add intraday-specific metadata
        if data["status"] == "success":
            data["data"]["session_type"] = "intraday"
            data["data"]["trading_day"] = datetime.now().strftime("%Y-%m-%d")
        
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching ZTECH intraday data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/ztech/technical-analysis")
async def get_ztech_technical_analysis():
    """Get ZTECH technical analysis and indicators"""
    try:
        comprehensive_data = await get_ztech_comprehensive()
        
        if comprehensive_data["status"] != "success":
            return comprehensive_data
        
        data = comprehensive_data["data"]
        
        # Extract technical analysis specific data
        technical_data = {
            "symbol": "ZTECH",
            "company_name": "Zentech Systems Limited",
            "current_price": data.get("live_price", {}).get("price"),
            "technical_indicators": data.get("technical_indicators", {}),
            "price_levels": {
                "day_high": data.get("live_price", {}).get("high"),
                "day_low": data.get("live_price", {}).get("low"),
                "day_open": data.get("live_price", {}).get("open"),
                "previous_close": data.get("live_price", {}).get("previous_close")
            },
            "volume_analysis": {
                "current_volume": data.get("live_price", {}).get("volume"),
                "avg_volume": data.get("technical_indicators", {}).get("avg_volume_20"),
                "volume_ratio": data.get("technical_indicators", {}).get("volume_ratio")
            },
            "trend_analysis": {
                "sma_20": data.get("technical_indicators", {}).get("sma_20"),
                "sma_50": data.get("technical_indicators", {}).get("sma_50"),
                "rsi": data.get("technical_indicators", {}).get("rsi")
            },
            "support_resistance": {
                "bollinger_upper": data.get("technical_indicators", {}).get("bollinger_upper"),
                "bollinger_lower": data.get("technical_indicators", {}).get("bollinger_lower"),
                "52_week_high": data.get("technical_indicators", {}).get("52_week_high"),
                "52_week_low": data.get("technical_indicators", {}).get("52_week_low")
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "data": technical_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching ZTECH technical analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/ztech/company-info")
async def get_ztech_company_info():
    """Get ZTECH company information"""
    try:
        # Get company info from enhanced database
        company_info = enhanced_indian_companies_db.companies.get("ZTECH")
        
        if not company_info:
            raise HTTPException(status_code=404, detail="ZTECH company information not found")
        
        return {
            "status": "success",
            "data": {
                "symbol": company_info.symbol,
                "name": company_info.name,
                "sector": company_info.sector,
                "industry": company_info.industry,
                "exchange": company_info.exchange.value,
                "market_cap_category": company_info.market_cap_category.value,
                "bse_code": company_info.bse_code,
                "isin": company_info.isin,
                "listing_date": company_info.listing_date,
                "face_value": company_info.face_value,
                "aliases": company_info.aliases,
                "keywords": company_info.keywords,
                "nse_symbol": "ZTECH.NS",
                "bse_symbol": f"{company_info.bse_code}.BO" if company_info.bse_code else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching ZTECH company info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize dual ZTECH service
dual_ztech_service = DualZTechAPIService()

# Initialize real-time search engine
realtime_search_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global finance_engine, realtime_search_engine, universal_search_engine
    finance_engine = None
    realtime_search_engine = None
    universal_search_engine = None
    realtime_search_engine = ZTechRealtimeSearchEngine()
    await realtime_search_engine.start_background_updates()
    logger.info("🚀 Real-time ZTECH search engine initialized")
    
    # Initialize universal search engine
    universal_search_engine = UniversalFinancialSearchEngine()
    await universal_search_engine.start_background_updates()
    logger.info("🌐 Universal financial search engine initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if realtime_search_engine:
        realtime_search_engine.stop_background_updates()
        logger.info("🛑 Real-time search engine stopped")

@app.get("/api/v2/ztech-dual/query")
async def query_dual_ztech(query: str):
    """
    Query both ZTECH companies with natural language
    Supports queries like:
    - "ztech india live price"
    - "zentech systems ohlc data"
    - "compare all ztech companies"
    """
    try:
        result = await dual_ztech_service.process_query(query)
        return result
    except Exception as e:
        logger.error(f"Error processing dual ZTECH query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/ztech-dual/all-companies")
async def get_all_ztech_companies():
    """
    Get comprehensive data for all ZTECH companies
    Returns data for both Z-Tech (India) Limited and Zentech Systems Limited
    """
    try:
        result = await dual_ztech_service.data_service.get_all_companies_data()
        return result
    except Exception as e:
        logger.error(f"Error fetching all ZTECH companies data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/ztech-dual/ztech-india")
async def get_ztech_india_data():
    """
    Get comprehensive data for Z-Tech (India) Limited - NSE Main Board
    """
    try:
        result = await dual_ztech_service.data_service.get_comprehensive_data(ZTechCompanyType.ZTECH_INDIA)
        return result
    except Exception as e:
        logger.error(f"Error fetching Z-Tech India data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/ztech-dual/zentech-systems")
async def get_zentech_systems_data():
    """
    Get comprehensive data for Zentech Systems Limited - NSE Emerge
    """
    try:
        result = await dual_ztech_service.data_service.get_comprehensive_data(ZTechCompanyType.ZENTECH_SYSTEMS)
        return result
    except Exception as e:
        logger.error(f"Error fetching Zentech Systems data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/ztech-dual/live-prices")
async def get_all_ztech_live_prices():
    """
    Get live prices for all ZTECH companies
    """
    try:
        ztech_india_data = await dual_ztech_service.data_service.get_live_data(ZTechCompanyType.ZTECH_INDIA)
        zentech_systems_data = await dual_ztech_service.data_service.get_live_data(ZTechCompanyType.ZENTECH_SYSTEMS)
        
        return {
            "ztech_india": ztech_india_data.__dict__ if ztech_india_data else None,
            "zentech_systems": zentech_systems_data.__dict__ if zentech_systems_data else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching ZTECH live prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/ztech-dual/comparison")
async def get_ztech_comparison():
    """
    Get detailed comparison between Z-Tech India and Zentech Systems
    """
    try:
        all_data = await dual_ztech_service.data_service.get_all_companies_data()
        return {
            "comparison": all_data.get("comparison", {}),
            "ztech_india_summary": {
                "name": "Z-Tech (India) Limited",
                "exchange": "NSE Main Board",
                "price": all_data.get("ztech_india", {}).get("live_data", {}).get("current_price"),
                "volume": all_data.get("ztech_india", {}).get("live_data", {}).get("volume")
            },
            "zentech_systems_summary": {
                "name": "Zentech Systems Limited",
                "exchange": "NSE Emerge",
                "price": all_data.get("zentech_systems", {}).get("live_data", {}).get("current_price"),
                "volume": all_data.get("zentech_systems", {}).get("live_data", {}).get("volume")
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating ZTECH comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Google-level Real-time Search Endpoints
@app.get("/api/v2/search/realtime")
async def realtime_search(q: str = Query(..., description="Search query as user types")):
    """
    Google-level real-time search for ZTECH
    Provides instant suggestions and data as user types
    
    Example: /api/v2/search/realtime?q=ztech
    """
    try:
        if not realtime_search_engine:
            raise HTTPException(status_code=503, detail="Real-time search engine not initialized")
        
        result = await realtime_search_engine.search_realtime(q)
        
        return {
            "query": result.query,
            "suggestions": [{
                "text": s.query,
                "type": s.type.value,
                "company": s.company,
                "description": s.description,
                "confidence": s.confidence,
                "instant_data": s.instant_data
            } for s in result.suggestions],
            "live_data": result.live_data,
            "response_time_ms": result.total_time_ms,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Real-time search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/search/autocomplete")
async def autocomplete_search(q: str = Query(..., description="Query for autocomplete")):
    """
    Google-style autocomplete for ZTECH queries
    Returns just the suggestion strings for dropdown
    
    Example: /api/v2/search/autocomplete?q=zt
    """
    try:
        if not realtime_search_engine:
            raise HTTPException(status_code=503, detail="Real-time search engine not initialized")
        
        suggestions = await realtime_search_engine.get_autocomplete(q, limit=8)
        
        return {
            "query": q,
            "suggestions": suggestions,
            "count": len(suggestions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Autocomplete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/search/instant")
async def instant_search_data(q: str = Query(..., description="Query for instant data")):
    """
    Instant data capture for ZTECH queries
    Returns live data immediately as user types
    
    Example: /api/v2/search/instant?q=ztech india
    """
    try:
        if not realtime_search_engine:
            raise HTTPException(status_code=503, detail="Real-time search engine not initialized")
        
        data = await realtime_search_engine.get_instant_data(q)
        
        return {
            "query": q,
            "instant_data": data,
            "has_data": bool(data and not data.get("error")),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Instant data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/search/suggestions/{query}")
async def get_search_suggestions(query: str):
    """
    Get detailed search suggestions for a specific query
    Includes confidence scores and instant data
    """
    try:
        if not realtime_search_engine:
            raise HTTPException(status_code=503, detail="Real-time search engine not initialized")
        
        result = await realtime_search_engine.search_realtime(query, max_suggestions=10)
        
        return {
            "query": query,
            "suggestions": [{
                "query": s.query,
                "type": s.type.value,
                "company": s.company,
                "description": s.description,
                "confidence": s.confidence,
                "has_instant_data": bool(s.instant_data)
            } for s in result.suggestions],
            "total_suggestions": len(result.suggestions),
            "response_time_ms": result.total_time_ms,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Search suggestions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources")
async def get_sources():
    """Get available data sources"""
    return {
        "data_sources": [
            "NSE (National Stock Exchange)",
            "BSE (Bombay Stock Exchange)", 
            "MoneyControl",
            "Economic Times",
            "Reuters",
            "Bloomberg",
            "Yahoo Finance",
            "SEBI Filings",
            "RBI Circulars",
            "BSE Filings",
            "CNBC",
            "Mint News"
        ],
        "source_types": ["price", "news", "regulatory", "research"],
        "total_companies": len(indian_companies_db.companies),
        "supported_sectors": list(set(c.sector for c in indian_companies_db.companies.values()))
    }

# Universal Financial Search Endpoints
@app.get("/api/v3/search/universal")
async def universal_search(q: str = Query(..., description="Search any financial instrument")):
    """
    Universal Google-level search for ANY financial instrument
    Supports stocks, mutual funds, ETFs, indices, bonds, commodities
    
    Example: /api/v3/search/universal?q=reliance
    Example: /api/v3/search/universal?q=sbi bluechip
    Example: /api/v3/search/universal?q=nifty
    """
    try:
        if not universal_search_engine:
            raise HTTPException(status_code=503, detail="Universal search engine not initialized")
        
        result = await universal_search_engine.search_universal(q)
        
        return {
            "query": result.query,
            "suggestions": [{
                "text": s.query,
                "category": s.category.value,
                "instrument_type": s.instrument_type.value,
                "symbol": s.symbol,
                "name": s.name,
                "description": s.description,
                "confidence": s.confidence,
                "exchange": s.exchange,
                "sector": s.sector,
                "market_cap": s.market_cap,
                "instant_data": s.instant_data
            } for s in result.suggestions],
            "live_data": result.live_data,
            "instruments_found": result.instruments_found,
            "categories_covered": result.categories_covered,
            "response_time_ms": result.total_time_ms,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Universal search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v3/search/autocomplete")
async def universal_autocomplete(q: str = Query(..., description="Get autocomplete for any instrument")):
    """
    Universal autocomplete for any financial instrument
    
    Example: /api/v3/search/autocomplete?q=rel
    Example: /api/v3/search/autocomplete?q=sbi
    """
    try:
        if not universal_search_engine:
            raise HTTPException(status_code=503, detail="Universal search engine not initialized")
        
        suggestions = await universal_search_engine.get_autocomplete(q)
        
        return {
            "query": q,
            "suggestions": suggestions,
            "total_suggestions": len(suggestions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Universal autocomplete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v3/search/instruments/{instrument_type}")
async def search_by_instrument_type(instrument_type: str, q: str = Query("", description="Optional search within instrument type")):
    """
    Search by specific instrument type
    
    Supported types: stock, mutual_fund, etf, index, bond, commodity
    
    Example: /api/v3/search/instruments/mutual_fund?q=sbi
    Example: /api/v3/search/instruments/etf?q=nifty
    """
    try:
        if not universal_search_engine:
            raise HTTPException(status_code=503, detail="Universal search engine not initialized")
        
        # Search with instrument type filter
        search_query = f"{instrument_type} {q}".strip()
        result = await universal_search_engine.search_universal(search_query)
        
        # Filter results by instrument type
        filtered_suggestions = [
            s for s in result.suggestions 
            if s.instrument_type.value == instrument_type
        ]
        
        return {
            "instrument_type": instrument_type,
            "query": q,
            "suggestions": [{
                "text": s.query,
                "symbol": s.symbol,
                "name": s.name,
                "description": s.description,
                "confidence": s.confidence,
                "exchange": s.exchange,
                "instant_data": s.instant_data
            } for s in filtered_suggestions],
            "total_found": len(filtered_suggestions),
            "response_time_ms": result.total_time_ms,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Instrument type search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v3/search/sectors/{sector}")
async def search_by_sector(sector: str, limit: int = Query(20, description="Maximum results")):
    """
    Search companies by sector
    
    Example: /api/v3/search/sectors/banking
    Example: /api/v3/search/sectors/it
    """
    try:
        if not universal_search_engine:
            raise HTTPException(status_code=503, detail="Universal search engine not initialized")
        
        result = await universal_search_engine.search_universal(f"{sector} sector", limit)
        
        # Filter for sector-specific results
        sector_suggestions = [
            s for s in result.suggestions 
            if s.sector and sector.lower() in s.sector.lower()
        ]
        
        return {
            "sector": sector,
            "companies": [{
                "name": s.name,
                "symbol": s.symbol,
                "description": s.description,
                "exchange": s.exchange,
                "market_cap": s.market_cap,
                "instant_data": s.instant_data
            } for s in sector_suggestions],
            "total_companies": len(sector_suggestions),
            "response_time_ms": result.total_time_ms,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Sector search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
