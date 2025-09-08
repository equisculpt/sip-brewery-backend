"""
🌟 INTEGRATED ASI MASTER
Complete integration of all ASI components for True ASI System
Real Market Data + Continuous Learning + Financial LLMs + Python ASI Master

@author 35+ Year Experienced AI Engineer
@version 2.0.0 - Complete ASI Integration
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import json
from dataclasses import dataclass, asdict
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# Import our ASI components (relative imports within package)
from .real_market_data_asi import RealMarketDataASI
from .continuous_learning_asi import ContinuousLearningASI
from .financial_llm_asi import FinancialLLMASI
from .python_asi_master import PythonASIMaster, ASIRequest, ASIResponse
from .autonomous_learning_curriculum import AutonomousLearningCurriculum
from .meta_learning_asi import MetaLearningASI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("integrated_asi_master")

@dataclass
class IntegratedASIResponse:
    request_id: str
    query: str
    asi_response: Dict[str, Any]
    market_data: Dict[str, Any]
    financial_analysis: Dict[str, Any]
    learning_stats: Dict[str, Any]
    confidence: float
    processing_time: float
    timestamp: datetime

class QueryRequest(BaseModel):
    query: str
    query_type: str = "general"
    symbols: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None
    require_market_data: bool = True
    require_financial_analysis: bool = True
    precision: str = "high"
    urgency: str = "normal"

class PairwiseRankRequest(BaseModel):
    """Request to compare two instruments/funds and pick the likely better performer.
    - a, b: equity symbol (e.g., RELIANCE) or mutual fund scheme code (digits)
    - asset_type: 'stock' | 'fund' | 'auto'
    - horizon: '1D' | '1W' | '1M' (advisory; current implementation uses available signals)
    """
    a: str
    b: str
    asset_type: Optional[str] = "auto"
    horizon: Optional[str] = "1D"

class IntegratedASIMaster:
    """
    Integrated ASI Master that combines all ASI components
    for complete artificial super intelligence capabilities
    """
    
    def __init__(self):
        # Initialize all ASI components
        self.python_asi = PythonASIMaster()
        self.market_data_asi = RealMarketDataASI()
        self.continuous_learning_asi = ContinuousLearningASI()
        self.financial_llm_asi = FinancialLLMASI()
        self.autonomous_curriculum = AutonomousLearningCurriculum()
        self.meta_learner = MetaLearningASI()
        
        # System state
        self.initialized = False
        self.request_counter = 0
        
        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'average_response_time': 0.0,
            'accuracy_score': 0.0,
            'uptime_start': datetime.now()
        }
        
        logger.info("🌟 Integrated ASI Master initialized")
    
    async def initialize(self):
        """Initialize all ASI components"""
        if self.initialized:
            return
        
        logger.info("🚀 Initializing Integrated ASI Master...")
        
        try:
            # Initialize all components in parallel
            init_tasks = [
                self.python_asi.initialize(),
                self.market_data_asi.initialize(),
                self.continuous_learning_asi.start_continuous_learning(),
                self.financial_llm_asi.initialize_models(),
                self.autonomous_curriculum.initialize(),
                self.meta_learner.initialize()
            ]
            
            await asyncio.gather(*init_tasks)
            
            self.initialized = True
            logger.info("✅ Integrated ASI Master fully initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ASI Master: {e}")
            raise e
    
    async def process_integrated_request(self, request: QueryRequest) -> IntegratedASIResponse:
        """
        Process a complete ASI request using all components
        """
        start_time = datetime.now()
        self.request_counter += 1
        request_id = f"asi_req_{self.request_counter}_{int(start_time.timestamp())}"
        
        logger.info(f"🧠 Processing integrated request {request_id}: {request.query[:100]}...")
        
        try:
            # Ensure system is initialized
            if not self.initialized:
                await self.initialize()
            
            # Create ASI request for Python ASI Master
            asi_request = ASIRequest(
                query=request.query,
                type=request.query_type,
                data=request.context or {},
                precision=request.precision,
                urgency=request.urgency
            )
            
            # Process with all components in parallel
            tasks = []
            
            # 1. Core ASI Processing
            tasks.append(self._process_core_asi(asi_request))
            
            # 2. Market Data Collection (if required)
            if request.require_market_data and request.symbols:
                tasks.append(self._collect_market_data(request.symbols))
            else:
                tasks.append(asyncio.create_task(self._return_empty_market_data()))
            
            # 3. Financial LLM Analysis (if required)
            if request.require_financial_analysis:
                tasks.append(self._perform_financial_analysis(request.query, request.context))
            else:
                tasks.append(asyncio.create_task(self._return_empty_financial_analysis()))
            
            # 4. Get learning statistics
            tasks.append(self._get_learning_stats())
            
            # 5. Get autonomous learning status
            tasks.append(self._get_autonomous_learning_status())
            
            # 6. Get meta-learning insights
            tasks.append(self._get_meta_learning_insights())
            
            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            asi_response = results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])}
            market_data = results[1] if not isinstance(results[1], Exception) else {}
            financial_analysis = results[2] if not isinstance(results[2], Exception) else {}
            learning_stats = results[3] if not isinstance(results[3], Exception) else {}
            autonomous_learning_status = results[4] if not isinstance(results[4], Exception) else {}
            meta_learning_insights = results[5] if not isinstance(results[5], Exception) else {}
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(asi_response, market_data, financial_analysis)
            
            # Record prediction for learning (if applicable)
            if 'prediction' in asi_response and request.symbols:
                await self._record_prediction_for_learning(
                    request_id, request.symbols[0], request.query_type,
                    asi_response['prediction'], asi_response.get('model_version', 'v1.0')
                )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update performance metrics
            self._update_performance_metrics(processing_time, confidence)
            
            # Create integrated response
            integrated_response = IntegratedASIResponse(
                request_id=request_id,
                query=request.query,
                asi_response=asi_response,
                market_data=market_data,
                financial_analysis=financial_analysis,
                learning_stats={
                    **learning_stats,
                    'autonomous_learning': autonomous_learning_status,
                    'meta_learning': meta_learning_insights
                },
                confidence=confidence,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            logger.info(f"✅ Completed integrated request {request_id} in {processing_time:.2f}s (Confidence: {confidence:.2f})")
            
            return integrated_response
            
        except Exception as e:
            logger.error(f"❌ Error processing integrated request {request_id}: {e}")
            raise HTTPException(status_code=500, detail=f"ASI processing error: {str(e)}")
    
    async def _process_core_asi(self, request: ASIRequest) -> Dict[str, Any]:
        """Process request with core Python ASI Master"""
        try:
            response = await self.python_asi.process_request(request)
            return asdict(response)
        except Exception as e:
            logger.error(f"Core ASI processing error: {e}")
            return {"error": str(e)}
    
    async def _collect_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Collect real-time market data"""
        try:
            market_data = {}
            
            for symbol in symbols:
                # Get real-time data
                data = await self.market_data_asi.get_real_time_data(symbol)
                market_data[symbol] = data
                
                # Add mutual fund data if symbol looks like scheme code
                if symbol.isdigit():
                    fund_data = await self.market_data_asi.get_mutual_fund_data(symbol)
                    market_data[f"{symbol}_fund"] = fund_data
            
            # Get market indices
            indices = await self.market_data_asi.get_market_indices()
            market_data['indices'] = indices
            
            # Get economic calendar
            events = await self.market_data_asi.get_economic_calendar()
            market_data['economic_events'] = events
            
            return market_data
            
        except Exception as e:
            logger.error(f"Market data collection error: {e}")
            return {"error": str(e)}
    
    async def _return_empty_market_data(self) -> Dict[str, Any]:
        """Return empty market data structure"""
        return {"message": "Market data not requested"}
    
    async def rank_pairwise(self, a: str, b: str, asset_type: Optional[str] = None, horizon: str = "1D") -> Dict[str, Any]:
        """Compare two instruments (stocks or mutual funds) and rank the likely better performer.
        Heuristics (initial version):
        - Stocks: prefer higher intraday momentum (change_percent) with data confidence.
        - Funds: prefer higher trailing returns_1y (fallback to nav if unavailable).
        Returns a structured result with winner, scores, features, and confidence.
        """
        def is_fund(x: str) -> bool:
            return x.isdigit()

        atype = (asset_type or "auto").lower()
        if atype == "auto":
            if is_fund(a) and is_fund(b):
                atype = "fund"
            else:
                atype = "stock"

        try:
            if atype == "stock":
                data_a = await self.market_data_asi.get_real_time_data(a)
                data_b = await self.market_data_asi.get_real_time_data(b)
                score_a = float(data_a.get("change_percent") or 0.0)
                score_b = float(data_b.get("change_percent") or 0.0)
                conf_a = float(data_a.get("confidence") or 0.0)
                conf_b = float(data_b.get("confidence") or 0.0)
                winner = a if score_a > score_b else b
                confidence = float(min(0.99, 0.5 * (conf_a + conf_b) + 0.25 * (1.0 if winner == a and score_a - score_b != 0 else 0.5)))
                return {
                    "type": "stock",
                    "horizon": horizon,
                    "winner": winner,
                    "scores": {a: score_a, b: score_b},
                    "features": {
                        a: {"change_percent": score_a, "confidence": conf_a},
                        b: {"change_percent": score_b, "confidence": conf_b},
                    },
                    "confidence": round(confidence, 3),
                    "reason": "Selected higher intraday momentum proxy (change_percent) with data confidence weighting"
                }
            else:
                # fund comparison by scheme code
                data_a = await self.market_data_asi.get_mutual_fund_data(a)
                data_b = await self.market_data_asi.get_mutual_fund_data(b)
                # Prefer trailing 1Y returns if available, else NAV level as weak proxy
                ra = data_a.get("returns_1y")
                rb = data_b.get("returns_1y")
                if ra is None:
                    ra = (data_a.get("nav") or 0.0)
                if rb is None:
                    rb = (data_b.get("nav") or 0.0)
                score_a = float(ra or 0.0)
                score_b = float(rb or 0.0)
                conf_a = float(data_a.get("confidence") or 0.0)
                conf_b = float(data_b.get("confidence") or 0.0)
                winner = a if score_a > score_b else b
                confidence = float(min(0.99, 0.5 * (conf_a + conf_b) + 0.25 * (1.0 if winner == a and score_a - score_b != 0 else 0.5)))
                return {
                    "type": "fund",
                    "horizon": horizon,
                    "winner": winner,
                    "scores": {a: score_a, b: score_b},
                    "features": {
                        a: {"returns_1y_or_nav": score_a, "confidence": conf_a},
                        b: {"returns_1y_or_nav": score_b, "confidence": conf_b},
                    },
                    "confidence": round(confidence, 3),
                    "reason": "Selected higher trailing 1Y return (fallback to NAV) with data confidence weighting"
                }
        except Exception as e:
            logger.error(f"Pairwise ranking error for {a} vs {b}: {e}")
            raise HTTPException(status_code=500, detail=f"pairwise_ranking_failed: {str(e)}")

    async def _perform_financial_analysis(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform advanced financial analysis"""
        try:
            analysis = await self.financial_llm_asi.advanced_financial_reasoning(query, context)
            return asdict(analysis)
        except Exception as e:
            logger.error(f"Financial analysis error: {e}")
            return {"error": str(e)}
    
    async def _return_empty_financial_analysis(self) -> Dict[str, Any]:
        """Return empty financial analysis structure"""
        return {"message": "Financial analysis not requested"}
    
    async def _get_learning_stats(self) -> Dict[str, Any]:
        """Get continuous learning statistics"""
        try:
            return self.continuous_learning_asi.get_learning_stats()
        except Exception as e:
            logger.error(f"Learning stats error: {e}")
            return {"error": str(e)}
    
    async def _get_autonomous_learning_status(self) -> Dict[str, Any]:
        """Get autonomous learning curriculum status"""
        try:
            return self.autonomous_curriculum.get_learning_status()
        except Exception as e:
            logger.error(f"Autonomous learning status error: {e}")
            return {"error": str(e)}
    
    async def _get_meta_learning_insights(self) -> Dict[str, Any]:
        """Get meta-learning insights and recommendations"""
        try:
            return self.meta_learner.get_meta_learning_status()
        except Exception as e:
            logger.error(f"Meta-learning insights error: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_confidence(self, asi_response: Dict, market_data: Dict, 
                                    financial_analysis: Dict) -> float:
        """Calculate overall confidence score"""
        confidence_factors = []
        
        # ASI response confidence
        if 'confidence' in asi_response:
            confidence_factors.append(asi_response['confidence'])
        
        # Market data confidence
        if 'confidence' in market_data:
            confidence_factors.append(market_data['confidence'])
        
        # Financial analysis confidence
        if 'confidence' in financial_analysis:
            confidence_factors.append(financial_analysis['confidence'])
        
        # Default confidence if no factors available
        if not confidence_factors:
            confidence_factors = [0.7]  # Default moderate confidence
        
        return np.mean(confidence_factors)
    
    async def _record_prediction_for_learning(self, request_id: str, symbol: str, 
                                            prediction_type: str, prediction: Any,
                                            model_version: str):
        """Record prediction for continuous learning"""
        try:
            # This would be called later when actual values are available
            # For now, we just log the prediction
            logger.info(f"📝 Recorded prediction {request_id}: {symbol} - {prediction_type} = {prediction}")
        except Exception as e:
            logger.error(f"Error recording prediction: {e}")
    
    def _update_performance_metrics(self, processing_time: float, confidence: float):
        """Update system performance metrics"""
        self.performance_metrics['total_requests'] += 1
        self.performance_metrics['successful_requests'] += 1
        
        # Update average response time
        current_avg = self.performance_metrics['average_response_time']
        total_requests = self.performance_metrics['total_requests']
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.performance_metrics['average_response_time'] = new_avg
        
        # Update accuracy score (using confidence as proxy)
        current_accuracy = self.performance_metrics['accuracy_score']
        new_accuracy = ((current_accuracy * (total_requests - 1)) + confidence) / total_requests
        self.performance_metrics['accuracy_score'] = new_accuracy
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        uptime = (datetime.now() - self.performance_metrics['uptime_start']).total_seconds()
        
        return {
            'system_status': 'operational' if self.initialized else 'initializing',
            'components': {
                'python_asi': self.python_asi.get_system_status(),
                'market_data_asi': self.market_data_asi.get_cache_stats(),
                'continuous_learning_asi': self.continuous_learning_asi.get_learning_stats(),
                'financial_llm_asi': self.financial_llm_asi.get_model_status(),
                'autonomous_curriculum': self.autonomous_curriculum.get_learning_status(),
                'meta_learner': self.meta_learner.get_meta_learning_status()
            },
            'performance_metrics': {
                **self.performance_metrics,
                'uptime_seconds': uptime,
                'uptime_hours': uptime / 3600
            },
            'timestamp': datetime.now()
        }
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("🛑 Shutting down Integrated ASI Master...")
        
        try:
            # Stop continuous learning
            self.continuous_learning_asi.stop_continuous_learning()
            
            # Stop autonomous learning
            self.autonomous_curriculum.stop_autonomous_learning()
            
            # Stop meta-learning
            self.meta_learner.stop_meta_learning()
            
            # Close market data connections
            await self.market_data_asi.close()
            
            # Close autonomous curriculum
            await self.autonomous_curriculum.close()
            
            logger.info("✅ Integrated ASI Master shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Integrated ASI Master API...")
    await asi_master.initialize()
    yield
    # Shutdown
    await asi_master.shutdown()

# Global ASI Master instance
asi_master = IntegratedASIMaster()

# Create FastAPI app
app = FastAPI(
    title="Integrated ASI Master API",
    description="Complete Artificial Super Intelligence System",
    version="2.0.0",
    lifespan=lifespan
)

@app.post("/asi/process", response_model=Dict[str, Any])
async def process_asi_request(request: QueryRequest):
    """Process a complete ASI request"""
    response = await asi_master.process_integrated_request(request)
    return asdict(response)

@app.post("/asi/rank/pairwise")
async def rank_pairwise_api(req: PairwiseRankRequest):
    """Pairwise ranking between two instruments/funds.
    Returns the likely better performer with a confidence score and features used.
    """
    return await asi_master.rank_pairwise(req.a, req.b, req.asset_type, req.horizon)

@app.get("/asi/status")
async def get_system_status():
    """Get system status"""
    return await asi_master.get_system_status()

@app.post("/asi/batch")
async def batch_process(requests: List[QueryRequest]):
    """Process multiple requests in batch"""
    tasks = [asi_master.process_integrated_request(req) for req in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return {
        'batch_size': len(requests),
        'results': [asdict(r) if not isinstance(r, Exception) else {'error': str(r)} for r in results],
        'timestamp': datetime.now()
    }

@app.get("/asi/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy' if asi_master.initialized else 'initializing',
        'timestamp': datetime.now()
    }

@app.post("/asi/feedback")
async def record_feedback(request_id: str, actual_value: float, symbol: str, prediction_type: str):
    """Record actual values for continuous learning"""
    try:
        # This would update the continuous learning system with actual results
        await asi_master.continuous_learning_asi.record_prediction_error(
            prediction_id=request_id,
            symbol=symbol,
            prediction_type=prediction_type,
            predicted_value=0.0,  # Would get from stored prediction
            actual_value=actual_value,
            model_version="v1.0",
            features_used=[]
        )
        
        return {"message": "Feedback recorded successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example usage and testing
async def test_integrated_asi():
    """Test the integrated ASI system"""
    asi = IntegratedASIMaster()
    
    try:
        await asi.initialize()
        
        # Test request
        request = QueryRequest(
            query="What is the investment outlook for Indian banking stocks?",
            query_type="investment_analysis",
            symbols=["HDFCBANK", "ICICIBANK"],
            context={"sector": "banking", "market": "indian"},
            require_market_data=True,
            require_financial_analysis=True,
            precision="high",
            urgency="normal"
        )
        
        # Process request
        response = await asi.process_integrated_request(request)
        
        print(f"Request ID: {response.request_id}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Processing Time: {response.processing_time:.2f}s")
        print(f"ASI Response: {response.asi_response}")
        
        # Get system status
        status = await asi.get_system_status()
        print(f"System Status: {status}")
        
    finally:
        await asi.shutdown()

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "integrated_asi_master:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
