"""
ASI Integration Layer for Finance Search Engine
Real-time data streaming, automated insight generation, predictive analytics
"""
import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import websockets
import aioredis
from enum import Enum

# ML Libraries for predictive analytics
try:
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    logging.warning("ML libraries not available. Predictive analytics will be limited.")

class InsightType(Enum):
    PRICE_MOVEMENT = "price_movement"
    VOLUME_ANOMALY = "volume_anomaly"
    NEWS_SENTIMENT = "news_sentiment"
    TECHNICAL_PATTERN = "technical_pattern"
    FUNDAMENTAL_CHANGE = "fundamental_change"
    MARKET_CORRELATION = "market_correlation"
    RISK_ALERT = "risk_alert"
    OPPORTUNITY = "opportunity"

class AlertPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class MarketInsight:
    insight_id: str
    symbol: str
    insight_type: InsightType
    priority: AlertPriority
    confidence: float
    title: str
    description: str
    data: Dict[str, Any]
    timestamp: datetime
    expiry: Optional[datetime] = None
    actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictiveSignal:
    symbol: str
    signal_type: str  # buy, sell, hold, watch
    confidence: float
    target_price: Optional[float]
    time_horizon: str  # short, medium, long
    reasoning: List[str]
    risk_factors: List[str]
    supporting_data: Dict[str, Any]
    timestamp: datetime

@dataclass
class RealTimeDataPoint:
    symbol: str
    data_type: str  # price, volume, news, sentiment
    value: Any
    timestamp: datetime
    source: str
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class RealTimeDataStreamer:
    """Stream real-time financial data to ASI"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.data_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.streaming = False
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            logging.info("Connected to Redis for real-time streaming")
        except Exception as e:
            logging.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            
    async def start_streaming(self):
        """Start real-time data streaming"""
        self.streaming = True
        
        # Start multiple streaming tasks
        tasks = [
            self._stream_price_data(),
            self._stream_news_data(),
            self._stream_sentiment_data(),
            self._process_data_buffer()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def stop_streaming(self):
        """Stop real-time data streaming"""
        self.streaming = False
        if self.redis_client:
            await self.redis_client.close()
            
    def subscribe(self, data_type: str, callback: Callable[[RealTimeDataPoint], None]):
        """Subscribe to real-time data updates"""
        self.subscribers[data_type].append(callback)
        
    async def publish_data(self, data_point: RealTimeDataPoint):
        """Publish data point to subscribers and Redis"""
        # Add to buffer
        self.data_buffer[data_point.symbol].append(data_point)
        
        # Notify subscribers
        for callback in self.subscribers[data_point.data_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data_point)
                else:
                    callback(data_point)
            except Exception as e:
                logging.error(f"Error in subscriber callback: {e}")
                
        # Publish to Redis
        if self.redis_client:
            try:
                await self.redis_client.publish(
                    f"finance_data:{data_point.data_type}",
                    json.dumps({
                        'symbol': data_point.symbol,
                        'value': data_point.value,
                        'timestamp': data_point.timestamp.isoformat(),
                        'source': data_point.source,
                        'quality_score': data_point.quality_score,
                        'metadata': data_point.metadata
                    })
                )
            except Exception as e:
                logging.error(f"Failed to publish to Redis: {e}")
                
    async def _stream_price_data(self):
        """Stream real-time price data"""
        while self.streaming:
            try:
                # This would integrate with real price feeds (NSE, BSE APIs)
                # For now, simulate price updates
                await asyncio.sleep(1)
                
                # In production, this would fetch from real APIs
                # symbols = ['TCS', 'INFY', 'RELIANCE', 'HDFCBANK']
                # for symbol in symbols:
                #     price_data = await fetch_real_price(symbol)
                #     await self.publish_data(RealTimeDataPoint(...))
                
            except Exception as e:
                logging.error(f"Error in price streaming: {e}")
                await asyncio.sleep(5)
                
    async def _stream_news_data(self):
        """Stream real-time news data"""
        while self.streaming:
            try:
                # This would integrate with news APIs and RSS feeds
                await asyncio.sleep(10)
                
                # In production, this would fetch from real news sources
                # news_items = await fetch_latest_news()
                # for item in news_items:
                #     await self.publish_data(RealTimeDataPoint(...))
                
            except Exception as e:
                logging.error(f"Error in news streaming: {e}")
                await asyncio.sleep(30)
                
    async def _stream_sentiment_data(self):
        """Stream real-time sentiment analysis"""
        while self.streaming:
            try:
                # This would analyze social media, news sentiment
                await asyncio.sleep(30)
                
                # In production, this would analyze sentiment from various sources
                # sentiment_data = await analyze_market_sentiment()
                # await self.publish_data(RealTimeDataPoint(...))
                
            except Exception as e:
                logging.error(f"Error in sentiment streaming: {e}")
                await asyncio.sleep(60)
                
    async def _process_data_buffer(self):
        """Process buffered data for patterns and insights"""
        while self.streaming:
            try:
                await asyncio.sleep(5)
                
                # Process data buffer for patterns
                for symbol, buffer in self.data_buffer.items():
                    if len(buffer) > 10:  # Minimum data points for analysis
                        await self._analyze_data_patterns(symbol, list(buffer))
                        
            except Exception as e:
                logging.error(f"Error in data processing: {e}")
                await asyncio.sleep(10)
                
    async def _analyze_data_patterns(self, symbol: str, data_points: List[RealTimeDataPoint]):
        """Analyze data patterns and generate insights"""
        # This would implement pattern recognition algorithms
        # For now, placeholder for pattern analysis
        pass

class InsightGenerator:
    """Generate automated insights from financial data"""
    
    def __init__(self):
        self.insight_rules = self._load_insight_rules()
        self.pattern_detectors = self._initialize_pattern_detectors()
        
    def _load_insight_rules(self) -> Dict[str, Any]:
        """Load insight generation rules"""
        return {
            'price_movement': {
                'significant_change_threshold': 0.05,  # 5%
                'volume_spike_threshold': 2.0,  # 2x average
                'time_window': 300  # 5 minutes
            },
            'news_sentiment': {
                'sentiment_threshold': 0.7,
                'impact_score_threshold': 0.6
            },
            'technical_patterns': {
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'volume_ma_threshold': 1.5
            }
        }
        
    def _initialize_pattern_detectors(self) -> Dict[str, Callable]:
        """Initialize pattern detection algorithms"""
        return {
            'price_spike': self._detect_price_spike,
            'volume_anomaly': self._detect_volume_anomaly,
            'sentiment_shift': self._detect_sentiment_shift,
            'correlation_break': self._detect_correlation_break
        }
        
    async def generate_insights(self, symbol: str, data_points: List[RealTimeDataPoint]) -> List[MarketInsight]:
        """Generate insights from data points"""
        insights = []
        
        # Run all pattern detectors
        for pattern_name, detector in self.pattern_detectors.items():
            try:
                insight = await detector(symbol, data_points)
                if insight:
                    insights.append(insight)
            except Exception as e:
                logging.error(f"Error in pattern detector {pattern_name}: {e}")
                
        return insights
        
    async def _detect_price_spike(self, symbol: str, data_points: List[RealTimeDataPoint]) -> Optional[MarketInsight]:
        """Detect significant price movements"""
        price_points = [dp for dp in data_points if dp.data_type == 'price']
        
        if len(price_points) < 2:
            return None
            
        latest_price = float(price_points[-1].value)
        previous_price = float(price_points[-2].value)
        
        change_percent = (latest_price - previous_price) / previous_price
        
        if abs(change_percent) > self.insight_rules['price_movement']['significant_change_threshold']:
            direction = "increased" if change_percent > 0 else "decreased"
            
            return MarketInsight(
                insight_id=f"price_spike_{symbol}_{int(datetime.now().timestamp())}",
                symbol=symbol,
                insight_type=InsightType.PRICE_MOVEMENT,
                priority=AlertPriority.HIGH if abs(change_percent) > 0.1 else AlertPriority.MEDIUM,
                confidence=0.9,
                title=f"{symbol} Price Alert",
                description=f"{symbol} price has {direction} by {abs(change_percent)*100:.2f}% to ₹{latest_price}",
                data={
                    'previous_price': previous_price,
                    'current_price': latest_price,
                    'change_percent': change_percent,
                    'change_amount': latest_price - previous_price
                },
                timestamp=datetime.now(),
                actions=['monitor_closely', 'check_news', 'analyze_volume']
            )
            
        return None
        
    async def _detect_volume_anomaly(self, symbol: str, data_points: List[RealTimeDataPoint]) -> Optional[MarketInsight]:
        """Detect unusual volume patterns"""
        volume_points = [dp for dp in data_points if dp.data_type == 'volume']
        
        if len(volume_points) < 10:
            return None
            
        volumes = [float(dp.value) for dp in volume_points]
        avg_volume = np.mean(volumes[:-1])  # Exclude latest
        latest_volume = volumes[-1]
        
        if latest_volume > avg_volume * self.insight_rules['price_movement']['volume_spike_threshold']:
            return MarketInsight(
                insight_id=f"volume_spike_{symbol}_{int(datetime.now().timestamp())}",
                symbol=symbol,
                insight_type=InsightType.VOLUME_ANOMALY,
                priority=AlertPriority.MEDIUM,
                confidence=0.8,
                title=f"{symbol} Volume Spike",
                description=f"Unusual volume detected: {latest_volume:,.0f} vs avg {avg_volume:,.0f}",
                data={
                    'current_volume': latest_volume,
                    'average_volume': avg_volume,
                    'spike_ratio': latest_volume / avg_volume
                },
                timestamp=datetime.now(),
                actions=['investigate_cause', 'check_news', 'monitor_price']
            )
            
        return None
        
    async def _detect_sentiment_shift(self, symbol: str, data_points: List[RealTimeDataPoint]) -> Optional[MarketInsight]:
        """Detect sentiment changes"""
        sentiment_points = [dp for dp in data_points if dp.data_type == 'sentiment']
        
        if len(sentiment_points) < 5:
            return None
            
        # Analyze sentiment trend
        recent_sentiment = np.mean([float(dp.value) for dp in sentiment_points[-3:]])
        older_sentiment = np.mean([float(dp.value) for dp in sentiment_points[-5:-2]])
        
        sentiment_change = recent_sentiment - older_sentiment
        
        if abs(sentiment_change) > 0.3:  # Significant sentiment shift
            direction = "improved" if sentiment_change > 0 else "deteriorated"
            
            return MarketInsight(
                insight_id=f"sentiment_shift_{symbol}_{int(datetime.now().timestamp())}",
                symbol=symbol,
                insight_type=InsightType.NEWS_SENTIMENT,
                priority=AlertPriority.MEDIUM,
                confidence=0.7,
                title=f"{symbol} Sentiment Change",
                description=f"Market sentiment has {direction} significantly",
                data={
                    'recent_sentiment': recent_sentiment,
                    'previous_sentiment': older_sentiment,
                    'sentiment_change': sentiment_change
                },
                timestamp=datetime.now(),
                actions=['review_news', 'assess_impact', 'monitor_price']
            )
            
        return None
        
    async def _detect_correlation_break(self, symbol: str, data_points: List[RealTimeDataPoint]) -> Optional[MarketInsight]:
        """Detect correlation breakdowns with market/sector"""
        # This would require market/sector data for comparison
        # Placeholder for correlation analysis
        return None

class PredictiveAnalytics:
    """Predictive analytics for price and trend forecasting"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        if HAS_ML_LIBS:
            self.ml_available = True
        else:
            self.ml_available = False
            logging.warning("ML libraries not available. Predictive analytics disabled.")
            
    async def train_model(self, symbol: str, historical_data: pd.DataFrame) -> bool:
        """Train predictive model for a symbol"""
        if not self.ml_available:
            return False
            
        try:
            # Prepare features
            features = self._prepare_features(historical_data)
            target = historical_data['close'].shift(-1).dropna()  # Next day price
            
            # Align features and target
            features = features.iloc[:-1]  # Remove last row
            
            # Split data
            split_idx = int(len(features) * 0.8)
            X_train, X_test = features[:split_idx], features[split_idx:]
            y_train, y_test = target[:split_idx], target[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble model
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store model and performance
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.model_performance[symbol] = {
                'mse': mse,
                'r2': r2,
                'train_date': datetime.now().isoformat()
            }
            
            logging.info(f"Model trained for {symbol}: R² = {r2:.3f}, MSE = {mse:.3f}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to train model for {symbol}: {e}")
            return False
            
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        features = pd.DataFrame()
        
        # Price-based features
        features['price'] = data['close']
        features['price_change'] = data['close'].pct_change()
        features['price_ma_5'] = data['close'].rolling(5).mean()
        features['price_ma_20'] = data['close'].rolling(20).mean()
        
        # Volume features
        features['volume'] = data['volume']
        features['volume_ma'] = data['volume'].rolling(10).mean()
        features['volume_ratio'] = data['volume'] / features['volume_ma']
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['close'])
        features['macd'] = self._calculate_macd(data['close'])
        
        # Volatility
        features['volatility'] = data['close'].rolling(10).std()
        
        # Time features
        features['day_of_week'] = pd.to_datetime(data.index).dayofweek
        features['month'] = pd.to_datetime(data.index).month
        
        return features.dropna()
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        return ema_12 - ema_26
        
    async def generate_prediction(self, symbol: str, current_data: Dict[str, Any]) -> Optional[PredictiveSignal]:
        """Generate prediction for a symbol"""
        if not self.ml_available or symbol not in self.models:
            return None
            
        try:
            model = self.models[symbol]
            scaler = self.scalers[symbol]
            
            # Prepare current features (this would need current market data)
            # For now, return a placeholder prediction
            
            return PredictiveSignal(
                symbol=symbol,
                signal_type="hold",  # Placeholder
                confidence=0.6,
                target_price=None,
                time_horizon="short",
                reasoning=["Model-based prediction", "Technical analysis"],
                risk_factors=["Market volatility", "Economic uncertainty"],
                supporting_data=current_data,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logging.error(f"Failed to generate prediction for {symbol}: {e}")
            return None

class ASIIntegrationEngine:
    """Main ASI integration engine coordinating all components"""
    
    def __init__(self):
        self.data_streamer = RealTimeDataStreamer()
        self.insight_generator = InsightGenerator()
        self.predictive_analytics = PredictiveAnalytics()
        
        self.active_insights: Dict[str, MarketInsight] = {}
        self.prediction_cache: Dict[str, PredictiveSignal] = {}
        
        # WebSocket connections for real-time updates
        self.websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
        
    async def initialize(self):
        """Initialize all components"""
        await self.data_streamer.initialize()
        
        # Subscribe to data updates
        self.data_streamer.subscribe('price', self._handle_price_update)
        self.data_streamer.subscribe('volume', self._handle_volume_update)
        self.data_streamer.subscribe('news', self._handle_news_update)
        
    async def start(self):
        """Start the ASI integration engine"""
        logging.info("Starting ASI Integration Engine...")
        
        # Start data streaming
        streaming_task = asyncio.create_task(self.data_streamer.start_streaming())
        
        # Start WebSocket server for real-time updates
        websocket_task = asyncio.create_task(self._start_websocket_server())
        
        # Start insight processing
        insight_task = asyncio.create_task(self._process_insights())
        
        await asyncio.gather(streaming_task, websocket_task, insight_task, return_exceptions=True)
        
    async def _handle_price_update(self, data_point: RealTimeDataPoint):
        """Handle real-time price updates"""
        # Generate insights from price data
        recent_data = list(self.data_streamer.data_buffer[data_point.symbol])
        insights = await self.insight_generator.generate_insights(data_point.symbol, recent_data)
        
        for insight in insights:
            self.active_insights[insight.insight_id] = insight
            await self._broadcast_insight(insight)
            
    async def _handle_volume_update(self, data_point: RealTimeDataPoint):
        """Handle real-time volume updates"""
        # Similar to price updates
        pass
        
    async def _handle_news_update(self, data_point: RealTimeDataPoint):
        """Handle real-time news updates"""
        # Process news for sentiment and impact
        pass
        
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        async def handle_client(websocket, path):
            self.websocket_clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.websocket_clients.discard(websocket)
                
        # Start WebSocket server on port 8765
        start_server = websockets.serve(handle_client, "localhost", 8765)
        await start_server
        
    async def _broadcast_insight(self, insight: MarketInsight):
        """Broadcast insight to all connected clients"""
        if self.websocket_clients:
            message = json.dumps({
                'type': 'insight',
                'data': {
                    'insight_id': insight.insight_id,
                    'symbol': insight.symbol,
                    'insight_type': insight.insight_type.value,
                    'priority': insight.priority.value,
                    'title': insight.title,
                    'description': insight.description,
                    'confidence': insight.confidence,
                    'timestamp': insight.timestamp.isoformat(),
                    'data': insight.data
                }
            })
            
            # Send to all clients
            disconnected = set()
            for client in self.websocket_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
                    
            # Remove disconnected clients
            self.websocket_clients -= disconnected
            
    async def _process_insights(self):
        """Process and manage insights"""
        while True:
            try:
                # Clean up expired insights
                current_time = datetime.now()
                expired_insights = [
                    insight_id for insight_id, insight in self.active_insights.items()
                    if insight.expiry and insight.expiry < current_time
                ]
                
                for insight_id in expired_insights:
                    del self.active_insights[insight_id]
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logging.error(f"Error in insight processing: {e}")
                await asyncio.sleep(60)
                
    async def get_insights_for_symbol(self, symbol: str) -> List[MarketInsight]:
        """Get active insights for a symbol"""
        return [
            insight for insight in self.active_insights.values()
            if insight.symbol == symbol
        ]
        
    async def get_prediction_for_symbol(self, symbol: str) -> Optional[PredictiveSignal]:
        """Get prediction for a symbol"""
        if symbol in self.prediction_cache:
            prediction = self.prediction_cache[symbol]
            # Check if prediction is still valid (not too old)
            if (datetime.now() - prediction.timestamp).total_seconds() < 3600:  # 1 hour
                return prediction
                
        # Generate new prediction
        current_data = {}  # This would be populated with current market data
        prediction = await self.predictive_analytics.generate_prediction(symbol, current_data)
        
        if prediction:
            self.prediction_cache[symbol] = prediction
            
        return prediction

# Global instance
asi_integration_engine = ASIIntegrationEngine()
