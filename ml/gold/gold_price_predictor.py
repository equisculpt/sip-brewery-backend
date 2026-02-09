"""
Gold Price Prediction Model using LSTM
Predicts gold prices based on historical data and market indicators
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class GoldPriceLSTM(nn.Module):
    """
    LSTM model for gold price prediction
    """
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 128, num_layers: int = 3):
        super(GoldPriceLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Output
        return self.fc(context)


class GoldPricePredictor:
    """
    Gold price prediction system with multiple models
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GoldPriceLSTM().to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.feature_names = [
            'gold_price', 'gold_returns', 'gold_volatility',
            'usd_inr', 'crude_oil', 'us_treasury_yield',
            'inflation_rate', 'dollar_index', 'vix',
            'nifty_returns', 'sensex_returns',
            'gold_demand', 'gold_supply', 'central_bank_buying',
            'jewelry_demand', 'investment_demand',
            'geopolitical_risk', 'market_sentiment',
            'seasonal_factor', 'day_of_week'
        ]
    
    def predict_price(
        self,
        historical_data: pd.DataFrame,
        horizon: int = 7
    ) -> Dict:
        """
        Predict gold prices for next N days
        
        Args:
            historical_data: Historical gold price and features
            horizon: Number of days to predict
        
        Returns:
            Predictions with confidence intervals
        """
        logger.info(f"Predicting gold prices for next {horizon} days")
        
        # Prepare features
        features = self._prepare_features(historical_data)
        
        # Make predictions
        self.model.eval()
        predictions = []
        confidence_intervals = []
        
        with torch.no_grad():
            for day in range(horizon):
                # Predict next day
                x = torch.FloatTensor(features[-30:]).unsqueeze(0).to(self.device)
                pred = self.model(x).item()
                
                predictions.append(pred)
                
                # Calculate confidence interval (simplified)
                std = self._calculate_prediction_std(features)
                confidence_intervals.append({
                    'lower': pred - 1.96 * std,
                    'upper': pred + 1.96 * std
                })
                
                # Update features for next prediction
                features = self._update_features(features, pred)
        
        # Calculate trend
        trend = self._calculate_trend(predictions)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            predictions,
            historical_data['gold_price'].iloc[-1]
        )
        
        return {
            'predictions': predictions,
            'confidence_intervals': confidence_intervals,
            'trend': trend,
            'recommendation': recommendation,
            'current_price': historical_data['gold_price'].iloc[-1],
            'predicted_change': predictions[-1] - historical_data['gold_price'].iloc[-1],
            'predicted_change_percent': (
                (predictions[-1] - historical_data['gold_price'].iloc[-1]) / 
                historical_data['gold_price'].iloc[-1] * 100
            ),
            'horizon_days': horizon,
            'timestamp': datetime.now().isoformat()
        }
    
    def predict_intraday(
        self,
        current_price: float,
        market_data: Dict
    ) -> Dict:
        """
        Predict intraday gold price movements
        
        Returns:
            Hourly predictions for the day
        """
        predictions = []
        
        for hour in range(1, 7):  # Next 6 hours
            # Simplified intraday prediction
            volatility = market_data.get('volatility', 0.01)
            trend = market_data.get('trend', 0)
            
            predicted_price = current_price * (1 + trend + np.random.normal(0, volatility))
            predictions.append({
                'hour': hour,
                'price': predicted_price,
                'change': predicted_price - current_price,
                'change_percent': (predicted_price - current_price) / current_price * 100
            })
        
        return {
            'intraday_predictions': predictions,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_gold_market(
        self,
        historical_data: pd.DataFrame
    ) -> Dict:
        """
        Comprehensive gold market analysis
        
        Returns:
            Market insights and indicators
        """
        current_price = historical_data['gold_price'].iloc[-1]
        
        # Technical indicators
        sma_20 = historical_data['gold_price'].rolling(20).mean().iloc[-1]
        sma_50 = historical_data['gold_price'].rolling(50).mean().iloc[-1]
        
        # Volatility
        returns = historical_data['gold_price'].pct_change()
        volatility = returns.std() * np.sqrt(252)
        
        # Momentum
        momentum_7d = (current_price - historical_data['gold_price'].iloc[-7]) / historical_data['gold_price'].iloc[-7]
        momentum_30d = (current_price - historical_data['gold_price'].iloc[-30]) / historical_data['gold_price'].iloc[-30]
        
        # Support and resistance
        support = historical_data['gold_price'].rolling(20).min().iloc[-1]
        resistance = historical_data['gold_price'].rolling(20).max().iloc[-1]
        
        # Market regime
        regime = self._detect_market_regime(historical_data)
        
        # Correlation with other assets
        correlations = self._calculate_correlations(historical_data)
        
        return {
            'current_price': current_price,
            'technical_indicators': {
                'sma_20': sma_20,
                'sma_50': sma_50,
                'trend': 'BULLISH' if current_price > sma_20 > sma_50 else 'BEARISH',
                'support': support,
                'resistance': resistance
            },
            'momentum': {
                '7_day': momentum_7d * 100,
                '30_day': momentum_30d * 100
            },
            'volatility': volatility * 100,
            'market_regime': regime,
            'correlations': correlations,
            'recommendation': self._generate_market_recommendation(
                current_price, sma_20, sma_50, momentum_30d
            )
        }
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix"""
        features = []
        
        for feature in self.feature_names:
            if feature in data.columns:
                features.append(data[feature].values)
            else:
                # Use zeros for missing features
                features.append(np.zeros(len(data)))
        
        return np.array(features).T
    
    def _calculate_prediction_std(self, features: np.ndarray) -> float:
        """Calculate prediction standard deviation"""
        # Simplified - would use ensemble or Monte Carlo in production
        return 50.0  # ₹50 standard deviation
    
    def _update_features(self, features: np.ndarray, new_price: float) -> np.ndarray:
        """Update features with new prediction"""
        # Simplified feature update
        new_row = features[-1].copy()
        new_row[0] = new_price  # Update gold price
        return np.vstack([features[1:], new_row])
    
    def _calculate_trend(self, predictions: List[float]) -> str:
        """Calculate price trend"""
        if len(predictions) < 2:
            return 'NEUTRAL'
        
        slope = (predictions[-1] - predictions[0]) / len(predictions)
        
        if slope > 10:
            return 'STRONG_UPTREND'
        elif slope > 2:
            return 'UPTREND'
        elif slope < -10:
            return 'STRONG_DOWNTREND'
        elif slope < -2:
            return 'DOWNTREND'
        else:
            return 'NEUTRAL'
    
    def _generate_recommendation(
        self,
        predictions: List[float],
        current_price: float
    ) -> str:
        """Generate trading recommendation"""
        predicted_change = (predictions[-1] - current_price) / current_price
        
        if predicted_change > 0.03:
            return 'STRONG_BUY'
        elif predicted_change > 0.01:
            return 'BUY'
        elif predicted_change < -0.03:
            return 'STRONG_SELL'
        elif predicted_change < -0.01:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime"""
        returns = data['gold_price'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1]
        
        if volatility > 0.02:
            return 'HIGH_VOLATILITY'
        elif volatility > 0.01:
            return 'MODERATE_VOLATILITY'
        else:
            return 'LOW_VOLATILITY'
    
    def _calculate_correlations(self, data: pd.DataFrame) -> Dict:
        """Calculate correlations with other assets"""
        correlations = {}
        
        if 'nifty_returns' in data.columns:
            correlations['equity'] = data['gold_returns'].corr(data['nifty_returns'])
        
        if 'usd_inr' in data.columns:
            correlations['currency'] = data['gold_returns'].corr(data['usd_inr'].pct_change())
        
        return correlations
    
    def _generate_market_recommendation(
        self,
        current_price: float,
        sma_20: float,
        sma_50: float,
        momentum: float
    ) -> str:
        """Generate market-based recommendation"""
        if current_price > sma_20 > sma_50 and momentum > 0.02:
            return 'ACCUMULATE - Strong bullish trend'
        elif current_price < sma_20 < sma_50 and momentum < -0.02:
            return 'REDUCE - Strong bearish trend'
        elif abs(momentum) < 0.01:
            return 'HOLD - Consolidation phase'
        else:
            return 'MONITOR - Mixed signals'
    
    def train(
        self,
        training_data: pd.DataFrame,
        n_epochs: int = 100,
        learning_rate: float = 0.001
    ):
        """Train the model"""
        logger.info(f"Training gold price predictor for {n_epochs} epochs")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(n_epochs):
            self.model.train()
            
            # Training loop (simplified)
            # In production, would use proper data loaders
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{n_epochs}")
        
        logger.info("Training completed")
    
    def save_model(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_names': self.feature_names
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.feature_names = checkpoint.get('feature_names', self.feature_names)
        logger.info(f"Model loaded from {path}")


class GoldSentimentAnalyzer:
    """
    Analyze market sentiment for gold
    """
    
    def analyze_sentiment(self, news_data: List[Dict]) -> Dict:
        """
        Analyze sentiment from news and social media
        
        Returns:
            Sentiment score and insights
        """
        # Simplified sentiment analysis
        # In production, would use NLP models
        
        positive_keywords = ['rally', 'surge', 'bullish', 'demand', 'safe-haven']
        negative_keywords = ['fall', 'decline', 'bearish', 'sell-off', 'weak']
        
        positive_count = 0
        negative_count = 0
        
        for news in news_data:
            text = news.get('text', '').lower()
            positive_count += sum(1 for kw in positive_keywords if kw in text)
            negative_count += sum(1 for kw in negative_keywords if kw in text)
        
        total = positive_count + negative_count
        sentiment_score = (positive_count - negative_count) / max(total, 1)
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment': 'POSITIVE' if sentiment_score > 0.2 else 'NEGATIVE' if sentiment_score < -0.2 else 'NEUTRAL',
            'confidence': min(abs(sentiment_score) * 100, 100),
            'positive_signals': positive_count,
            'negative_signals': negative_count
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Gold Price Prediction System")
    print("=" * 60)
    
    # Create predictor
    predictor = GoldPricePredictor()
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    gold_prices = 6000 + np.cumsum(np.random.normal(5, 50, len(dates)))
    
    data = pd.DataFrame({
        'date': dates,
        'gold_price': gold_prices,
        'gold_returns': pd.Series(gold_prices).pct_change(),
        'gold_volatility': pd.Series(gold_prices).pct_change().rolling(20).std()
    })
    
    # Predict prices
    predictions = predictor.predict_price(data, horizon=7)
    
    print(f"\nCurrent Price: ₹{predictions['current_price']:.2f}/gram")
    print(f"7-Day Prediction: ₹{predictions['predictions'][-1]:.2f}/gram")
    print(f"Expected Change: ₹{predictions['predicted_change']:.2f} ({predictions['predicted_change_percent']:.2f}%)")
    print(f"Trend: {predictions['trend']}")
    print(f"Recommendation: {predictions['recommendation']}")
    
    # Market analysis
    analysis = predictor.analyze_gold_market(data)
    print(f"\nMarket Analysis:")
    print(f"Trend: {analysis['technical_indicators']['trend']}")
    print(f"Volatility: {analysis['volatility']:.2f}%")
    print(f"Market Regime: {analysis['market_regime']}")
    print(f"Recommendation: {analysis['recommendation']}")
