"""
🧠 CONTINUOUS LEARNING ASI
Real-time model training and adaptation system
Continuously learns from prediction errors and market changes

@author 35+ Year Experienced AI Engineer
@version 1.0.0 - Continuous Learning Implementation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from dataclasses import dataclass, asdict
from collections import deque
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("continuous_learning_asi")

@dataclass
class PredictionError:
    prediction_id: str
    symbol: str
    prediction_type: str
    predicted_value: float
    actual_value: float
    error: float
    error_percentage: float
    timestamp: datetime
    model_version: str
    features_used: List[str]

@dataclass
class ModelPerformance:
    model_name: str
    accuracy: float
    mse: float
    mae: float
    r2_score: float
    prediction_count: int
    last_updated: datetime
    version: str

class ContinuousLearningASI:
    """
    Continuous learning system that automatically retrains models
    based on prediction errors and new market data
    """
    
    def __init__(self, model_dir: str = "models", max_errors: int = 1000):
        self.model_dir = model_dir
        self.max_errors = max_errors
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Error tracking
        self.prediction_errors = deque(maxlen=max_errors)
        self.performance_history = {}
        
        # Model registry
        self.models = {}
        self.scalers = {}
        self.model_versions = {}
        
        # Learning configuration
        self.learning_config = {
            'min_errors_for_retrain': 100,
            'retrain_interval_hours': 6,
            'performance_threshold': 0.05,  # 5% error threshold
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'early_stopping_patience': 10
        }
        
        # Background learning control
        self.learning_active = False
        self.learning_thread = None
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🧠 Continuous Learning ASI initialized on {self.device}")
    
    async def start_continuous_learning(self):
        """Start the continuous learning background process"""
        if self.learning_active:
            logger.warning("Continuous learning already active")
            return
        
        self.learning_active = True
        self.learning_thread = threading.Thread(target=self._continuous_learning_loop, daemon=True)
        self.learning_thread.start()
        
        logger.info("🚀 Continuous learning started")
    
    def stop_continuous_learning(self):
        """Stop the continuous learning process"""
        self.learning_active = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        
        logger.info("⏹️ Continuous learning stopped")
    
    def _continuous_learning_loop(self):
        """Main continuous learning loop (runs in background thread)"""
        logger.info("🔄 Continuous learning loop started")
        
        while self.learning_active:
            try:
                # Check if we have enough errors for retraining
                if len(self.prediction_errors) >= self.learning_config['min_errors_for_retrain']:
                    logger.info(f"📊 {len(self.prediction_errors)} prediction errors available for learning")
                    
                    # Group errors by model type
                    error_groups = self._group_errors_by_type()
                    
                    for model_type, errors in error_groups.items():
                        if len(errors) >= 50:  # Minimum errors per model
                            logger.info(f"🔄 Retraining {model_type} model with {len(errors)} errors")
                            self._retrain_model(model_type, errors)
                
                # Sleep for the configured interval
                sleep_time = self.learning_config['retrain_interval_hours'] * 3600
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
                time.sleep(300)  # Sleep 5 minutes on error
    
    async def record_prediction_error(self, prediction_id: str, symbol: str, 
                                    prediction_type: str, predicted_value: float,
                                    actual_value: float, model_version: str,
                                    features_used: List[str]):
        """Record a prediction error for learning"""
        error = abs(predicted_value - actual_value)
        error_percentage = (error / abs(actual_value)) * 100 if actual_value != 0 else 0
        
        prediction_error = PredictionError(
            prediction_id=prediction_id,
            symbol=symbol,
            prediction_type=prediction_type,
            predicted_value=predicted_value,
            actual_value=actual_value,
            error=error,
            error_percentage=error_percentage,
            timestamp=datetime.now(),
            model_version=model_version,
            features_used=features_used
        )
        
        self.prediction_errors.append(prediction_error)
        
        logger.info(f"📝 Recorded prediction error: {prediction_type} for {symbol}, "
                   f"Error: {error:.4f} ({error_percentage:.2f}%)")
        
        # Check if immediate retraining is needed (high error)
        if error_percentage > 20:  # 20% error threshold for immediate action
            await self._trigger_immediate_retraining(prediction_type)
    
    def _group_errors_by_type(self) -> Dict[str, List[PredictionError]]:
        """Group prediction errors by model type"""
        groups = {}
        
        for error in self.prediction_errors:
            if error.prediction_type not in groups:
                groups[error.prediction_type] = []
            groups[error.prediction_type].append(error)
        
        return groups
    
    def _retrain_model(self, model_type: str, errors: List[PredictionError]):
        """Retrain a specific model based on errors"""
        try:
            logger.info(f"🔄 Starting retraining for {model_type}")
            
            # Prepare training data from errors
            X, y = self._prepare_training_data(errors)
            
            if len(X) < 10:  # Minimum samples required
                logger.warning(f"Insufficient data for retraining {model_type}")
                return
            
            # Create or load existing model
            model = self._get_or_create_model(model_type)
            scaler = self._get_or_create_scaler(model_type)
            
            # Preprocess data
            X_scaled = scaler.fit_transform(X)
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # Train the model
            old_performance = self._evaluate_model(model, X_tensor, y_tensor)
            
            trained_model = self._train_pytorch_model(model, X_tensor, y_tensor)
            
            new_performance = self._evaluate_model(trained_model, X_tensor, y_tensor)
            
            # Check if performance improved
            if new_performance['mse'] < old_performance['mse']:
                logger.info(f"✅ Model {model_type} improved: MSE {old_performance['mse']:.4f} → {new_performance['mse']:.4f}")
                
                # Save the improved model
                self._save_model(model_type, trained_model, scaler)
                self.models[model_type] = trained_model
                self.scalers[model_type] = scaler
                
                # Update version
                self.model_versions[model_type] = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Record performance
                self.performance_history[model_type] = ModelPerformance(
                    model_name=model_type,
                    accuracy=new_performance.get('accuracy', 0),
                    mse=new_performance['mse'],
                    mae=new_performance.get('mae', 0),
                    r2_score=new_performance.get('r2', 0),
                    prediction_count=len(errors),
                    last_updated=datetime.now(),
                    version=self.model_versions[model_type]
                )
                
            else:
                logger.warning(f"⚠️ Model {model_type} did not improve: keeping old version")
                
        except Exception as e:
            logger.error(f"Error retraining model {model_type}: {e}")
    
    def _prepare_training_data(self, errors: List[PredictionError]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from prediction errors"""
        # This is a simplified implementation
        # In practice, you'd reconstruct the original features and targets
        
        X = []
        y = []
        
        for error in errors:
            # Use actual values as targets for retraining
            # Features would be reconstructed from the original prediction context
            # For now, using simplified feature representation
            
            features = [
                hash(error.symbol) % 1000,  # Symbol encoding
                error.timestamp.hour,       # Time feature
                error.timestamp.weekday(),  # Day of week
                len(error.features_used),   # Feature count
                error.predicted_value       # Previous prediction
            ]
            
            X.append(features)
            y.append(error.actual_value)
        
        return np.array(X), np.array(y)
    
    def _get_or_create_model(self, model_type: str) -> nn.Module:
        """Get existing model or create new one"""
        if model_type in self.models:
            return self.models[model_type]
        
        # Create model based on type
        if model_type in ['nav_prediction', 'price_prediction']:
            model = self._create_lstm_model()
        elif model_type in ['risk_analysis', 'volatility_prediction']:
            model = self._create_regression_model()
        elif model_type in ['sentiment_analysis', 'market_direction']:
            model = self._create_classification_model()
        else:
            model = self._create_default_model()
        
        model.to(self.device)
        self.models[model_type] = model
        
        return model
    
    def _get_or_create_scaler(self, model_type: str) -> StandardScaler:
        """Get existing scaler or create new one"""
        if model_type not in self.scalers:
            self.scalers[model_type] = StandardScaler()
        
        return self.scalers[model_type]
    
    def _create_lstm_model(self) -> nn.Module:
        """Create LSTM model for time series prediction"""
        class LSTMModel(nn.Module):
            def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                # Reshape for LSTM (batch_size, seq_len, features)
                if len(x.shape) == 2:
                    x = x.unsqueeze(1)  # Add sequence dimension
                
                lstm_out, _ = self.lstm(x)
                output = self.fc(self.dropout(lstm_out[:, -1, :]))
                return output
        
        return LSTMModel()
    
    def _create_regression_model(self) -> nn.Module:
        """Create regression model"""
        class RegressionModel(nn.Module):
            def __init__(self, input_size=5, hidden_size=128, output_size=1):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size // 2, output_size)
                )
            
            def forward(self, x):
                return self.network(x)
        
        return RegressionModel()
    
    def _create_classification_model(self) -> nn.Module:
        """Create classification model"""
        class ClassificationModel(nn.Module):
            def __init__(self, input_size=5, hidden_size=128, num_classes=3):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size // 2, num_classes),
                    nn.Softmax(dim=1)
                )
            
            def forward(self, x):
                return self.network(x)
        
        return ClassificationModel()
    
    def _create_default_model(self) -> nn.Module:
        """Create default neural network model"""
        return self._create_regression_model()
    
    def _train_pytorch_model(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> nn.Module:
        """Train PyTorch model"""
        model.train()
        
        # Loss function and optimizer
        if isinstance(model, type(self._create_classification_model())):
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=self.learning_config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.learning_config['epochs']):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X)
            if len(outputs.shape) > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze()
            
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Learning rate scheduling
            scheduler.step(loss)
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
                if patience_counter >= self.learning_config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        model.eval()
        return model
    
    def _evaluate_model(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance"""
        model.eval()
        
        with torch.no_grad():
            predictions = model(X)
            if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                predictions = predictions.squeeze()
            
            # Convert to numpy for sklearn metrics
            y_true = y.cpu().numpy()
            y_pred = predictions.cpu().numpy()
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            mae = np.mean(np.abs(y_true - y_pred))
            
            # R² score
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
    
    def _save_model(self, model_type: str, model: nn.Module, scaler: StandardScaler):
        """Save model and scaler to disk"""
        model_path = os.path.join(self.model_dir, f"{model_type}_model.pth")
        scaler_path = os.path.join(self.model_dir, f"{model_type}_scaler.pkl")
        
        # Save PyTorch model
        torch.save(model.state_dict(), model_path)
        
        # Save scaler
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"💾 Saved {model_type} model and scaler")
    
    def _load_model(self, model_type: str) -> Tuple[Optional[nn.Module], Optional[StandardScaler]]:
        """Load model and scaler from disk"""
        model_path = os.path.join(self.model_dir, f"{model_type}_model.pth")
        scaler_path = os.path.join(self.model_dir, f"{model_type}_scaler.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                # Load model
                model = self._get_or_create_model(model_type)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                
                # Load scaler
                scaler = joblib.load(scaler_path)
                
                logger.info(f"📂 Loaded {model_type} model and scaler")
                return model, scaler
                
            except Exception as e:
                logger.error(f"Error loading {model_type} model: {e}")
        
        return None, None
    
    async def _trigger_immediate_retraining(self, prediction_type: str):
        """Trigger immediate retraining for high-error predictions"""
        logger.warning(f"🚨 High error detected for {prediction_type}, triggering immediate retraining")
        
        # Get recent errors for this prediction type
        recent_errors = [
            error for error in self.prediction_errors
            if error.prediction_type == prediction_type and
            (datetime.now() - error.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        if len(recent_errors) >= 10:  # Minimum for immediate retraining
            self._retrain_model(prediction_type, recent_errors)
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get continuous learning statistics"""
        return {
            'total_errors_recorded': len(self.prediction_errors),
            'models_trained': len(self.models),
            'learning_active': self.learning_active,
            'performance_history': {
                model_type: asdict(performance) 
                for model_type, performance in self.performance_history.items()
            },
            'model_versions': self.model_versions,
            'learning_config': self.learning_config
        }
    
    def get_model_performance(self, model_type: str) -> Optional[ModelPerformance]:
        """Get performance metrics for a specific model"""
        return self.performance_history.get(model_type)
    
    async def predict_with_learning(self, model_type: str, features: np.ndarray) -> Tuple[float, str]:
        """Make prediction and return model version used"""
        model = self.models.get(model_type)
        scaler = self.scalers.get(model_type)
        
        if model is None or scaler is None:
            # Try to load from disk
            model, scaler = self._load_model(model_type)
            
            if model is None:
                # Create new model if none exists
                model = self._get_or_create_model(model_type)
                scaler = self._get_or_create_scaler(model_type)
        
        # Preprocess features
        features_scaled = scaler.transform(features.reshape(1, -1))
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            prediction = model(features_tensor)
            prediction_value = prediction.cpu().numpy()[0]
            
            if len(prediction_value.shape) > 0:
                prediction_value = prediction_value[0]
        
        model_version = self.model_versions.get(model_type, "initial")
        
        return float(prediction_value), model_version

# Example usage
async def main():
    asi = ContinuousLearningASI()
    
    try:
        # Start continuous learning
        await asi.start_continuous_learning()
        
        # Simulate some prediction errors
        await asi.record_prediction_error(
            prediction_id="pred_001",
            symbol="RELIANCE",
            prediction_type="nav_prediction",
            predicted_value=2500.0,
            actual_value=2520.0,
            model_version="v1.0",
            features_used=["price", "volume", "sentiment"]
        )
        
        # Get learning statistics
        stats = asi.get_learning_stats()
        print(f"Learning Stats: {stats}")
        
        # Make a prediction with learning
        features = np.array([2500, 1000, 0.5, 2, 2510])
        prediction, version = await asi.predict_with_learning("nav_prediction", features)
        print(f"Prediction: {prediction}, Model Version: {version}")
        
        # Keep running for a while to see continuous learning in action
        await asyncio.sleep(10)
        
    finally:
        asi.stop_continuous_learning()

if __name__ == "__main__":
    asyncio.run(main())
