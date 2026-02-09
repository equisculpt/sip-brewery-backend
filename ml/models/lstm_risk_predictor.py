"""
LSTM Risk Predictor with Attention Mechanism
Predicts portfolio risk metrics: VaR, CVaR, volatility, drawdown
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM outputs"""
    
    def __init__(self, hidden_dim: int):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
        
        Returns:
            context_vector: (batch, hidden_dim)
            attention_weights: (batch, seq_len)
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)  # (batch, seq_len)
        
        # Calculate context vector
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output  # (batch, seq_len, hidden_dim)
        ).squeeze(1)  # (batch, hidden_dim)
        
        return context_vector, attention_weights


class LSTMRiskPredictor(nn.Module):
    """
    LSTM with Attention for risk prediction
    
    Predicts:
    - 1-day, 1-week, 1-month VaR (Value at Risk)
    - CVaR (Conditional VaR)
    - Volatility
    - Maximum drawdown
    - Tail risk metrics
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super(LSTMRiskPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = AttentionLayer(lstm_output_dim)
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Prediction heads
        self.var_predictor = nn.Linear(hidden_dim // 2, 3)  # 1D, 1W, 1M VaR
        self.cvar_predictor = nn.Linear(hidden_dim // 2, 3)  # 1D, 1W, 1M CVaR
        self.volatility_predictor = nn.Linear(hidden_dim // 2, 3)  # 1D, 1W, 1M volatility
        self.drawdown_predictor = nn.Linear(hidden_dim // 2, 2)  # Current, max drawdown
        self.tail_risk_predictor = nn.Linear(hidden_dim // 2, 2)  # Skewness, kurtosis
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary with risk predictions
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * 2)
        
        # Apply attention
        context, attention_weights = self.attention(lstm_out)
        
        # Extract features
        features = self.feature_extractor(context)
        
        # Predictions
        var_pred = self.var_predictor(features)
        cvar_pred = self.cvar_predictor(features)
        volatility_pred = self.volatility_predictor(features)
        drawdown_pred = self.drawdown_predictor(features)
        tail_risk_pred = self.tail_risk_predictor(features)
        confidence = self.confidence_estimator(features)
        
        result = {
            'var': var_pred,  # (batch, 3) - 1D, 1W, 1M
            'cvar': cvar_pred,  # (batch, 3)
            'volatility': volatility_pred,  # (batch, 3)
            'drawdown': drawdown_pred,  # (batch, 2) - current, max
            'tail_risk': tail_risk_pred,  # (batch, 2) - skewness, kurtosis
            'confidence': confidence  # (batch, 1)
        }
        
        if return_attention:
            result['attention_weights'] = attention_weights
        
        return result


class RiskPredictionSystem:
    """
    Complete risk prediction system with training and inference
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        sequence_length: int = 60,  # 60 days of history
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.sequence_length = sequence_length
        
        self.model = LSTMRiskPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_model_state = None
    
    def prepare_sequences(
        self,
        data: pd.DataFrame,
        target_columns: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequences for LSTM
        
        Args:
            data: Time series data
            target_columns: Columns to predict
        
        Returns:
            X: Input sequences (n_samples, seq_len, n_features)
            y: Target values (n_samples, n_targets)
        """
        X_sequences = []
        y_targets = []
        
        feature_columns = [col for col in data.columns if col not in target_columns]
        
        for i in range(len(data) - self.sequence_length):
            # Input sequence
            X_seq = data[feature_columns].iloc[i:i + self.sequence_length].values
            X_sequences.append(X_seq)
            
            # Target (next time step)
            y_target = data[target_columns].iloc[i + self.sequence_length].values
            y_targets.append(y_target)
        
        X = torch.FloatTensor(np.array(X_sequences))
        y = torch.FloatTensor(np.array(y_targets))
        
        return X, y
    
    def calculate_risk_metrics(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate actual risk metrics from returns
        
        Args:
            returns: Portfolio returns
            confidence_level: Confidence level for VaR/CVaR
        
        Returns:
            Dictionary of risk metrics
        """
        # VaR (Value at Risk)
        var_1d = np.percentile(returns, (1 - confidence_level) * 100)
        var_1w = np.percentile(returns[-5:], (1 - confidence_level) * 100) if len(returns) >= 5 else var_1d
        var_1m = np.percentile(returns[-20:], (1 - confidence_level) * 100) if len(returns) >= 20 else var_1d
        
        # CVaR (Conditional VaR / Expected Shortfall)
        threshold = np.percentile(returns, (1 - confidence_level) * 100)
        cvar_1d = returns[returns <= threshold].mean() if len(returns[returns <= threshold]) > 0 else var_1d
        cvar_1w = returns[-5:][returns[-5:] <= threshold].mean() if len(returns) >= 5 else cvar_1d
        cvar_1m = returns[-20:][returns[-20:] <= threshold].mean() if len(returns) >= 20 else cvar_1d
        
        # Volatility
        vol_1d = np.std(returns)
        vol_1w = np.std(returns[-5:]) if len(returns) >= 5 else vol_1d
        vol_1m = np.std(returns[-20:]) if len(returns) >= 20 else vol_1d
        
        # Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        current_drawdown = drawdown[-1]
        max_drawdown = np.min(drawdown)
        
        # Tail risk
        from scipy import stats
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return {
            'var_1d': var_1d,
            'var_1w': var_1w,
            'var_1m': var_1m,
            'cvar_1d': cvar_1d,
            'cvar_1w': cvar_1w,
            'cvar_1m': cvar_1m,
            'vol_1d': vol_1d,
            'vol_1w': vol_1w,
            'vol_1m': vol_1m,
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def train_epoch(
        self,
        X_train: torch.Tensor,
        y_train: Dict[str, torch.Tensor]
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        
        X_train = X_train.to(self.device)
        for key in y_train:
            y_train[key] = y_train[key].to(self.device)
        
        # Forward pass
        predictions = self.model(X_train)
        
        # Calculate losses
        var_loss = F.mse_loss(predictions['var'], y_train['var'])
        cvar_loss = F.mse_loss(predictions['cvar'], y_train['cvar'])
        vol_loss = F.mse_loss(predictions['volatility'], y_train['volatility'])
        drawdown_loss = F.mse_loss(predictions['drawdown'], y_train['drawdown'])
        tail_loss = F.mse_loss(predictions['tail_risk'], y_train['tail_risk'])
        
        # Weighted combination
        total_loss = (
            var_loss * 0.3 +
            cvar_loss * 0.25 +
            vol_loss * 0.2 +
            drawdown_loss * 0.15 +
            tail_loss * 0.1
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def validate(
        self,
        X_val: torch.Tensor,
        y_val: Dict[str, torch.Tensor]
    ) -> Tuple[float, Dict]:
        """Validate model"""
        self.model.eval()
        
        X_val = X_val.to(self.device)
        for key in y_val:
            y_val[key] = y_val[key].to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_val)
            
            var_loss = F.mse_loss(predictions['var'], y_val['var'])
            cvar_loss = F.mse_loss(predictions['cvar'], y_val['cvar'])
            vol_loss = F.mse_loss(predictions['volatility'], y_val['volatility'])
            drawdown_loss = F.mse_loss(predictions['drawdown'], y_val['drawdown'])
            tail_loss = F.mse_loss(predictions['tail_risk'], y_val['tail_risk'])
            
            total_loss = (
                var_loss * 0.3 +
                cvar_loss * 0.25 +
                vol_loss * 0.2 +
                drawdown_loss * 0.15 +
                tail_loss * 0.1
            )
            
            # Calculate MAE for each metric
            metrics = {
                'var_mae': F.l1_loss(predictions['var'], y_val['var']).item(),
                'cvar_mae': F.l1_loss(predictions['cvar'], y_val['cvar']).item(),
                'vol_mae': F.l1_loss(predictions['volatility'], y_val['volatility']).item(),
                'drawdown_mae': F.l1_loss(predictions['drawdown'], y_val['drawdown']).item(),
                'avg_confidence': predictions['confidence'].mean().item()
            }
        
        return total_loss.item(), metrics
    
    def train(
        self,
        X_train: torch.Tensor,
        y_train: Dict[str, torch.Tensor],
        X_val: torch.Tensor,
        y_val: Dict[str, torch.Tensor],
        n_epochs: int = 100,
        early_stopping_patience: int = 20
    ) -> Dict:
        """Train the model"""
        logger.info(f"Starting LSTM risk predictor training for {n_epochs} epochs")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Train
            train_loss = self.train_epoch(X_train, y_train)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate(X_val, y_val)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{n_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"VaR MAE: {metrics['var_mae']:.4f} | "
                    f"Confidence: {metrics['avg_confidence']:.3f}"
                )
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        logger.info("Training completed")
        
        return {
            'best_val_loss': best_val_loss,
            'final_metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def predict(
        self,
        X: torch.Tensor,
        return_confidence: bool = True
    ) -> Dict:
        """
        Predict risk metrics
        
        Args:
            X: Input sequences (batch, seq_len, features)
            return_confidence: Whether to return confidence scores
        
        Returns:
            Risk predictions with confidence intervals
        """
        self.model.eval()
        
        X = X.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X, return_attention=True)
            
            result = {
                'var_1d': predictions['var'][:, 0].cpu().numpy(),
                'var_1w': predictions['var'][:, 1].cpu().numpy(),
                'var_1m': predictions['var'][:, 2].cpu().numpy(),
                'cvar_1d': predictions['cvar'][:, 0].cpu().numpy(),
                'cvar_1w': predictions['cvar'][:, 1].cpu().numpy(),
                'cvar_1m': predictions['cvar'][:, 2].cpu().numpy(),
                'volatility_1d': predictions['volatility'][:, 0].cpu().numpy(),
                'volatility_1w': predictions['volatility'][:, 1].cpu().numpy(),
                'volatility_1m': predictions['volatility'][:, 2].cpu().numpy(),
                'current_drawdown': predictions['drawdown'][:, 0].cpu().numpy(),
                'max_drawdown': predictions['drawdown'][:, 1].cpu().numpy(),
                'skewness': predictions['tail_risk'][:, 0].cpu().numpy(),
                'kurtosis': predictions['tail_risk'][:, 1].cpu().numpy()
            }
            
            if return_confidence:
                result['confidence'] = predictions['confidence'].cpu().numpy()
                result['attention_weights'] = predictions['attention_weights'].cpu().numpy()
        
        return result
    
    def save_model(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_model_state': self.best_model_state
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_model_state = checkpoint['best_model_state']
        logger.info(f"Model loaded from {path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic time series data
    n_samples = 1000
    n_features = 20
    seq_length = 60
    
    # Simulate market data
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add synthetic risk metrics as targets
    data['var_1d'] = np.random.uniform(-0.05, -0.01, n_samples)
    data['var_1w'] = np.random.uniform(-0.08, -0.02, n_samples)
    data['var_1m'] = np.random.uniform(-0.12, -0.03, n_samples)
    data['cvar_1d'] = data['var_1d'] * 1.2
    data['cvar_1w'] = data['var_1w'] * 1.2
    data['cvar_1m'] = data['var_1m'] * 1.2
    data['vol_1d'] = np.random.uniform(0.01, 0.03, n_samples)
    data['vol_1w'] = np.random.uniform(0.02, 0.05, n_samples)
    data['vol_1m'] = np.random.uniform(0.03, 0.08, n_samples)
    data['current_drawdown'] = np.random.uniform(-0.15, 0, n_samples)
    data['max_drawdown'] = np.random.uniform(-0.30, -0.05, n_samples)
    data['skewness'] = np.random.uniform(-1, 1, n_samples)
    data['kurtosis'] = np.random.uniform(0, 5, n_samples)
    
    # Create system
    system = RiskPredictionSystem(
        input_dim=n_features,
        hidden_dim=128,
        sequence_length=seq_length
    )
    
    # Prepare data
    target_cols = [
        'var_1d', 'var_1w', 'var_1m',
        'cvar_1d', 'cvar_1w', 'cvar_1m',
        'vol_1d', 'vol_1w', 'vol_1m',
        'current_drawdown', 'max_drawdown',
        'skewness', 'kurtosis'
    ]
    
    X, y_raw = system.prepare_sequences(data, target_cols)
    
    # Split targets
    y_targets = {
        'var': y_raw[:, :3],
        'cvar': y_raw[:, 3:6],
        'volatility': y_raw[:, 6:9],
        'drawdown': y_raw[:, 9:11],
        'tail_risk': y_raw[:, 11:13]
    }
    
    # Split train/val
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train = {k: v[:split_idx] for k, v in y_targets.items()}
    y_val = {k: v[split_idx:] for k, v in y_targets.items()}
    
    # Train
    metrics = system.train(X_train, y_train, X_val, y_val, n_epochs=50)
    
    print(f"\nTraining Results:")
    print(f"Best Val Loss: {metrics['best_val_loss']:.4f}")
    print(f"VaR MAE: {metrics['final_metrics']['var_mae']:.4f}")
    
    # Predict
    predictions = system.predict(X_val[:10])
    print(f"\nSample Predictions:")
    print(f"1-Day VaR: {predictions['var_1d'][:5]}")
    print(f"Confidence: {predictions['confidence'][:5].flatten()}")
    
    # Save model
    system.save_model('lstm_risk_model.pth')
