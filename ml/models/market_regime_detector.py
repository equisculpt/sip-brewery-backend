"""
Market Regime Detector using Hidden Markov Model + Neural Network
Classifies market conditions: Bull, Bear, Sideways, Volatile
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from hmmlearn import hmm
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MarketRegimeHMM:
    """
    Hidden Markov Model for market regime detection
    
    States: Bull, Bear, Sideways, Volatile
    Observations: Returns, volatility, volume, sentiment
    """
    
    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='full',
            n_iter=100,
            random_state=42
        )
        
        self.regime_names = {
            0: 'BULL',
            1: 'BEAR',
            2: 'SIDEWAYS',
            3: 'VOLATILE'
        }
        
        self.is_fitted = False
    
    def prepare_observations(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        volume: np.ndarray,
        sentiment: np.ndarray
    ) -> np.ndarray:
        """
        Prepare observation matrix for HMM
        
        Args:
            returns: Daily returns
            volatility: Rolling volatility
            volume: Trading volume
            sentiment: Market sentiment score
        
        Returns:
            observations: (n_samples, n_features)
        """
        observations = np.column_stack([
            returns,
            volatility,
            volume,
            sentiment
        ])
        return observations
    
    def fit(self, observations: np.ndarray):
        """Fit HMM to observations"""
        self.model.fit(observations)
        self.is_fitted = True
        
        # Interpret states based on mean returns
        state_means = self.model.means_[:, 0]  # Returns column
        
        # Sort states by mean return
        sorted_indices = np.argsort(state_means)
        
        # Map states: lowest return = BEAR, highest = BULL
        self.state_mapping = {
            sorted_indices[0]: 1,  # BEAR
            sorted_indices[-1]: 0,  # BULL
        }
        
        # Middle states based on volatility
        remaining = [i for i in range(self.n_states) if i not in [sorted_indices[0], sorted_indices[-1]]]
        state_vols = self.model.covars_[remaining, 0, 0]  # Variance of returns
        
        if len(remaining) == 2:
            if state_vols[0] > state_vols[1]:
                self.state_mapping[remaining[0]] = 3  # VOLATILE
                self.state_mapping[remaining[1]] = 2  # SIDEWAYS
            else:
                self.state_mapping[remaining[0]] = 2  # SIDEWAYS
                self.state_mapping[remaining[1]] = 3  # VOLATILE
        
        logger.info("HMM fitted successfully")
        logger.info(f"State means: {state_means}")
        logger.info(f"State mapping: {self.state_mapping}")
    
    def predict(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regime and probabilities
        
        Returns:
            regimes: Predicted regime for each time step
            probabilities: Probability distribution over regimes
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Predict hidden states
        hidden_states = self.model.predict(observations)
        
        # Map to regime names
        regimes = np.array([self.state_mapping.get(s, s) for s in hidden_states])
        
        # Get state probabilities
        probabilities = self.model.predict_proba(observations)
        
        # Reorder probabilities according to mapping
        reordered_probs = np.zeros_like(probabilities)
        for original_state, mapped_state in self.state_mapping.items():
            reordered_probs[:, mapped_state] = probabilities[:, original_state]
        
        return regimes, reordered_probs
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get regime transition probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Reorder transition matrix according to mapping
        n = self.n_states
        reordered = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                orig_i = [k for k, v in self.state_mapping.items() if v == i][0]
                orig_j = [k for k, v in self.state_mapping.items() if v == j][0]
                reordered[i, j] = self.model.transmat_[orig_i, orig_j]
        
        return reordered


class RegimeNeuralNetwork(nn.Module):
    """
    Neural Network for regime classification
    Complements HMM with learned features
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_regimes: int = 4,
        dropout: float = 0.3
    ):
        super(RegimeNeuralNetwork, self).__init__()
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout)
        )
        
        # Regime classifier
        self.regime_classifier = nn.Linear(hidden_dim // 2, num_regimes)
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Duration predictor (how long regime will last)
        self.duration_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input features (batch, input_dim)
        
        Returns:
            Dictionary with predictions
        """
        features = self.feature_extractor(x)
        
        regime_logits = self.regime_classifier(features)
        regime_probs = F.softmax(regime_logits, dim=1)
        
        confidence = self.confidence_estimator(features)
        duration = self.duration_predictor(features)
        
        return {
            'regime_logits': regime_logits,
            'regime_probs': regime_probs,
            'confidence': confidence,
            'expected_duration': duration
        }


class MarketRegimeDetector:
    """
    Hybrid market regime detector combining HMM and Neural Network
    """
    
    def __init__(
        self,
        input_dim: int,
        n_states: int = 4,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.n_states = n_states
        
        # HMM component
        self.hmm_model = MarketRegimeHMM(n_states=n_states)
        
        # Neural network component
        self.nn_model = RegimeNeuralNetwork(
            input_dim=input_dim,
            num_regimes=n_states
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.regime_names = {
            0: 'BULL',
            1: 'BEAR',
            2: 'SIDEWAYS',
            3: 'VOLATILE'
        }
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def fit_hmm(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        volume: np.ndarray,
        sentiment: np.ndarray
    ):
        """Fit HMM component"""
        observations = self.hmm_model.prepare_observations(
            returns, volatility, volume, sentiment
        )
        self.hmm_model.fit(observations)
        logger.info("HMM component fitted")
    
    def train_epoch(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor
    ) -> Tuple[float, float]:
        """Train neural network for one epoch"""
        self.nn_model.train()
        
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        
        # Forward pass
        predictions = self.nn_model(X_train)
        
        # Classification loss
        regime_loss = F.cross_entropy(predictions['regime_logits'], y_train)
        
        # Total loss
        total_loss = regime_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), 1.0)
        self.optimizer.step()
        
        # Calculate accuracy
        regime_preds = predictions['regime_logits'].argmax(dim=1)
        accuracy = (regime_preds == y_train).float().mean().item()
        
        return total_loss.item(), accuracy
    
    def validate(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor
    ) -> Tuple[float, float, Dict]:
        """Validate neural network"""
        self.nn_model.eval()
        
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        
        with torch.no_grad():
            predictions = self.nn_model(X_val)
            
            regime_loss = F.cross_entropy(predictions['regime_logits'], y_val)
            
            regime_preds = predictions['regime_logits'].argmax(dim=1)
            accuracy = (regime_preds == y_val).float().mean().item()
            
            # Per-class accuracy
            per_class_acc = {}
            for regime in range(self.n_states):
                mask = y_val == regime
                if mask.sum() > 0:
                    per_class_acc[self.regime_names[regime]] = (
                        regime_preds[mask] == y_val[mask]
                    ).float().mean().item()
            
            metrics = {
                'accuracy': accuracy,
                'per_class_accuracy': per_class_acc,
                'avg_confidence': predictions['confidence'].mean().item()
            }
        
        return regime_loss.item(), accuracy, metrics
    
    def train_nn(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        n_epochs: int = 100,
        early_stopping_patience: int = 20
    ) -> Dict:
        """Train neural network component"""
        logger.info(f"Starting NN training for {n_epochs} epochs")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(X_train, y_train)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc, metrics = self.validate(X_val, y_val)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_regime_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{n_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.3f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.3f}"
                )
        
        logger.info("NN training completed")
        
        return {
            'best_val_loss': best_val_loss,
            'final_metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def predict(
        self,
        features: np.ndarray,
        use_ensemble: bool = True
    ) -> Dict:
        """
        Predict market regime
        
        Args:
            features: Input features
            use_ensemble: Whether to ensemble HMM and NN predictions
        
        Returns:
            Regime predictions with probabilities
        """
        self.nn_model.eval()
        
        # Neural network prediction
        X = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            nn_predictions = self.nn_model(X)
            nn_probs = nn_predictions['regime_probs'].cpu().numpy()
            nn_regimes = nn_predictions['regime_logits'].argmax(dim=1).cpu().numpy()
            confidence = nn_predictions['confidence'].cpu().numpy()
            duration = nn_predictions['expected_duration'].cpu().numpy()
        
        if use_ensemble and self.hmm_model.is_fitted:
            # HMM prediction (requires time series observations)
            # For now, use NN predictions as primary
            # In production, combine with HMM for better accuracy
            ensemble_probs = nn_probs  # Can be weighted average of HMM and NN
            ensemble_regimes = nn_regimes
        else:
            ensemble_probs = nn_probs
            ensemble_regimes = nn_regimes
        
        results = []
        for i in range(len(features)):
            result = {
                'regime': self.regime_names[ensemble_regimes[i]],
                'regime_id': int(ensemble_regimes[i]),
                'probabilities': {
                    self.regime_names[j]: float(ensemble_probs[i, j])
                    for j in range(self.n_states)
                },
                'confidence': float(confidence[i]),
                'expected_duration_days': float(duration[i]),
                'transition_probabilities': None  # Will be filled if HMM is used
            }
            results.append(result)
        
        # Add transition probabilities if HMM is fitted
        if self.hmm_model.is_fitted:
            transition_matrix = self.hmm_model.get_transition_matrix()
            for i, result in enumerate(results):
                current_regime = result['regime_id']
                result['transition_probabilities'] = {
                    self.regime_names[j]: float(transition_matrix[current_regime, j])
                    for j in range(self.n_states)
                }
        
        return results
    
    def get_regime_characteristics(self) -> Dict:
        """Get characteristics of each regime from HMM"""
        if not self.hmm_model.is_fitted:
            return {}
        
        characteristics = {}
        for regime_id, regime_name in self.regime_names.items():
            # Find original HMM state
            original_state = [k for k, v in self.hmm_model.state_mapping.items() if v == regime_id][0]
            
            mean_return = self.hmm_model.model.means_[original_state, 0]
            volatility = np.sqrt(self.hmm_model.model.covars_[original_state, 0, 0])
            
            characteristics[regime_name] = {
                'mean_return': float(mean_return),
                'volatility': float(volatility),
                'description': self._get_regime_description(regime_name, mean_return, volatility)
            }
        
        return characteristics
    
    def _get_regime_description(self, regime: str, mean_return: float, volatility: float) -> str:
        """Generate human-readable regime description"""
        descriptions = {
            'BULL': f"Strong upward trend with {mean_return*100:.2f}% avg daily return and {volatility*100:.2f}% volatility",
            'BEAR': f"Downward trend with {mean_return*100:.2f}% avg daily return and {volatility*100:.2f}% volatility",
            'SIDEWAYS': f"Range-bound market with {mean_return*100:.2f}% avg daily return and {volatility*100:.2f}% volatility",
            'VOLATILE': f"High uncertainty with {mean_return*100:.2f}% avg daily return and {volatility*100:.2f}% volatility"
        }
        return descriptions.get(regime, "Unknown regime")
    
    def save_model(self, path: str):
        """Save neural network model"""
        torch.save({
            'nn_model_state_dict': self.nn_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load neural network model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.nn_model.load_state_dict(checkpoint['nn_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']
        logger.info(f"Model loaded from {path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic market data
    n_samples = 1000
    n_features = 20
    
    np.random.seed(42)
    
    # Simulate different market regimes
    regime_lengths = [250, 250, 250, 250]  # 250 days each
    true_regimes = []
    features_list = []
    
    for regime_id, length in enumerate(regime_lengths):
        if regime_id == 0:  # BULL
            returns = np.random.normal(0.001, 0.01, length)
            volatility = np.random.uniform(0.008, 0.012, length)
        elif regime_id == 1:  # BEAR
            returns = np.random.normal(-0.001, 0.015, length)
            volatility = np.random.uniform(0.012, 0.020, length)
        elif regime_id == 2:  # SIDEWAYS
            returns = np.random.normal(0.0, 0.008, length)
            volatility = np.random.uniform(0.006, 0.010, length)
        else:  # VOLATILE
            returns = np.random.normal(0.0, 0.025, length)
            volatility = np.random.uniform(0.020, 0.035, length)
        
        volume = np.random.uniform(0.8, 1.2, length)
        sentiment = np.random.uniform(-1, 1, length)
        
        # Create features
        regime_features = np.column_stack([
            returns,
            volatility,
            volume,
            sentiment,
            np.random.randn(length, n_features - 4)
        ])
        
        features_list.append(regime_features)
        true_regimes.extend([regime_id] * length)
    
    features = np.vstack(features_list)
    true_regimes = np.array(true_regimes)
    
    # Create detector
    detector = MarketRegimeDetector(input_dim=n_features)
    
    # Fit HMM
    detector.fit_hmm(
        returns=features[:, 0],
        volatility=features[:, 1],
        volume=features[:, 2],
        sentiment=features[:, 3]
    )
    
    # Prepare data for NN
    X = torch.FloatTensor(features)
    y = torch.LongTensor(true_regimes)
    
    # Split train/val
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train NN
    metrics = detector.train_nn(X_train, y_train, X_val, y_val, n_epochs=50)
    
    print(f"\nTraining Results:")
    print(f"Best Val Loss: {metrics['best_val_loss']:.4f}")
    print(f"Final Accuracy: {metrics['final_metrics']['accuracy']:.3f}")
    print(f"Per-class Accuracy:")
    for regime, acc in metrics['final_metrics']['per_class_accuracy'].items():
        print(f"  {regime}: {acc:.3f}")
    
    # Predict
    predictions = detector.predict(features[-10:].numpy())
    
    print(f"\nSample Predictions:")
    for i, pred in enumerate(predictions[:3]):
        print(f"\nTime {i+1}:")
        print(f"  Regime: {pred['regime']} (confidence: {pred['confidence']:.2%})")
        print(f"  Expected Duration: {pred['expected_duration_days']:.0f} days")
        print(f"  Probabilities: {pred['probabilities']}")
    
    # Get regime characteristics
    characteristics = detector.get_regime_characteristics()
    print(f"\nRegime Characteristics:")
    for regime, chars in characteristics.items():
        print(f"\n{regime}:")
        print(f"  {chars['description']}")
