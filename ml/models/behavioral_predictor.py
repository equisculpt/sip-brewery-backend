"""
Behavioral Predictor using Fine-tuned FinBERT
Predicts user actions, churn probability, and behavioral biases
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class UserActionEncoder(nn.Module):
    """Encode user action sequences"""
    
    def __init__(self, num_actions: int, embedding_dim: int = 64):
        super(UserActionEncoder, self).__init__()
        self.action_embedding = nn.Embedding(num_actions, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, action_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_sequence: (batch, seq_len) - sequence of action IDs
        Returns:
            encoded: (batch, 256) - encoded representation
        """
        embedded = self.action_embedding(action_sequence)  # (batch, seq_len, 64)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Concatenate forward and backward hidden states
        encoded = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, 256)
        return encoded


class BehavioralPredictor(nn.Module):
    """
    BERT-based behavioral prediction model
    
    Predicts:
    - Next user action (BUY, SELL, HOLD, REBALANCE, WITHDRAW, etc.)
    - Churn probability
    - Investment amount
    - Behavioral biases (loss aversion, recency bias, overconfidence, etc.)
    """
    
    def __init__(
        self,
        bert_model_name: str = 'yiyanghkust/finbert-tone',
        num_actions: int = 10,
        num_biases: int = 8,
        dropout: float = 0.3
    ):
        super(BehavioralPredictor, self).__init__()
        
        # Load pre-trained FinBERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # Freeze BERT layers initially (will fine-tune later)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Unfreeze last 2 layers for fine-tuning
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        bert_hidden_size = self.bert.config.hidden_size  # 768
        
        # User action sequence encoder
        self.action_encoder = UserActionEncoder(num_actions)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(bert_hidden_size + 256 + 64, 512),  # BERT + actions + user features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512)
        )
        
        # Prediction heads
        self.action_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_actions)
        )
        
        self.churn_predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.amount_predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self.bias_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_biases),
            nn.Sigmoid()
        )
        
        # Attention for interpretability
        self.attention = nn.Linear(512, 1)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_sequence: torch.Tensor,
        user_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: (batch, seq_len) - tokenized text (chat history, notifications)
            attention_mask: (batch, seq_len) - attention mask
            action_sequence: (batch, action_seq_len) - past user actions
            user_features: (batch, feature_dim) - user demographics, portfolio stats
        
        Returns:
            Dictionary with predictions
        """
        # BERT encoding for text
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = bert_output.last_hidden_state[:, 0, :]  # [CLS] token (batch, 768)
        
        # Action sequence encoding
        action_features = self.action_encoder(action_sequence)  # (batch, 256)
        
        # Concatenate all features
        combined_features = torch.cat([
            text_features,
            action_features,
            user_features
        ], dim=1)
        
        # Feature fusion
        fused_features = self.feature_fusion(combined_features)  # (batch, 512)
        
        # Predictions
        next_action_logits = self.action_predictor(fused_features)
        churn_prob = self.churn_predictor(fused_features)
        predicted_amount = self.amount_predictor(fused_features)
        bias_scores = self.bias_predictor(fused_features)
        
        # Attention weights for interpretability
        attention_weights = F.softmax(self.attention(fused_features), dim=0)
        
        return {
            'next_action_logits': next_action_logits,  # (batch, num_actions)
            'next_action_probs': F.softmax(next_action_logits, dim=1),
            'churn_probability': churn_prob,  # (batch, 1)
            'predicted_amount': predicted_amount,  # (batch, 1)
            'bias_scores': bias_scores,  # (batch, num_biases)
            'attention_weights': attention_weights
        }


class BehavioralPredictionSystem:
    """
    Complete behavioral prediction system
    """
    
    def __init__(
        self,
        bert_model_name: str = 'yiyanghkust/finbert-tone',
        num_actions: int = 10,
        num_biases: int = 8,
        learning_rate: float = 2e-5,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.num_biases = num_biases
        
        self.model = BehavioralPredictor(
            bert_model_name=bert_model_name,
            num_actions=num_actions,
            num_biases=num_biases
        ).to(self.device)
        
        self.tokenizer = self.model.tokenizer
        
        # Different learning rates for BERT and other layers
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.bert.parameters(), 'lr': learning_rate},
            {'params': self.model.action_encoder.parameters(), 'lr': learning_rate * 10},
            {'params': self.model.feature_fusion.parameters(), 'lr': learning_rate * 10},
            {'params': self.model.action_predictor.parameters(), 'lr': learning_rate * 10},
            {'params': self.model.churn_predictor.parameters(), 'lr': learning_rate * 10},
            {'params': self.model.amount_predictor.parameters(), 'lr': learning_rate * 10},
            {'params': self.model.bias_predictor.parameters(), 'lr': learning_rate * 10}
        ])
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Action mapping
        self.action_map = {
            0: 'BUY',
            1: 'SELL',
            2: 'HOLD',
            3: 'REBALANCE',
            4: 'WITHDRAW',
            5: 'DEPOSIT',
            6: 'VIEW_PORTFOLIO',
            7: 'VIEW_INSIGHTS',
            8: 'CONTACT_SUPPORT',
            9: 'CHURN'
        }
        
        # Bias mapping
        self.bias_map = {
            0: 'loss_aversion',
            1: 'recency_bias',
            2: 'overconfidence',
            3: 'anchoring',
            4: 'herd_mentality',
            5: 'confirmation_bias',
            6: 'availability_bias',
            7: 'mental_accounting'
        }
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        
    def prepare_data(
        self,
        texts: List[str],
        action_sequences: List[List[int]],
        user_features: np.ndarray,
        max_text_length: int = 128,
        max_action_length: int = 20
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare data for model input
        
        Args:
            texts: List of text inputs (chat history, notifications)
            action_sequences: List of action sequences
            user_features: User feature matrix
        
        Returns:
            Dictionary of tensors
        """
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_text_length,
            return_tensors='pt'
        )
        
        # Pad action sequences
        padded_actions = []
        for seq in action_sequences:
            if len(seq) < max_action_length:
                seq = seq + [0] * (max_action_length - len(seq))
            else:
                seq = seq[-max_action_length:]
            padded_actions.append(seq)
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'action_sequence': torch.LongTensor(padded_actions),
            'user_features': torch.FloatTensor(user_features)
        }
    
    def train_epoch(
        self,
        train_data: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        
        # Move to device
        for key in train_data:
            train_data[key] = train_data[key].to(self.device)
        for key in targets:
            targets[key] = targets[key].to(self.device)
        
        # Forward pass
        predictions = self.model(
            input_ids=train_data['input_ids'],
            attention_mask=train_data['attention_mask'],
            action_sequence=train_data['action_sequence'],
            user_features=train_data['user_features']
        )
        
        # Calculate losses
        action_loss = F.cross_entropy(
            predictions['next_action_logits'],
            targets['next_action']
        )
        
        churn_loss = F.binary_cross_entropy(
            predictions['churn_probability'],
            targets['churn'].unsqueeze(1)
        )
        
        amount_loss = F.mse_loss(
            predictions['predicted_amount'],
            targets['amount'].unsqueeze(1)
        )
        
        bias_loss = F.binary_cross_entropy(
            predictions['bias_scores'],
            targets['biases']
        )
        
        # Combined loss with weights
        total_loss = (
            action_loss * 0.4 +
            churn_loss * 0.3 +
            amount_loss * 0.2 +
            bias_loss * 0.1
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def validate(
        self,
        val_data: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[float, Dict]:
        """Validate model"""
        self.model.eval()
        
        for key in val_data:
            val_data[key] = val_data[key].to(self.device)
        for key in targets:
            targets[key] = targets[key].to(self.device)
        
        with torch.no_grad():
            predictions = self.model(
                input_ids=val_data['input_ids'],
                attention_mask=val_data['attention_mask'],
                action_sequence=val_data['action_sequence'],
                user_features=val_data['user_features']
            )
            
            action_loss = F.cross_entropy(
                predictions['next_action_logits'],
                targets['next_action']
            )
            
            churn_loss = F.binary_cross_entropy(
                predictions['churn_probability'],
                targets['churn'].unsqueeze(1)
            )
            
            amount_loss = F.mse_loss(
                predictions['predicted_amount'],
                targets['amount'].unsqueeze(1)
            )
            
            bias_loss = F.binary_cross_entropy(
                predictions['bias_scores'],
                targets['biases']
            )
            
            total_loss = (
                action_loss * 0.4 +
                churn_loss * 0.3 +
                amount_loss * 0.2 +
                bias_loss * 0.1
            )
            
            # Calculate metrics
            action_preds = predictions['next_action_logits'].argmax(dim=1)
            action_accuracy = (action_preds == targets['next_action']).float().mean()
            
            churn_preds = (predictions['churn_probability'] > 0.5).float()
            churn_accuracy = (churn_preds == targets['churn'].unsqueeze(1)).float().mean()
            
            metrics = {
                'action_accuracy': action_accuracy.item(),
                'churn_accuracy': churn_accuracy.item(),
                'amount_mae': F.l1_loss(predictions['predicted_amount'], targets['amount'].unsqueeze(1)).item(),
                'bias_accuracy': ((predictions['bias_scores'] > 0.5) == targets['biases']).float().mean().item()
            }
        
        return total_loss.item(), metrics
    
    def train(
        self,
        train_data: Dict[str, torch.Tensor],
        train_targets: Dict[str, torch.Tensor],
        val_data: Dict[str, torch.Tensor],
        val_targets: Dict[str, torch.Tensor],
        n_epochs: int = 20,
        early_stopping_patience: int = 5
    ) -> Dict:
        """Train the model"""
        logger.info(f"Starting behavioral predictor training for {n_epochs} epochs")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Train
            train_loss = self.train_epoch(train_data, train_targets)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate(val_data, val_targets)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_behavioral_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            logger.info(
                f"Epoch {epoch}/{n_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Action Acc: {metrics['action_accuracy']:.3f} | "
                f"Churn Acc: {metrics['churn_accuracy']:.3f}"
            )
        
        logger.info("Training completed")
        
        return {
            'best_val_loss': best_val_loss,
            'final_metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def predict(
        self,
        texts: List[str],
        action_sequences: List[List[int]],
        user_features: np.ndarray
    ) -> Dict:
        """
        Predict user behavior
        
        Returns:
            Predictions with interpretations
        """
        self.model.eval()
        
        # Prepare data
        data = self.prepare_data(texts, action_sequences, user_features)
        
        for key in data:
            data[key] = data[key].to(self.device)
        
        with torch.no_grad():
            predictions = self.model(
                input_ids=data['input_ids'],
                attention_mask=data['attention_mask'],
                action_sequence=data['action_sequence'],
                user_features=data['user_features']
            )
            
            # Get top-k action predictions
            action_probs, action_indices = predictions['next_action_probs'].topk(3, dim=1)
            
            results = []
            for i in range(len(texts)):
                result = {
                    'top_actions': [
                        {
                            'action': self.action_map[action_indices[i, j].item()],
                            'probability': action_probs[i, j].item()
                        }
                        for j in range(3)
                    ],
                    'churn_probability': predictions['churn_probability'][i].item(),
                    'predicted_amount': predictions['predicted_amount'][i].item(),
                    'behavioral_biases': {
                        self.bias_map[j]: predictions['bias_scores'][i, j].item()
                        for j in range(self.num_biases)
                    },
                    'confidence': predictions['attention_weights'][i].item()
                }
                results.append(result)
        
        return results
    
    def save_model(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        logger.info(f"Model loaded from {path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Example usage with synthetic data
    n_samples = 100
    
    # Sample texts (chat history, notifications)
    texts = [
        "Market is down 5% today. Should I sell my holdings?",
        "My portfolio gained 10% this month. Time to invest more!",
        "I want to withdraw some money for emergency",
    ] * 34  # Repeat to get 100+ samples
    
    # Sample action sequences
    action_sequences = [
        [6, 7, 0, 2],  # VIEW_PORTFOLIO, VIEW_INSIGHTS, BUY, HOLD
        [6, 0, 0, 2],  # VIEW_PORTFOLIO, BUY, BUY, HOLD
        [6, 4, 8, 2],  # VIEW_PORTFOLIO, WITHDRAW, CONTACT_SUPPORT, HOLD
    ] * 34
    
    # Sample user features (age, portfolio_value, risk_score, etc.)
    user_features = np.random.randn(n_samples, 64)
    
    # Sample targets
    train_targets = {
        'next_action': torch.randint(0, 10, (n_samples,)),
        'churn': torch.randint(0, 2, (n_samples,)).float(),
        'amount': torch.randn(n_samples) * 10000 + 50000,
        'biases': torch.randint(0, 2, (n_samples, 8)).float()
    }
    
    # Create system
    system = BehavioralPredictionSystem(num_actions=10, num_biases=8)
    
    # Prepare data
    train_data = system.prepare_data(texts, action_sequences, user_features)
    
    # Train (using same data for train and val for demo)
    metrics = system.train(
        train_data, train_targets,
        train_data, train_targets,
        n_epochs=5
    )
    
    print(f"\nTraining Results:")
    print(f"Best Val Loss: {metrics['best_val_loss']:.4f}")
    print(f"Action Accuracy: {metrics['final_metrics']['action_accuracy']:.3f}")
    print(f"Churn Accuracy: {metrics['final_metrics']['churn_accuracy']:.3f}")
    
    # Predict
    predictions = system.predict(
        texts[:3],
        action_sequences[:3],
        user_features[:3]
    )
    
    print(f"\nSample Predictions:")
    for i, pred in enumerate(predictions):
        print(f"\nUser {i+1}:")
        print(f"  Top Action: {pred['top_actions'][0]['action']} ({pred['top_actions'][0]['probability']:.2%})")
        print(f"  Churn Risk: {pred['churn_probability']:.2%}")
        print(f"  Predicted Amount: ₹{pred['predicted_amount']:.0f}")
        print(f"  Top Bias: {max(pred['behavioral_biases'].items(), key=lambda x: x[1])}")
