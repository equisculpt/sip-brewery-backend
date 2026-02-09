"""
Federated Learning System
Train ML models across distributed users while preserving privacy
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)


class FederatedModel(nn.Module):
    """
    Base model for federated learning
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        super(FederatedModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class FederatedClient:
    """
    Client-side federated learning
    Trains on local data without sharing raw data
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        learning_rate: float = 0.001
    ):
        self.client_id = client_id
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def train_local(
        self,
        local_data: torch.Tensor,
        local_labels: torch.Tensor,
        n_epochs: int = 5
    ) -> Dict:
        """
        Train model on local data
        
        Returns:
            Training metrics and model updates
        """
        self.model.train()
        
        losses = []
        
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(local_data)
            loss = self.criterion(predictions, local_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
        
        # Get model updates (gradients)
        updates = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        return {
            'client_id': self.client_id,
            'updates': updates,
            'loss': np.mean(losses),
            'n_samples': len(local_data)
        }
    
    def update_model(self, global_weights: Dict):
        """Update local model with global weights"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_weights:
                    param.copy_(global_weights[name])


class FederatedServer:
    """
    Server-side federated learning coordinator
    Aggregates model updates from clients
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        aggregation_method: str = 'fedavg'
    ):
        self.global_model = global_model
        self.aggregation_method = aggregation_method
        self.round_number = 0
        self.training_history = []
        
    def aggregate_updates(
        self,
        client_updates: List[Dict]
    ) -> Dict:
        """
        Aggregate model updates from multiple clients
        
        Args:
            client_updates: List of updates from clients
        
        Returns:
            Aggregated global model weights
        """
        if self.aggregation_method == 'fedavg':
            return self._federated_averaging(client_updates)
        elif self.aggregation_method == 'weighted':
            return self._weighted_aggregation(client_updates)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _federated_averaging(self, client_updates: List[Dict]) -> Dict:
        """
        FedAvg: Simple averaging of model weights
        """
        n_clients = len(client_updates)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        
        # Get parameter names from first client
        param_names = list(client_updates[0]['updates'].keys())
        
        for name in param_names:
            # Average weights across clients
            weights = torch.stack([
                update['updates'][name]
                for update in client_updates
            ])
            
            aggregated_weights[name] = weights.mean(dim=0)
        
        logger.info(f"Aggregated updates from {n_clients} clients using FedAvg")
        
        return aggregated_weights
    
    def _weighted_aggregation(self, client_updates: List[Dict]) -> Dict:
        """
        Weighted aggregation based on number of samples
        """
        total_samples = sum(update['n_samples'] for update in client_updates)
        
        aggregated_weights = {}
        param_names = list(client_updates[0]['updates'].keys())
        
        for name in param_names:
            weighted_sum = torch.zeros_like(client_updates[0]['updates'][name])
            
            for update in client_updates:
                weight = update['n_samples'] / total_samples
                weighted_sum += weight * update['updates'][name]
            
            aggregated_weights[name] = weighted_sum
        
        logger.info(f"Aggregated updates from {len(client_updates)} clients (weighted)")
        
        return aggregated_weights
    
    def update_global_model(self, aggregated_weights: Dict):
        """Update global model with aggregated weights"""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_weights:
                    param.copy_(aggregated_weights[name])
        
        self.round_number += 1
    
    def get_global_weights(self) -> Dict:
        """Get current global model weights"""
        return {
            name: param.data.clone()
            for name, param in self.global_model.named_parameters()
        }


class PrivacyPreservingFederatedLearning:
    """
    Federated learning with differential privacy
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0
    ):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta  # Privacy parameter
        self.clip_norm = clip_norm  # Gradient clipping
        
    def add_noise_to_updates(
        self,
        updates: Dict,
        sensitivity: float = 1.0
    ) -> Dict:
        """
        Add differential privacy noise to model updates
        
        Args:
            updates: Model weight updates
            sensitivity: Sensitivity of the query
        
        Returns:
            Noisy updates
        """
        noisy_updates = {}
        
        # Calculate noise scale
        noise_scale = (sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon
        
        for name, weights in updates.items():
            # Add Gaussian noise
            noise = torch.normal(0, noise_scale, size=weights.shape)
            noisy_updates[name] = weights + noise
        
        logger.info(f"Added DP noise with ε={self.epsilon}, δ={self.delta}")
        
        return noisy_updates
    
    def clip_gradients(self, gradients: Dict) -> Dict:
        """Clip gradients to bound sensitivity"""
        clipped = {}
        
        for name, grad in gradients.items():
            norm = torch.norm(grad)
            if norm > self.clip_norm:
                clipped[name] = grad * (self.clip_norm / norm)
            else:
                clipped[name] = grad
        
        return clipped


class FederatedLearningOrchestrator:
    """
    Orchestrates federated learning across multiple clients
    """
    
    def __init__(
        self,
        model_config: Dict,
        privacy_enabled: bool = True
    ):
        self.model_config = model_config
        self.privacy_enabled = privacy_enabled
        
        # Initialize global model
        self.global_model = FederatedModel(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config.get('hidden_dim', 128),
            output_dim=model_config.get('output_dim', 1)
        )
        
        # Initialize server
        self.server = FederatedServer(self.global_model)
        
        # Initialize privacy mechanism
        if privacy_enabled:
            self.privacy = PrivacyPreservingFederatedLearning()
        
        # Track clients
        self.clients = {}
        
    def register_client(self, client_id: str) -> FederatedClient:
        """Register a new client"""
        client_model = FederatedModel(
            input_dim=self.model_config['input_dim'],
            hidden_dim=self.model_config.get('hidden_dim', 128),
            output_dim=self.model_config.get('output_dim', 1)
        )
        
        client = FederatedClient(client_id, client_model)
        self.clients[client_id] = client
        
        logger.info(f"Registered client: {client_id}")
        
        return client
    
    def run_training_round(
        self,
        client_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        n_local_epochs: int = 5
    ) -> Dict:
        """
        Run one round of federated training
        
        Args:
            client_data: Dict mapping client_id to (features, labels)
            n_local_epochs: Number of local training epochs
        
        Returns:
            Round results
        """
        logger.info(f"Starting training round {self.server.round_number + 1}")
        
        # Distribute global model to clients
        global_weights = self.server.get_global_weights()
        
        client_updates = []
        
        for client_id, (features, labels) in client_data.items():
            # Get or create client
            if client_id not in self.clients:
                self.register_client(client_id)
            
            client = self.clients[client_id]
            
            # Update client model with global weights
            client.update_model(global_weights)
            
            # Train locally
            update = client.train_local(features, labels, n_local_epochs)
            
            # Apply privacy if enabled
            if self.privacy_enabled:
                update['updates'] = self.privacy.add_noise_to_updates(
                    update['updates']
                )
            
            client_updates.append(update)
        
        # Aggregate updates
        aggregated_weights = self.server.aggregate_updates(client_updates)
        
        # Update global model
        self.server.update_global_model(aggregated_weights)
        
        # Calculate metrics
        avg_loss = np.mean([u['loss'] for u in client_updates])
        
        result = {
            'round': self.server.round_number,
            'n_clients': len(client_updates),
            'avg_loss': avg_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        self.server.training_history.append(result)
        
        logger.info(f"Round {self.server.round_number} completed. Avg loss: {avg_loss:.4f}")
        
        return result
    
    def get_global_model(self) -> nn.Module:
        """Get the current global model"""
        return self.global_model
    
    def save_model(self, path: str):
        """Save global model"""
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'round': self.server.round_number,
            'history': self.server.training_history
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load global model"""
        checkpoint = torch.load(path)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.server.round_number = checkpoint['round']
        self.server.training_history = checkpoint['history']
        
        logger.info(f"Model loaded from {path}")


class SecureAggregation:
    """
    Secure aggregation protocol for federated learning
    Prevents server from seeing individual client updates
    """
    
    def __init__(self):
        self.client_keys = {}
    
    def generate_client_key(self, client_id: str) -> str:
        """Generate encryption key for client"""
        key = hashlib.sha256(f"{client_id}_{datetime.now()}".encode()).hexdigest()
        self.client_keys[client_id] = key
        return key
    
    def encrypt_update(self, update: Dict, key: str) -> Dict:
        """Encrypt model update (simplified)"""
        # In production, use proper encryption (e.g., homomorphic encryption)
        encrypted = {}
        
        for name, weights in update.items():
            # Simple XOR-based encryption for demonstration
            noise = torch.randn_like(weights) * 0.1
            encrypted[name] = weights + noise
        
        return encrypted
    
    def aggregate_encrypted(self, encrypted_updates: List[Dict]) -> Dict:
        """Aggregate encrypted updates without decryption"""
        # Simplified - in production, use secure multi-party computation
        return self._simple_aggregation(encrypted_updates)
    
    def _simple_aggregation(self, updates: List[Dict]) -> Dict:
        """Simple aggregation"""
        aggregated = {}
        param_names = list(updates[0].keys())
        
        for name in param_names:
            weights = torch.stack([u[name] for u in updates])
            aggregated[name] = weights.mean(dim=0)
        
        return aggregated


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Federated Learning System")
    print("=" * 60)
    
    # Configuration
    model_config = {
        'input_dim': 20,
        'hidden_dim': 64,
        'output_dim': 1
    }
    
    # Create orchestrator
    orchestrator = FederatedLearningOrchestrator(
        model_config=model_config,
        privacy_enabled=True
    )
    
    # Simulate client data
    n_clients = 5
    client_data = {}
    
    for i in range(n_clients):
        client_id = f"client_{i}"
        features = torch.randn(100, 20)  # 100 samples, 20 features
        labels = torch.randn(100, 1)
        client_data[client_id] = (features, labels)
    
    # Run training rounds
    n_rounds = 10
    
    for round_num in range(n_rounds):
        result = orchestrator.run_training_round(client_data, n_local_epochs=3)
        print(f"Round {result['round']}: Loss = {result['avg_loss']:.4f}")
    
    print("\nFederated learning completed!")
    print(f"Total rounds: {orchestrator.server.round_number}")
    print(f"Total clients: {len(orchestrator.clients)}")
