"""
Graph Neural Network Fund Predictor
Uses Graph Attention Networks (GAT) to predict fund performance
based on fund relationships, holdings, and market dynamics
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FundGraphBuilder:
    """
    Build graph representation of mutual funds
    
    Nodes: Funds, Stocks, Sectors, Managers, AMCs
    Edges: Holdings, Correlations, Management, Sector membership
    """
    
    def __init__(self):
        self.node_types = {
            'fund': 0,
            'stock': 1,
            'sector': 2,
            'manager': 3,
            'amc': 4
        }
        
    def build_graph(
        self,
        funds_data: pd.DataFrame,
        holdings_data: pd.DataFrame,
        correlation_matrix: np.ndarray,
        node_features: Dict[str, np.ndarray]
    ) -> Data:
        """
        Build PyTorch Geometric graph
        
        Args:
            funds_data: Fund metadata (returns, AUM, expense ratio, etc.)
            holdings_data: Fund holdings (fund_id, stock_id, weight)
            correlation_matrix: Fund return correlations
            node_features: Features for each node type
        
        Returns:
            PyTorch Geometric Data object
        """
        # Create node feature matrix
        all_features = []
        node_type_labels = []
        node_id_map = {}
        current_idx = 0
        
        # Add fund nodes
        for fund_id in funds_data['fund_id']:
            node_id_map[f'fund_{fund_id}'] = current_idx
            all_features.append(node_features['funds'][fund_id])
            node_type_labels.append(self.node_types['fund'])
            current_idx += 1
        
        # Add stock nodes
        unique_stocks = holdings_data['stock_id'].unique()
        for stock_id in unique_stocks:
            node_id_map[f'stock_{stock_id}'] = current_idx
            all_features.append(node_features['stocks'][stock_id])
            node_type_labels.append(self.node_types['stock'])
            current_idx += 1
        
        # Add sector nodes
        unique_sectors = funds_data['sector'].unique()
        for sector in unique_sectors:
            node_id_map[f'sector_{sector}'] = current_idx
            all_features.append(node_features['sectors'][sector])
            node_type_labels.append(self.node_types['sector'])
            current_idx += 1
        
        x = torch.FloatTensor(np.array(all_features))
        
        # Create edge index and edge attributes
        edge_index = []
        edge_attr = []
        
        # Holdings edges (fund -> stock)
        for _, row in holdings_data.iterrows():
            fund_idx = node_id_map[f"fund_{row['fund_id']}"]
            stock_idx = node_id_map[f"stock_{row['stock_id']}"]
            edge_index.append([fund_idx, stock_idx])
            edge_attr.append([row['weight'], 1.0, 0.0])  # [weight, edge_type_holdings, edge_type_correlation]
        
        # Correlation edges (fund <-> fund)
        n_funds = len(funds_data)
        for i in range(n_funds):
            for j in range(i + 1, n_funds):
                if abs(correlation_matrix[i, j]) > 0.5:  # Threshold
                    fund_i = node_id_map[f"fund_{funds_data.iloc[i]['fund_id']}"]
                    fund_j = node_id_map[f"fund_{funds_data.iloc[j]['fund_id']}"]
                    edge_index.append([fund_i, fund_j])
                    edge_index.append([fund_j, fund_i])  # Bidirectional
                    edge_attr.append([correlation_matrix[i, j], 0.0, 1.0])
                    edge_attr.append([correlation_matrix[i, j], 0.0, 1.0])
        
        # Sector membership edges (fund -> sector)
        for _, row in funds_data.iterrows():
            fund_idx = node_id_map[f"fund_{row['fund_id']}"]
            sector_idx = node_id_map[f"sector_{row['sector']}"]
            edge_index.append([fund_idx, sector_idx])
            edge_attr.append([1.0, 0.0, 0.0])
        
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        edge_attr = torch.FloatTensor(edge_attr)
        
        # Create graph data
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_type=torch.LongTensor(node_type_labels)
        )
        
        return data


class GATFundPredictor(nn.Module):
    """
    Graph Attention Network for fund performance prediction
    
    Architecture:
    - Multi-head GAT layers
    - Residual connections
    - Layer normalization
    - Prediction head for returns
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super(GATFundPredictor, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
                )
            self.layer_norms.append(nn.LayerNorm(hidden_channels))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Prediction heads
        self.return_predictor = nn.Linear(out_channels // 2, 4)  # 1M, 3M, 6M, 1Y returns
        self.risk_predictor = nn.Linear(out_channels // 2, 2)    # Volatility, max drawdown
        self.quality_predictor = nn.Linear(out_channels // 2, 3) # Sharpe, alpha, beta
        
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            data: PyTorch Geometric Data object
        
        Returns:
            Dictionary with predictions
        """
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GAT layers with residual connections
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_residual = x
            x = gat(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = norm(x + x_residual)  # Residual connection
        
        # Output projection
        x = self.output_proj(x)
        
        # Predictions
        return_pred = self.return_predictor(x)
        risk_pred = self.risk_predictor(x)
        quality_pred = self.quality_predictor(x)
        
        return {
            'returns': return_pred,      # [1M, 3M, 6M, 1Y]
            'risk': risk_pred,            # [volatility, max_drawdown]
            'quality': quality_pred       # [sharpe, alpha, beta]
        }


class FundPerformancePredictor:
    """
    Complete fund performance prediction system
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.model = GATFundPredictor(
            in_channels=in_channels,
            hidden_channels=hidden_channels
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.graph_builder = FundGraphBuilder()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(
        self,
        train_data: Data,
        targets: Dict[str, torch.Tensor]
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        
        # Forward pass
        predictions = self.model(train_data)
        
        # Calculate losses
        return_loss = F.mse_loss(predictions['returns'], targets['returns'])
        risk_loss = F.mse_loss(predictions['risk'], targets['risk'])
        quality_loss = F.mse_loss(predictions['quality'], targets['quality'])
        
        # Combined loss with weights
        total_loss = return_loss * 0.5 + risk_loss * 0.3 + quality_loss * 0.2
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def validate(
        self,
        val_data: Data,
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[float, Dict]:
        """Validate model"""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(val_data)
            
            return_loss = F.mse_loss(predictions['returns'], targets['returns'])
            risk_loss = F.mse_loss(predictions['risk'], targets['risk'])
            quality_loss = F.mse_loss(predictions['quality'], targets['quality'])
            
            total_loss = return_loss * 0.5 + risk_loss * 0.3 + quality_loss * 0.2
            
            # Calculate metrics
            metrics = {
                'return_mae': F.l1_loss(predictions['returns'], targets['returns']).item(),
                'risk_mae': F.l1_loss(predictions['risk'], targets['risk']).item(),
                'quality_mae': F.l1_loss(predictions['quality'], targets['quality']).item(),
                'total_loss': total_loss.item()
            }
        
        return total_loss.item(), metrics
    
    def train(
        self,
        train_data: Data,
        train_targets: Dict[str, torch.Tensor],
        val_data: Data,
        val_targets: Dict[str, torch.Tensor],
        n_epochs: int = 100,
        early_stopping_patience: int = 20
    ) -> Dict:
        """
        Train the model
        
        Returns:
            Training metrics
        """
        logger.info(f"Starting GNN training for {n_epochs} epochs")
        
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
                self.save_model('best_gnn_model.pth')
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
                    f"Return MAE: {metrics['return_mae']:.4f}"
                )
        
        logger.info("Training completed")
        
        return {
            'best_val_loss': best_val_loss,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def predict(
        self,
        graph_data: Data,
        fund_indices: List[int]
    ) -> Dict:
        """
        Predict fund performance
        
        Args:
            graph_data: Graph representation
            fund_indices: Indices of fund nodes to predict
        
        Returns:
            Predictions for specified funds
        """
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(graph_data)
            
            # Extract predictions for specified funds
            fund_predictions = {
                'returns_1m': predictions['returns'][fund_indices, 0].cpu().numpy(),
                'returns_3m': predictions['returns'][fund_indices, 1].cpu().numpy(),
                'returns_6m': predictions['returns'][fund_indices, 2].cpu().numpy(),
                'returns_1y': predictions['returns'][fund_indices, 3].cpu().numpy(),
                'volatility': predictions['risk'][fund_indices, 0].cpu().numpy(),
                'max_drawdown': predictions['risk'][fund_indices, 1].cpu().numpy(),
                'sharpe_ratio': predictions['quality'][fund_indices, 0].cpu().numpy(),
                'alpha': predictions['quality'][fund_indices, 1].cpu().numpy(),
                'beta': predictions['quality'][fund_indices, 2].cpu().numpy()
            }
        
        return fund_predictions
    
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
    n_funds = 50
    n_stocks = 200
    n_sectors = 10
    feature_dim = 32
    
    # Create synthetic fund data
    funds_data = pd.DataFrame({
        'fund_id': range(n_funds),
        'sector': np.random.choice(range(n_sectors), n_funds)
    })
    
    # Create synthetic holdings
    holdings_data = pd.DataFrame({
        'fund_id': np.repeat(range(n_funds), 10),
        'stock_id': np.random.choice(range(n_stocks), n_funds * 10),
        'weight': np.random.dirichlet(np.ones(10), n_funds).flatten()
    })
    
    # Create synthetic correlation matrix
    correlation_matrix = np.random.uniform(0.3, 0.9, (n_funds, n_funds))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Create synthetic node features
    node_features = {
        'funds': {i: np.random.randn(feature_dim) for i in range(n_funds)},
        'stocks': {i: np.random.randn(feature_dim) for i in range(n_stocks)},
        'sectors': {i: np.random.randn(feature_dim) for i in range(n_sectors)}
    }
    
    # Build graph
    graph_builder = FundGraphBuilder()
    graph_data = graph_builder.build_graph(
        funds_data, holdings_data, correlation_matrix, node_features
    )
    
    # Create synthetic targets
    targets = {
        'returns': torch.randn(graph_data.num_nodes, 4),
        'risk': torch.randn(graph_data.num_nodes, 2),
        'quality': torch.randn(graph_data.num_nodes, 3)
    }
    
    # Create and train predictor
    predictor = FundPerformancePredictor(in_channels=feature_dim)
    
    # Train (using same data for train and val for demo)
    metrics = predictor.train(
        train_data=graph_data,
        train_targets=targets,
        val_data=graph_data,
        val_targets=targets,
        n_epochs=50
    )
    
    print(f"\nTraining Results:")
    print(f"Best Val Loss: {metrics['best_val_loss']:.4f}")
    
    # Predict
    fund_indices = list(range(10))  # Predict first 10 funds
    predictions = predictor.predict(graph_data, fund_indices)
    
    print(f"\nPredictions for first 10 funds:")
    print(f"1Y Returns: {predictions['returns_1y']}")
    print(f"Sharpe Ratios: {predictions['sharpe_ratio']}")
