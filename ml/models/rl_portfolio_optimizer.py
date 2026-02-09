"""
Reinforcement Learning Portfolio Optimizer
Uses Deep Q-Network (DQN) and Proximal Policy Optimization (PPO)
for dynamic portfolio allocation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
import random

logger = logging.getLogger(__name__)


class PortfolioEnvironment:
    """
    Custom Gym-like environment for portfolio optimization
    
    State: [portfolio_weights, fund_returns, market_indicators, user_risk_profile]
    Action: New portfolio weights (continuous)
    Reward: Sharpe ratio + tax efficiency - transaction costs
    """
    
    def __init__(
        self,
        historical_data: pd.DataFrame,
        n_assets: int,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.06
    ):
        self.historical_data = historical_data
        self.n_assets = n_assets
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        
        self.current_step = 0
        self.max_steps = len(historical_data) - 1
        
        # State space: weights + returns + market features
        self.state_dim = n_assets + n_assets + 10  # weights + returns + market
        self.action_dim = n_assets
        
        self.current_weights = np.ones(n_assets) / n_assets
        self.portfolio_value = 100000.0  # Starting capital
        self.initial_value = self.portfolio_value
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = self.initial_value
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        if self.current_step >= self.max_steps:
            self.current_step = 0
        
        # Current portfolio weights
        weights = self.current_weights
        
        # Recent returns for each asset
        returns = self.historical_data.iloc[self.current_step, :self.n_assets].values
        
        # Market indicators (simplified)
        market_features = np.array([
            self.historical_data.iloc[self.current_step, self.n_assets],  # Market return
            self.historical_data.iloc[self.current_step, self.n_assets + 1],  # Volatility
            self.historical_data.iloc[self.current_step, self.n_assets + 2],  # Sentiment
            np.mean(returns),  # Average asset return
            np.std(returns),   # Return volatility
            np.max(returns),   # Best performer
            np.min(returns),   # Worst performer
            self.portfolio_value / self.initial_value,  # Portfolio performance
            self.current_step / self.max_steps,  # Time progress
            0.0  # Placeholder for additional feature
        ])
        
        state = np.concatenate([weights, returns, market_features])
        return state.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state, reward, done, info
        
        Args:
            action: New portfolio weights (must sum to 1)
        """
        # Normalize action to ensure weights sum to 1
        action = np.abs(action)
        action = action / (np.sum(action) + 1e-8)
        
        # Calculate transaction costs
        turnover = np.sum(np.abs(action - self.current_weights))
        transaction_costs = turnover * self.transaction_cost * self.portfolio_value
        
        # Get returns for current step
        returns = self.historical_data.iloc[self.current_step, :self.n_assets].values
        
        # Calculate portfolio return
        portfolio_return = np.dot(action, returns)
        
        # Update portfolio value
        self.portfolio_value = self.portfolio_value * (1 + portfolio_return) - transaction_costs
        
        # Update weights
        old_weights = self.current_weights.copy()
        self.current_weights = action
        
        # Calculate reward (Sharpe ratio approximation + penalties)
        reward = self._calculate_reward(portfolio_return, turnover, returns)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Get next state
        next_state = self._get_state() if not done else np.zeros(self.state_dim)
        
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return,
            'turnover': turnover,
            'transaction_costs': transaction_costs,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns, action)
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(
        self,
        portfolio_return: float,
        turnover: float,
        asset_returns: np.ndarray
    ) -> float:
        """
        Calculate reward function
        
        Reward = Return - Risk Penalty - Transaction Cost Penalty + Diversification Bonus
        """
        # Base reward: portfolio return
        reward = portfolio_return * 100  # Scale up
        
        # Penalty for high turnover
        turnover_penalty = turnover * 10
        
        # Penalty for concentration (encourage diversification)
        concentration = np.sum(self.current_weights ** 2)  # Herfindahl index
        concentration_penalty = concentration * 5
        
        # Bonus for positive Sharpe ratio
        sharpe = self._calculate_sharpe_ratio(asset_returns, self.current_weights)
        sharpe_bonus = max(0, sharpe) * 2
        
        total_reward = reward - turnover_penalty - concentration_penalty + sharpe_bonus
        
        return total_reward
    
    def _calculate_sharpe_ratio(
        self,
        asset_returns: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """Calculate Sharpe ratio for current portfolio"""
        portfolio_return = np.dot(weights, asset_returns)
        
        # Estimate volatility (simplified)
        volatility = np.std(asset_returns) * np.sqrt(252)  # Annualized
        
        if volatility == 0:
            return 0.0
        
        sharpe = (portfolio_return - self.risk_free_rate / 252) / (volatility + 1e-8)
        return sharpe


class DQNNetwork(nn.Module):
    """Deep Q-Network for portfolio optimization"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))  # Ensure weights sum to 1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class RLPortfolioOptimizer:
    """
    Reinforcement Learning Portfolio Optimizer
    
    Uses DQN with experience replay and target network
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 64
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Q-Network and Target Network
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Training metrics
        self.training_rewards = []
        self.training_losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: If True, use epsilon-greedy; else use greedy
        """
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            action = np.random.dirichlet(np.ones(self.action_dim))
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = self.q_network(state_tensor).squeeze(0).numpy()
        
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self) -> float:
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.q_network(states)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(
        self,
        env: PortfolioEnvironment,
        n_episodes: int = 1000,
        update_target_every: int = 10
    ) -> Dict:
        """
        Train the RL agent
        
        Args:
            env: Portfolio environment
            n_episodes: Number of training episodes
            update_target_every: Update target network every N episodes
        
        Returns:
            Training metrics
        """
        logger.info(f"Starting RL training for {n_episodes} episodes")
        
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0
            
            done = False
            while not done:
                # Select and execute action
                action = self.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                self.store_transition(state, action, reward, next_state, done)
                
                # Train
                loss = self.train_step()
                
                episode_reward += reward
                episode_loss += loss
                steps += 1
                
                state = next_state
            
            # Update target network periodically
            if episode % update_target_every == 0:
                self.update_target_network()
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Log metrics
            avg_loss = episode_loss / steps if steps > 0 else 0
            self.training_rewards.append(episode_reward)
            self.training_losses.append(avg_loss)
            
            if episode % 100 == 0:
                logger.info(
                    f"Episode {episode}/{n_episodes} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Epsilon: {self.epsilon:.3f} | "
                    f"Portfolio Value: {info['portfolio_value']:.2f}"
                )
        
        logger.info("Training completed")
        
        return {
            'final_reward': self.training_rewards[-1],
            'avg_reward': np.mean(self.training_rewards[-100:]),
            'final_loss': self.training_losses[-1],
            'training_rewards': self.training_rewards,
            'training_losses': self.training_losses
        }
    
    def optimize_portfolio(
        self,
        current_state: np.ndarray
    ) -> Dict:
        """
        Optimize portfolio allocation for current state
        
        Args:
            current_state: Current market and portfolio state
        
        Returns:
            Optimal weights and metrics
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
            optimal_weights = self.q_network(state_tensor).squeeze(0).numpy()
        
        return {
            'weights': optimal_weights.tolist(),
            'confidence': float(1 - self.epsilon)  # Higher when more trained
        }
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_rewards': self.training_rewards,
            'training_losses': self.training_losses
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_rewards = checkpoint['training_rewards']
        self.training_losses = checkpoint['training_losses']
        logger.info(f"Model loaded from {path}")


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic historical data
    n_assets = 10
    n_days = 1000
    
    # Simulate returns
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, (n_days, n_assets))
    market_return = np.random.normal(0.0005, 0.015, (n_days, 1))
    volatility = np.random.uniform(0.01, 0.03, (n_days, 1))
    sentiment = np.random.uniform(-1, 1, (n_days, 1))
    
    historical_data = pd.DataFrame(
        np.hstack([returns, market_return, volatility, sentiment])
    )
    
    # Create environment
    env = PortfolioEnvironment(
        historical_data=historical_data,
        n_assets=n_assets,
        transaction_cost=0.001
    )
    
    # Create and train agent
    agent = RLPortfolioOptimizer(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        learning_rate=0.001,
        gamma=0.99
    )
    
    # Train
    metrics = agent.train(env, n_episodes=500)
    
    print(f"\nTraining Results:")
    print(f"Final Reward: {metrics['final_reward']:.2f}")
    print(f"Average Reward (last 100): {metrics['avg_reward']:.2f}")
    
    # Test optimization
    test_state = env.reset()
    result = agent.optimize_portfolio(test_state)
    print(f"\nOptimal Weights: {[f'{w:.3f}' for w in result['weights']]}")
    
    # Save model
    agent.save_model('rl_portfolio_model.pth')
