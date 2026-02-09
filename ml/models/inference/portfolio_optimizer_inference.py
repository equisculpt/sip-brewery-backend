#!/usr/bin/env python3
"""
Portfolio Optimizer Inference Script
Wrapper for RL Portfolio Optimizer model inference
"""

import sys
import json
import numpy as np
import torch
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rl_portfolio_optimizer import RLPortfolioOptimizer, PortfolioEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_state(input_data):
    """
    Prepare state vector from input data
    
    Expected input format:
    {
        "user_id": "USER123",
        "portfolio": {
            "holdings": [
                {"fund_id": "FUND001", "weight": 0.2, "return": 0.12},
                ...
            ]
        },
        "user_profile": {
            "risk_tolerance": "moderate",
            "age": 35,
            "investment_horizon": 120
        }
    }
    """
    portfolio = input_data.get('portfolio', {})
    holdings = portfolio.get('holdings', [])
    user_profile = input_data.get('user_profile', {})
    
    # Extract current weights
    weights = np.array([h.get('weight', 0.0) for h in holdings])
    if len(weights) < 10:
        weights = np.pad(weights, (0, 10 - len(weights)), constant_values=0.0)
    weights = weights[:10]
    
    # Extract returns
    returns = np.array([h.get('return', 0.0) for h in holdings])
    if len(returns) < 10:
        returns = np.pad(returns, (0, 10 - len(returns)), constant_values=0.0)
    returns = returns[:10]
    
    # Market indicators (simplified - in production, fetch real-time data)
    market_indicators = np.array([
        0.001,  # Market return
        0.015,  # Volatility
        0.5,    # Sentiment
        np.mean(returns),  # Average return
        np.std(returns),   # Return volatility
        np.max(returns),   # Best performer
        np.min(returns),   # Worst performer
        1.0,    # Portfolio performance
        0.5,    # Time progress
        0.0     # Placeholder
    ])
    
    # Concatenate all features
    state = np.concatenate([weights, returns, market_indicators])
    
    return state.astype(np.float32)


def main():
    try:
        # Read input from stdin
        input_json = sys.stdin.read()
        input_data = json.loads(input_json)
        
        logger.info(f"Processing portfolio optimization for user: {input_data.get('user_id')}")
        
        # Model path
        model_path = Path(__file__).parent.parent / 'trained' / 'rl_portfolio_model.pth'
        
        # Initialize model
        model = RLPortfolioOptimizer(
            state_dim=30,
            action_dim=10,
            learning_rate=0.001
        )
        
        # Load trained weights if available
        if model_path.exists():
            model.load_model(str(model_path))
            logger.info("Loaded trained model")
        else:
            logger.warning("No trained model found, using initialized model")
        
        # Prepare state
        state = prepare_state(input_data)
        
        # Optimize portfolio
        result = model.optimize_portfolio(state)
        
        # Add metadata
        result['model_version'] = '1.0.0'
        result['timestamp'] = str(np.datetime64('now'))
        
        # Calculate turnover if current weights provided
        current_weights = np.array([h.get('weight', 0.0) for h in input_data.get('portfolio', {}).get('holdings', [])])
        if len(current_weights) > 0:
            optimal_weights = np.array(result['weights'])
            if len(current_weights) < len(optimal_weights):
                current_weights = np.pad(current_weights, (0, len(optimal_weights) - len(current_weights)))
            result['turnover'] = float(np.sum(np.abs(optimal_weights - current_weights[:len(optimal_weights)])))
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {str(e)}")
        error_result = {
            'error': str(e),
            'weights': [0.1] * 10,  # Fallback: equal weights
            'confidence': 0.0
        }
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == '__main__':
    main()
