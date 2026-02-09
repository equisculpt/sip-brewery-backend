"""
Portfolio Optimizer - Baseline Model
Uses Modern Portfolio Theory (MPT) with constraints
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Baseline portfolio optimization using Mean-Variance Optimization
    
    Features:
    - Maximize Sharpe ratio
    - Minimum variance portfolio
    - Risk-constrained optimization
    - Sector/category constraints
    - Tax-aware optimization
    """
    
    def __init__(self, risk_free_rate: float = 0.06):
        """
        Initialize optimizer
        
        Args:
            risk_free_rate: Annual risk-free rate (default 6% for India)
        """
        self.risk_free_rate = risk_free_rate
        self.min_weight = 0.0
        self.max_weight = 0.25  # Max 25% in single fund
        self.max_sector_weight = 0.35  # Max 35% in single sector
        
    def optimize_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        fund_metadata: pd.DataFrame,
        user_risk_profile: str = 'moderate',
        current_holdings: Dict[str, float] = None,
        constraints: Dict = None
    ) -> Dict:
        """
        Optimize portfolio allocation
        
        Args:
            expected_returns: Expected annual returns for each fund
            covariance_matrix: Covariance matrix of returns
            fund_metadata: DataFrame with fund info (category, expense_ratio, etc.)
            user_risk_profile: 'conservative', 'moderate', 'aggressive'
            current_holdings: Current portfolio weights (for tax optimization)
            constraints: Additional constraints
            
        Returns:
            Dictionary with optimal weights and metrics
        """
        n_assets = len(expected_returns)
        
        # Set risk tolerance based on profile
        risk_tolerance = {
            'conservative': 0.05,
            'moderate': 0.10,
            'aggressive': 0.15
        }.get(user_risk_profile, 0.10)
        
        # Initial guess - equal weight
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Constraints
        constraints_list = [
            # Weights sum to 1
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Add volatility constraint
        constraints_list.append({
            'type': 'ineq',
            'fun': lambda w: risk_tolerance - self._portfolio_volatility(w, covariance_matrix)
        })
        
        # Add sector constraints
        if 'category' in fund_metadata.columns:
            for category in fund_metadata['category'].unique():
                category_mask = (fund_metadata['category'] == category).values
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda w, mask=category_mask: self.max_sector_weight - np.sum(w[mask])
                })
        
        # Bounds for each weight
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))
        
        # Optimize for maximum Sharpe ratio
        result = minimize(
            fun=lambda w: -self._sharpe_ratio(w, expected_returns, covariance_matrix),
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_volatility = self._portfolio_volatility(optimal_weights, covariance_matrix)
        portfolio_sharpe = self._sharpe_ratio(optimal_weights, expected_returns, covariance_matrix)
        
        # Calculate turnover if current holdings provided
        turnover = 0.0
        if current_holdings is not None:
            current_weights = np.array([current_holdings.get(i, 0.0) for i in range(n_assets)])
            turnover = np.sum(np.abs(optimal_weights - current_weights))
        
        return {
            'weights': optimal_weights.tolist(),
            'expected_return': float(portfolio_return),
            'volatility': float(portfolio_volatility),
            'sharpe_ratio': float(portfolio_sharpe),
            'turnover': float(turnover),
            'optimization_success': result.success,
            'optimization_message': result.message
        }
    
    def optimize_minimum_variance(
        self,
        covariance_matrix: np.ndarray,
        fund_metadata: pd.DataFrame
    ) -> Dict:
        """
        Optimize for minimum variance portfolio
        """
        n_assets = covariance_matrix.shape[0]
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))
        
        result = minimize(
            fun=lambda w: self._portfolio_volatility(w, covariance_matrix),
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        optimal_weights = result.x
        portfolio_volatility = self._portfolio_volatility(optimal_weights, covariance_matrix)
        
        return {
            'weights': optimal_weights.tolist(),
            'volatility': float(portfolio_volatility),
            'optimization_success': result.success
        }
    
    def efficient_frontier(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        num_portfolios: int = 100
    ) -> pd.DataFrame:
        """
        Generate efficient frontier
        
        Returns:
            DataFrame with return, volatility, and weights for each portfolio
        """
        results = []
        
        # Target returns from min to max
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        n_assets = len(expected_returns)
        
        for target_return in target_returns:
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return}
            ]
            
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            result = minimize(
                fun=lambda w: self._portfolio_volatility(w, covariance_matrix),
                x0=initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
                volatility = self._portfolio_volatility(weights, covariance_matrix)
                sharpe = (target_return - self.risk_free_rate) / volatility
                
                results.append({
                    'return': target_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'weights': weights.tolist()
                })
        
        return pd.DataFrame(results)
    
    def _portfolio_return(self, weights: np.ndarray, expected_returns: np.ndarray) -> float:
        """Calculate portfolio expected return"""
        return np.dot(weights, expected_returns)
    
    def _portfolio_volatility(self, weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
        """Calculate portfolio volatility (standard deviation)"""
        return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    
    def _sharpe_ratio(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> float:
        """Calculate Sharpe ratio"""
        portfolio_return = self._portfolio_return(weights, expected_returns)
        portfolio_volatility = self._portfolio_volatility(weights, covariance_matrix)
        
        if portfolio_volatility == 0:
            return 0.0
        
        return (portfolio_return - self.risk_free_rate) / portfolio_volatility
    
    def rebalance_recommendations(
        self,
        current_weights: np.ndarray,
        optimal_weights: np.ndarray,
        fund_names: List[str],
        threshold: float = 0.05
    ) -> List[Dict]:
        """
        Generate rebalancing recommendations
        
        Args:
            current_weights: Current portfolio weights
            optimal_weights: Optimal portfolio weights
            fund_names: List of fund names
            threshold: Minimum deviation to trigger rebalancing (default 5%)
            
        Returns:
            List of rebalancing actions
        """
        recommendations = []
        
        for i, (current, optimal, name) in enumerate(zip(current_weights, optimal_weights, fund_names)):
            deviation = optimal - current
            
            if abs(deviation) > threshold:
                action = 'BUY' if deviation > 0 else 'SELL'
                recommendations.append({
                    'fund_name': name,
                    'action': action,
                    'current_weight': float(current),
                    'optimal_weight': float(optimal),
                    'deviation': float(deviation),
                    'priority': 'HIGH' if abs(deviation) > 0.10 else 'MEDIUM'
                })
        
        # Sort by absolute deviation
        recommendations.sort(key=lambda x: abs(x['deviation']), reverse=True)
        
        return recommendations


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Simulate 10 funds
    n_funds = 10
    expected_returns = np.random.uniform(0.08, 0.18, n_funds)  # 8-18% annual returns
    
    # Generate covariance matrix
    correlation = np.random.uniform(0.3, 0.7, (n_funds, n_funds))
    correlation = (correlation + correlation.T) / 2  # Make symmetric
    np.fill_diagonal(correlation, 1.0)
    
    volatilities = np.random.uniform(0.10, 0.25, n_funds)
    covariance_matrix = np.outer(volatilities, volatilities) * correlation
    
    # Fund metadata
    fund_metadata = pd.DataFrame({
        'fund_id': [f'FUND_{i}' for i in range(n_funds)],
        'category': np.random.choice(['Equity', 'Debt', 'Hybrid'], n_funds),
        'expense_ratio': np.random.uniform(0.005, 0.02, n_funds)
    })
    
    # Optimize
    optimizer = PortfolioOptimizer()
    result = optimizer.optimize_portfolio(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        fund_metadata=fund_metadata,
        user_risk_profile='moderate'
    )
    
    print("Optimization Result:")
    print(f"Expected Return: {result['expected_return']:.2%}")
    print(f"Volatility: {result['volatility']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"\nOptimal Weights:")
    for i, weight in enumerate(result['weights']):
        if weight > 0.01:  # Only show significant allocations
            print(f"  Fund {i}: {weight:.2%}")
