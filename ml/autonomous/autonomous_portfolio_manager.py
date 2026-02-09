"""
Autonomous Portfolio Manager using Reinforcement Learning
Fully autonomous decision-making for portfolio management with safety constraints
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class SafetyConstraints:
    """Safety constraints for autonomous decisions"""
    
    def __init__(self):
        self.max_single_trade_pct = 0.10  # Max 10% of portfolio in single trade
        self.max_daily_trades = 5
        self.max_turnover_daily = 0.20  # Max 20% daily turnover
        self.min_diversification = 5  # Min 5 holdings
        self.max_concentration = 0.25  # Max 25% in single holding
        self.max_sector_exposure = 0.35  # Max 35% in single sector
        self.stop_loss_threshold = -0.15  # -15% stop loss
        self.profit_target = 0.50  # 50% profit target
        
    def validate_action(self, action: Dict, current_state: Dict) -> Tuple[bool, str]:
        """
        Validate if action is safe to execute
        
        Returns:
            (is_valid, reason)
        """
        # Check single trade size
        if action.get('amount', 0) > current_state['portfolio_value'] * self.max_single_trade_pct:
            return False, f"Trade size exceeds {self.max_single_trade_pct*100}% limit"
        
        # Check daily trade count
        if current_state.get('trades_today', 0) >= self.max_daily_trades:
            return False, f"Daily trade limit ({self.max_daily_trades}) reached"
        
        # Check turnover
        if current_state.get('daily_turnover', 0) >= self.max_turnover_daily:
            return False, f"Daily turnover limit ({self.max_turnover_daily*100}%) reached"
        
        # Check diversification
        if action['type'] == 'SELL' and current_state.get('num_holdings', 0) <= self.min_diversification:
            return False, f"Cannot reduce holdings below {self.min_diversification}"
        
        # Check concentration after trade
        new_concentration = self._calculate_new_concentration(action, current_state)
        if new_concentration > self.max_concentration:
            return False, f"Would exceed concentration limit ({self.max_concentration*100}%)"
        
        return True, "Action is safe"
    
    def _calculate_new_concentration(self, action: Dict, state: Dict) -> float:
        """Calculate concentration after proposed action"""
        # Simplified calculation
        holdings = state.get('holdings', {})
        total_value = state.get('portfolio_value', 1)
        
        if action['type'] == 'BUY':
            fund_id = action['fund_id']
            current_value = holdings.get(fund_id, 0)
            new_value = current_value + action['amount']
            return new_value / (total_value + action['amount'])
        
        return 0.0


class AutonomousDecisionEngine(nn.Module):
    """
    Neural network for autonomous decision making
    
    Outputs:
    - Action type (BUY, SELL, HOLD, REBALANCE)
    - Fund selection
    - Amount/percentage
    - Confidence score
    """
    
    def __init__(self, state_dim: int, num_funds: int, hidden_dim: int = 256):
        super(AutonomousDecisionEngine, self).__init__()
        
        self.num_funds = num_funds
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Action type predictor
        self.action_type = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # BUY, SELL, HOLD, REBALANCE
        )
        
        # Fund selector
        self.fund_selector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_funds)
        )
        
        # Amount predictor
        self.amount_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output 0-1 (percentage)
        )
        
        # Confidence estimator
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        encoded = self.state_encoder(state)
        
        return {
            'action_type_logits': self.action_type(encoded),
            'action_type_probs': torch.softmax(self.action_type(encoded), dim=-1),
            'fund_logits': self.fund_selector(encoded),
            'fund_probs': torch.softmax(self.fund_selector(encoded), dim=-1),
            'amount': self.amount_predictor(encoded),
            'confidence': self.confidence(encoded)
        }


class AutonomousPortfolioManager:
    """
    Fully autonomous portfolio management system
    
    Features:
    - Autonomous decision making
    - Safety constraints
    - Risk management
    - Performance tracking
    - Audit logging
    """
    
    def __init__(
        self,
        state_dim: int = 50,
        num_funds: int = 100,
        confidence_threshold: float = 0.75,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        
        # Decision engine
        self.engine = AutonomousDecisionEngine(
            state_dim=state_dim,
            num_funds=num_funds
        ).to(self.device)
        
        # Safety constraints
        self.safety = SafetyConstraints()
        
        # Decision history
        self.decision_history = []
        self.performance_metrics = {
            'total_decisions': 0,
            'executed_decisions': 0,
            'rejected_decisions': 0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
        # Action mapping
        self.action_types = {
            0: 'BUY',
            1: 'SELL',
            2: 'HOLD',
            3: 'REBALANCE'
        }
    
    def make_decision(
        self,
        current_state: Dict,
        available_funds: List[str],
        user_preferences: Optional[Dict] = None
    ) -> Dict:
        """
        Make autonomous portfolio decision
        
        Args:
            current_state: Current portfolio and market state
            available_funds: List of fund IDs available for trading
            user_preferences: Optional user constraints
        
        Returns:
            Decision with action, reasoning, and confidence
        """
        logger.info("Making autonomous decision")
        
        # Prepare state tensor
        state_tensor = self._prepare_state(current_state)
        
        # Get decision from neural network
        with torch.no_grad():
            output = self.engine(state_tensor)
        
        # Extract decision components
        action_type_id = output['action_type_probs'].argmax(dim=-1).item()
        action_type = self.action_types[action_type_id]
        
        fund_id_idx = output['fund_probs'].argmax(dim=-1).item()
        fund_id = available_funds[fund_id_idx] if fund_id_idx < len(available_funds) else available_funds[0]
        
        amount_pct = output['amount'].item()
        confidence = output['confidence'].item()
        
        # Create proposed action
        proposed_action = {
            'type': action_type,
            'fund_id': fund_id,
            'amount': current_state['portfolio_value'] * amount_pct,
            'amount_pct': amount_pct,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        # Validate with safety constraints
        is_safe, safety_reason = self.safety.validate_action(proposed_action, current_state)
        
        # Check confidence threshold
        meets_confidence = confidence >= self.confidence_threshold
        
        # Make final decision
        should_execute = is_safe and meets_confidence
        
        decision = {
            'action': proposed_action,
            'should_execute': should_execute,
            'confidence': confidence,
            'safety_check': {
                'passed': is_safe,
                'reason': safety_reason
            },
            'confidence_check': {
                'passed': meets_confidence,
                'threshold': self.confidence_threshold
            },
            'reasoning': self._generate_reasoning(
                action_type, fund_id, amount_pct, confidence, current_state
            ),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log decision
        self._log_decision(decision)
        
        # Update metrics
        self.performance_metrics['total_decisions'] += 1
        if should_execute:
            self.performance_metrics['executed_decisions'] += 1
        else:
            self.performance_metrics['rejected_decisions'] += 1
        
        return decision
    
    def execute_decision(self, decision: Dict) -> Dict:
        """
        Execute approved decision
        
        Returns:
            Execution result
        """
        if not decision['should_execute']:
            return {
                'success': False,
                'reason': 'Decision not approved for execution'
            }
        
        action = decision['action']
        
        logger.info(f"Executing autonomous decision: {action['type']} {action['fund_id']}")
        
        # In production, this would call actual trading APIs
        # For now, return simulated execution
        
        execution_result = {
            'success': True,
            'action': action,
            'execution_time': datetime.now().isoformat(),
            'order_id': f"AUTO_{datetime.now().timestamp()}",
            'status': 'PENDING'
        }
        
        return execution_result
    
    def _prepare_state(self, state: Dict) -> torch.Tensor:
        """Convert state dict to tensor"""
        # Extract relevant features
        features = [
            state.get('portfolio_value', 0) / 1000000,  # Normalize
            state.get('total_return', 0),
            state.get('sharpe_ratio', 0),
            state.get('volatility', 0),
            state.get('num_holdings', 0) / 100,
            state.get('concentration_hhi', 0),
            state.get('equity_allocation', 0),
            state.get('debt_allocation', 0),
            state.get('gold_allocation', 0),
            state.get('market_sentiment', 0),
            # Add more features as needed
        ]
        
        # Pad to state_dim
        while len(features) < 50:
            features.append(0.0)
        
        return torch.FloatTensor(features[:50]).unsqueeze(0).to(self.device)
    
    def _generate_reasoning(
        self,
        action_type: str,
        fund_id: str,
        amount_pct: float,
        confidence: float,
        state: Dict
    ) -> str:
        """Generate human-readable reasoning for decision"""
        
        reasoning_parts = []
        
        # Action reasoning
        if action_type == 'BUY':
            reasoning_parts.append(f"Recommending BUY of {fund_id}")
            reasoning_parts.append(f"Amount: {amount_pct*100:.1f}% of portfolio")
        elif action_type == 'SELL':
            reasoning_parts.append(f"Recommending SELL of {fund_id}")
            reasoning_parts.append(f"Amount: {amount_pct*100:.1f}% of holding")
        elif action_type == 'REBALANCE':
            reasoning_parts.append("Recommending portfolio rebalancing")
        else:
            reasoning_parts.append("Recommending HOLD - no action needed")
        
        # Market context
        if state.get('market_sentiment') == 'BULLISH':
            reasoning_parts.append("Market sentiment is positive")
        elif state.get('market_sentiment') == 'BEARISH':
            reasoning_parts.append("Market sentiment is negative, exercising caution")
        
        # Portfolio context
        if state.get('concentration_hhi', 0) > 0.2:
            reasoning_parts.append("Portfolio concentration is high")
        
        if state.get('sharpe_ratio', 0) < 1.0:
            reasoning_parts.append("Risk-adjusted returns below target")
        
        # Confidence
        reasoning_parts.append(f"Decision confidence: {confidence*100:.1f}%")
        
        return ". ".join(reasoning_parts)
    
    def _log_decision(self, decision: Dict):
        """Log decision to history"""
        self.decision_history.append(decision)
        
        # Keep only last 1000 decisions in memory
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    def get_performance_report(self) -> Dict:
        """Get performance metrics"""
        return {
            **self.performance_metrics,
            'execution_rate': (
                self.performance_metrics['executed_decisions'] / 
                max(self.performance_metrics['total_decisions'], 1)
            ),
            'recent_decisions': self.decision_history[-10:]
        }
    
    def train(
        self,
        historical_data: pd.DataFrame,
        n_epochs: int = 100,
        learning_rate: float = 0.001
    ):
        """
        Train the autonomous decision engine
        
        Args:
            historical_data: Historical market and portfolio data
            n_epochs: Number of training epochs
            learning_rate: Learning rate
        """
        logger.info(f"Training autonomous decision engine for {n_epochs} epochs")
        
        optimizer = torch.optim.Adam(self.engine.parameters(), lr=learning_rate)
        
        for epoch in range(n_epochs):
            total_loss = 0
            
            # Training loop (simplified)
            # In production, this would use actual historical data
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{n_epochs}, Loss: {total_loss:.4f}")
        
        logger.info("Training completed")
    
    def save_model(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.engine.state_dict(),
            'performance_metrics': self.performance_metrics,
            'decision_history': self.decision_history[-100:]  # Save last 100
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.engine.load_state_dict(checkpoint['model_state_dict'])
        self.performance_metrics = checkpoint.get('performance_metrics', self.performance_metrics)
        logger.info(f"Model loaded from {path}")


class AutonomousPortfolioOrchestrator:
    """
    Orchestrates multiple autonomous managers for different strategies
    """
    
    def __init__(self):
        self.managers = {
            'conservative': AutonomousPortfolioManager(confidence_threshold=0.85),
            'moderate': AutonomousPortfolioManager(confidence_threshold=0.75),
            'aggressive': AutonomousPortfolioManager(confidence_threshold=0.65)
        }
        
    def get_decision(
        self,
        user_id: str,
        risk_profile: str,
        current_state: Dict,
        available_funds: List[str]
    ) -> Dict:
        """Get decision from appropriate manager based on risk profile"""
        
        manager = self.managers.get(risk_profile.lower(), self.managers['moderate'])
        
        decision = manager.make_decision(current_state, available_funds)
        
        decision['risk_profile'] = risk_profile
        decision['user_id'] = user_id
        
        return decision
    
    def get_all_performance_reports(self) -> Dict:
        """Get performance reports from all managers"""
        return {
            strategy: manager.get_performance_report()
            for strategy, manager in self.managers.items()
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    manager = AutonomousPortfolioManager()
    
    # Simulate current state
    current_state = {
        'portfolio_value': 500000,
        'total_return': 0.15,
        'sharpe_ratio': 1.2,
        'volatility': 0.12,
        'num_holdings': 8,
        'concentration_hhi': 0.18,
        'equity_allocation': 0.65,
        'debt_allocation': 0.30,
        'gold_allocation': 0.05,
        'market_sentiment': 'BULLISH',
        'trades_today': 2,
        'daily_turnover': 0.05,
        'holdings': {
            'FUND001': 100000,
            'FUND002': 80000,
            'FUND003': 70000
        }
    }
    
    available_funds = ['FUND001', 'FUND002', 'FUND003', 'FUND004', 'FUND005']
    
    # Make decision
    decision = manager.make_decision(current_state, available_funds)
    
    print("\n" + "="*60)
    print("AUTONOMOUS DECISION")
    print("="*60)
    print(f"Action: {decision['action']['type']}")
    print(f"Fund: {decision['action']['fund_id']}")
    print(f"Amount: ₹{decision['action']['amount']:,.0f} ({decision['action']['amount_pct']*100:.1f}%)")
    print(f"Confidence: {decision['confidence']*100:.1f}%")
    print(f"Should Execute: {decision['should_execute']}")
    print(f"\nReasoning: {decision['reasoning']}")
    print(f"\nSafety Check: {decision['safety_check']}")
    print(f"Confidence Check: {decision['confidence_check']}")
    
    # Get performance report
    report = manager.get_performance_report()
    print(f"\n" + "="*60)
    print("PERFORMANCE REPORT")
    print("="*60)
    print(f"Total Decisions: {report['total_decisions']}")
    print(f"Executed: {report['executed_decisions']}")
    print(f"Rejected: {report['rejected_decisions']}")
    print(f"Execution Rate: {report['execution_rate']*100:.1f}%")
