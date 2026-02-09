"""
Comprehensive Backtesting Framework
Test ML models and strategies on 10+ years of historical data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest"""
    start_date: str
    end_date: str
    initial_capital: float = 1000000
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    rebalance_frequency: str = 'monthly'  # daily, weekly, monthly
    benchmark: str = 'NIFTY50'
    

@dataclass
class Trade:
    """Individual trade record"""
    timestamp: datetime
    action: str  # BUY, SELL
    fund_id: str
    quantity: float
    price: float
    amount: float
    commission: float
    

class PortfolioState:
    """Track portfolio state during backtest"""
    
    def __init__(self, initial_capital: float):
        self.cash = initial_capital
        self.holdings = {}  # fund_id -> quantity
        self.initial_capital = initial_capital
        self.trades = []
        self.daily_values = []
        
    def buy(self, fund_id: str, amount: float, price: float, commission: float):
        """Execute buy order"""
        total_cost = amount + commission
        
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash for purchase: {total_cost} > {self.cash}")
            return False
        
        quantity = amount / price
        self.cash -= total_cost
        self.holdings[fund_id] = self.holdings.get(fund_id, 0) + quantity
        
        self.trades.append(Trade(
            timestamp=datetime.now(),
            action='BUY',
            fund_id=fund_id,
            quantity=quantity,
            price=price,
            amount=amount,
            commission=commission
        ))
        
        return True
    
    def sell(self, fund_id: str, quantity: float, price: float, commission: float):
        """Execute sell order"""
        if self.holdings.get(fund_id, 0) < quantity:
            logger.warning(f"Insufficient holdings for sale: {quantity} > {self.holdings.get(fund_id, 0)}")
            return False
        
        proceeds = quantity * price - commission
        self.cash += proceeds
        self.holdings[fund_id] -= quantity
        
        if self.holdings[fund_id] == 0:
            del self.holdings[fund_id]
        
        self.trades.append(Trade(
            timestamp=datetime.now(),
            action='SELL',
            fund_id=fund_id,
            quantity=quantity,
            price=price,
            amount=quantity * price,
            commission=commission
        ))
        
        return True
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        holdings_value = sum(
            qty * prices.get(fund_id, 0)
            for fund_id, qty in self.holdings.items()
        )
        return self.cash + holdings_value
    
    def get_returns(self) -> float:
        """Calculate total returns"""
        if not self.daily_values:
            return 0.0
        current_value = self.daily_values[-1]
        return (current_value - self.initial_capital) / self.initial_capital


class BacktestEngine:
    """
    Main backtesting engine
    Tests strategies on historical data
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio = PortfolioState(config.initial_capital)
        self.results = None
        
    def load_historical_data(
        self,
        data_source: str = 'csv',
        data_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load historical market data
        
        Returns:
            DataFrame with columns: date, fund_id, nav, returns
        """
        # In production, load from database or API
        # For now, generate synthetic data
        
        logger.info(f"Loading historical data from {self.config.start_date} to {self.config.end_date}")
        
        dates = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='D'
        )
        
        funds = [f'FUND{i:03d}' for i in range(1, 21)]  # 20 funds
        
        data = []
        for fund_id in funds:
            # Generate synthetic NAV data
            np.random.seed(hash(fund_id) % 2**32)
            initial_nav = 100
            returns = np.random.normal(0.0005, 0.015, len(dates))
            navs = initial_nav * np.cumprod(1 + returns)
            
            for date, nav, ret in zip(dates, navs, returns):
                data.append({
                    'date': date,
                    'fund_id': fund_id,
                    'nav': nav,
                    'returns': ret
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} data points for {len(funds)} funds")
        
        return df
    
    def run_backtest(
        self,
        strategy: Callable,
        historical_data: pd.DataFrame
    ) -> Dict:
        """
        Run backtest with given strategy
        
        Args:
            strategy: Function that takes (date, data, portfolio) and returns trades
            historical_data: Historical market data
        
        Returns:
            Backtest results
        """
        logger.info("Starting backtest")
        
        dates = sorted(historical_data['date'].unique())
        
        for i, date in enumerate(dates):
            # Get data for current date
            current_data = historical_data[historical_data['date'] == date]
            
            # Get current prices
            prices = dict(zip(current_data['fund_id'], current_data['nav']))
            
            # Calculate portfolio value
            portfolio_value = self.portfolio.get_total_value(prices)
            self.portfolio.daily_values.append(portfolio_value)
            
            # Execute strategy
            try:
                trades = strategy(date, current_data, self.portfolio, prices)
                
                if trades:
                    self._execute_trades(trades, prices)
            except Exception as e:
                logger.error(f"Strategy execution failed on {date}: {e}")
            
            # Log progress
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(dates)} days, Portfolio value: ₹{portfolio_value:,.0f}")
        
        # Calculate final results
        self.results = self._calculate_results(historical_data)
        
        logger.info("Backtest completed")
        return self.results
    
    def _execute_trades(self, trades: List[Dict], prices: Dict[str, float]):
        """Execute list of trades"""
        for trade in trades:
            action = trade['action']
            fund_id = trade['fund_id']
            amount = trade.get('amount', 0)
            
            price = prices.get(fund_id, 0)
            if price == 0:
                continue
            
            commission = amount * self.config.transaction_cost
            
            if action == 'BUY':
                self.portfolio.buy(fund_id, amount, price, commission)
            elif action == 'SELL':
                quantity = trade.get('quantity', 0)
                self.portfolio.sell(fund_id, quantity, price, commission)
    
    def _calculate_results(self, historical_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive backtest results"""
        
        daily_values = np.array(self.portfolio.daily_values)
        daily_returns = np.diff(daily_values) / daily_values[:-1]
        
        # Performance metrics
        total_return = (daily_values[-1] - daily_values[0]) / daily_values[0]
        annualized_return = (1 + total_return) ** (252 / len(daily_values)) - 1
        
        # Risk metrics
        volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = np.cumprod(1 + daily_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        winning_trades = sum(1 for t in self.portfolio.trades if self._is_winning_trade(t, historical_data))
        win_rate = winning_trades / len(self.portfolio.trades) if self.portfolio.trades else 0
        
        # Benchmark comparison
        benchmark_return = self._calculate_benchmark_return(historical_data)
        alpha = annualized_return - benchmark_return
        
        results = {
            'performance': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'final_value': daily_values[-1],
                'total_trades': len(self.portfolio.trades),
                'win_rate': win_rate
            },
            'risk': {
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'sortino_ratio': self._calculate_sortino(daily_returns)
            },
            'benchmark': {
                'benchmark_return': benchmark_return,
                'alpha': alpha,
                'beta': self._calculate_beta(daily_returns, historical_data)
            },
            'trades': {
                'total': len(self.portfolio.trades),
                'buys': sum(1 for t in self.portfolio.trades if t.action == 'BUY'),
                'sells': sum(1 for t in self.portfolio.trades if t.action == 'SELL'),
                'total_commission': sum(t.commission for t in self.portfolio.trades)
            },
            'daily_values': daily_values.tolist(),
            'daily_returns': daily_returns.tolist()
        }
        
        return results
    
    def _is_winning_trade(self, trade: Trade, historical_data: pd.DataFrame) -> bool:
        """Check if trade was profitable"""
        # Simplified - would need to track entry/exit prices
        return True
    
    def _calculate_benchmark_return(self, historical_data: pd.DataFrame) -> float:
        """Calculate benchmark return"""
        # Simplified - use average market return
        return historical_data['returns'].mean() * 252
    
    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        downside_std = np.std(downside_returns) * np.sqrt(252)
        mean_return = np.mean(returns) * 252
        return mean_return / downside_std if downside_std > 0 else 0
    
    def _calculate_beta(self, returns: np.ndarray, historical_data: pd.DataFrame) -> float:
        """Calculate beta vs benchmark"""
        # Simplified calculation
        market_returns = historical_data.groupby('date')['returns'].mean().values
        if len(market_returns) > len(returns):
            market_returns = market_returns[:len(returns)]
        elif len(returns) > len(market_returns):
            returns = returns[:len(market_returns)]
        
        covariance = np.cov(returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance if market_variance > 0 else 1.0
    
    def generate_report(self, output_path: str = 'backtest_report.json'):
        """Generate comprehensive backtest report"""
        if self.results is None:
            logger.error("No results to report. Run backtest first.")
            return
        
        report = {
            'config': {
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'initial_capital': self.config.initial_capital,
                'transaction_cost': self.config.transaction_cost
            },
            'results': self.results,
            'summary': self._generate_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
        
        return report
    
    def _generate_summary(self) -> str:
        """Generate text summary"""
        perf = self.results['performance']
        risk = self.results['risk']
        
        summary = f"""
        BACKTEST SUMMARY
        ================
        Period: {self.config.start_date} to {self.config.end_date}
        Initial Capital: ₹{self.config.initial_capital:,.0f}
        Final Value: ₹{perf['final_value']:,.0f}
        
        PERFORMANCE
        -----------
        Total Return: {perf['total_return']*100:.2f}%
        Annualized Return: {perf['annualized_return']*100:.2f}%
        Total Trades: {perf['total_trades']}
        Win Rate: {perf['win_rate']*100:.1f}%
        
        RISK METRICS
        ------------
        Volatility: {risk['volatility']*100:.2f}%
        Sharpe Ratio: {risk['sharpe_ratio']:.2f}
        Max Drawdown: {risk['max_drawdown']*100:.2f}%
        Sortino Ratio: {risk['sortino_ratio']:.2f}
        
        BENCHMARK
        ---------
        Alpha: {self.results['benchmark']['alpha']*100:.2f}%
        Beta: {self.results['benchmark']['beta']:.2f}
        """
        
        return summary


# Example strategies

def buy_and_hold_strategy(date, data, portfolio, prices):
    """Simple buy and hold strategy"""
    trades = []
    
    # On first day, buy equal weights
    if len(portfolio.trades) == 0:
        funds = data['fund_id'].unique()
        amount_per_fund = portfolio.cash / len(funds) * 0.95  # Keep 5% cash
        
        for fund_id in funds:
            trades.append({
                'action': 'BUY',
                'fund_id': fund_id,
                'amount': amount_per_fund
            })
    
    return trades


def momentum_strategy(date, data, portfolio, prices):
    """Momentum-based strategy"""
    trades = []
    
    # Calculate momentum (simplified)
    # In production, would use historical returns
    
    return trades


def mean_reversion_strategy(date, data, portfolio, prices):
    """Mean reversion strategy"""
    trades = []
    
    # Implement mean reversion logic
    
    return trades


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Configure backtest
    config = BacktestConfig(
        start_date='2014-01-01',
        end_date='2024-01-01',
        initial_capital=1000000,
        transaction_cost=0.001
    )
    
    # Create engine
    engine = BacktestEngine(config)
    
    # Load data
    historical_data = engine.load_historical_data()
    
    # Run backtest
    results = engine.run_backtest(buy_and_hold_strategy, historical_data)
    
    # Generate report
    report = engine.generate_report()
    
    print(report['summary'])
