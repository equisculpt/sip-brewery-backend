# 🧠 Machine Learning Infrastructure

Production-ready ML infrastructure for SIP Brewery platform.

## 📁 Directory Structure

```
ml/
├── feature_store/          # Feast feature store configuration
│   ├── feast_config.py     # Feature definitions
│   ├── feature_repo/       # Feast repository
│   └── data/               # Feature data (parquet files)
├── models/                 # ML model implementations
│   ├── portfolio_optimizer.py  # Baseline portfolio optimizer
│   └── ...                 # Future models
├── training/               # Training scripts
├── inference/              # Inference services
└── notebooks/              # Jupyter notebooks for experimentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install feast scikit-learn scipy numpy pandas
```

### 2. Initialize Feast Feature Store

```bash
cd ml/feature_store/feature_repo
feast apply
```

### 3. Test Portfolio Optimizer

```bash
cd ml/models
python portfolio_optimizer.py
```

## 📊 Feature Store

### Available Features

**User Features:**
- age, risk_score, total_investment, portfolio_value
- investment_horizon_months, monthly_income
- kyc_status, account_age_days
- total_transactions, avg_transaction_amount

**Fund Features:**
- nav, aum, expense_ratio
- returns (1M, 3M, 6M, 1Y, 3Y, 5Y)
- risk metrics (sharpe_ratio, beta, alpha, volatility)
- category, fund_house

**Portfolio Features:**
- total_value, total_invested, returns
- risk metrics (sharpe, beta, volatility)
- allocation breakdown (equity, debt, gold, hybrid)
- concentration metrics

**Market Features:**
- nifty_50, sensex, vix
- market_regime, sentiment
- macro indicators (inflation, repo_rate, gdp_growth)

### Derived Features

**On-demand computed features:**
- risk_capacity
- investment_efficiency
- portfolio_health_score
- fund_momentum
- fund_quality_score
- market_adjusted_return

## 🎯 Portfolio Optimizer

### Features

- **Mean-Variance Optimization** (Markowitz)
- **Maximum Sharpe Ratio** optimization
- **Minimum Variance** portfolio
- **Efficient Frontier** generation
- **Risk-constrained** optimization
- **Sector/category constraints**
- **Rebalancing recommendations**

### Usage

```python
from portfolio_optimizer import PortfolioOptimizer
import numpy as np

# Initialize optimizer
optimizer = PortfolioOptimizer(risk_free_rate=0.06)

# Optimize portfolio
result = optimizer.optimize_portfolio(
    expected_returns=expected_returns,
    covariance_matrix=covariance_matrix,
    fund_metadata=fund_metadata,
    user_risk_profile='moderate'
)

print(f"Expected Return: {result['expected_return']:.2%}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
```

## 🔄 Next Steps

### Phase 1: Production ML Models (Months 1-4)

1. **Portfolio Optimizer RL** - Reinforcement learning for dynamic allocation
2. **Fund Performance Predictor** - Transformer + GNN for fund prediction
3. **Risk Predictor** - LSTM + Attention for risk forecasting
4. **Behavioral Predictor** - BERT-based NLP for user actions
5. **Market Regime Detector** - Classification model for market states

### Phase 2: MLOps Pipeline

1. **MLflow Integration** - Experiment tracking and model registry
2. **Kubeflow Pipelines** - Automated training workflows
3. **Model Monitoring** - Drift detection and performance tracking
4. **A/B Testing Framework** - Compare model versions
5. **Feature Engineering Pipeline** - Automated feature computation

### Phase 3: Advanced Models

1. **Graph Neural Networks** - Fund relationship analysis
2. **Federated Learning** - Privacy-preserving ML
3. **AutoML** - Automated model selection and tuning
4. **Ensemble Models** - Combine multiple models
5. **Quantum-inspired Optimization** - Ultra-fast portfolio optimization

## 📈 Model Performance Targets

| Model | Metric | Target |
|-------|--------|--------|
| Portfolio Optimizer | Sharpe Ratio | > 1.5 |
| Fund Predictor | Accuracy | > 85% |
| Risk Predictor | MAE | < 2% |
| Behavioral Predictor | F1 Score | > 0.80 |
| Market Regime | Accuracy | > 75% |

## 🔐 Security & Compliance

- All models comply with SEBI/AMFI regulations
- No PII used in training data
- Explainable AI for all predictions
- Audit trail for all model decisions
- Regular bias and fairness audits

## 📝 Documentation

- [Feature Store Guide](./feature_store/README.md)
- [Model Training Guide](./training/README.md)
- [Inference API Guide](./inference/README.md)
- [MLOps Best Practices](./docs/mlops.md)

## 🤝 Contributing

1. Create feature branch
2. Add tests for new models
3. Document model architecture
4. Submit PR with performance metrics

## 📞 Support

For ML infrastructure questions, contact the ML team.
