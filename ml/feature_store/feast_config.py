"""
Feast Feature Store Configuration
Centralized feature management for ML models
"""

from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.data_source import RequestDataSource
from feast.on_demand_feature_view import on_demand_feature_view
import pandas as pd

# Define entities
user_entity = Entity(
    name="user_id",
    value_type=ValueType.STRING,
    description="User identifier"
)

fund_entity = Entity(
    name="fund_id",
    value_type=ValueType.STRING,
    description="Mutual fund identifier"
)

portfolio_entity = Entity(
    name="portfolio_id",
    value_type=ValueType.STRING,
    description="Portfolio identifier"
)

# User Features Data Source
user_features_source = FileSource(
    path="data/user_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# User Features View
user_features_view = FeatureView(
    name="user_features",
    entities=["user_id"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="age", dtype=ValueType.INT64),
        Feature(name="risk_score", dtype=ValueType.DOUBLE),
        Feature(name="total_investment", dtype=ValueType.DOUBLE),
        Feature(name="portfolio_value", dtype=ValueType.DOUBLE),
        Feature(name="investment_horizon_months", dtype=ValueType.INT64),
        Feature(name="monthly_income", dtype=ValueType.DOUBLE),
        Feature(name="kyc_status", dtype=ValueType.STRING),
        Feature(name="account_age_days", dtype=ValueType.INT64),
        Feature(name="total_transactions", dtype=ValueType.INT64),
        Feature(name="avg_transaction_amount", dtype=ValueType.DOUBLE),
        Feature(name="last_transaction_days_ago", dtype=ValueType.INT64),
    ],
    online=True,
    source=user_features_source,
    tags={"team": "ml", "category": "user"}
)

# Fund Features Data Source
fund_features_source = FileSource(
    path="data/fund_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Fund Features View
fund_features_view = FeatureView(
    name="fund_features",
    entities=["fund_id"],
    ttl=timedelta(hours=1),
    features=[
        Feature(name="nav", dtype=ValueType.DOUBLE),
        Feature(name="aum", dtype=ValueType.DOUBLE),
        Feature(name="expense_ratio", dtype=ValueType.DOUBLE),
        Feature(name="return_1m", dtype=ValueType.DOUBLE),
        Feature(name="return_3m", dtype=ValueType.DOUBLE),
        Feature(name="return_6m", dtype=ValueType.DOUBLE),
        Feature(name="return_1y", dtype=ValueType.DOUBLE),
        Feature(name="return_3y", dtype=ValueType.DOUBLE),
        Feature(name="return_5y", dtype=ValueType.DOUBLE),
        Feature(name="sharpe_ratio", dtype=ValueType.DOUBLE),
        Feature(name="beta", dtype=ValueType.DOUBLE),
        Feature(name="alpha", dtype=ValueType.DOUBLE),
        Feature(name="volatility", dtype=ValueType.DOUBLE),
        Feature(name="max_drawdown", dtype=ValueType.DOUBLE),
        Feature(name="sortino_ratio", dtype=ValueType.DOUBLE),
        Feature(name="category", dtype=ValueType.STRING),
        Feature(name="fund_house", dtype=ValueType.STRING),
    ],
    online=True,
    source=fund_features_source,
    tags={"team": "ml", "category": "fund"}
)

# Portfolio Features Data Source
portfolio_features_source = FileSource(
    path="data/portfolio_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Portfolio Features View
portfolio_features_view = FeatureView(
    name="portfolio_features",
    entities=["portfolio_id"],
    ttl=timedelta(hours=6),
    features=[
        Feature(name="total_value", dtype=ValueType.DOUBLE),
        Feature(name="total_invested", dtype=ValueType.DOUBLE),
        Feature(name="total_returns", dtype=ValueType.DOUBLE),
        Feature(name="returns_percentage", dtype=ValueType.DOUBLE),
        Feature(name="portfolio_sharpe", dtype=ValueType.DOUBLE),
        Feature(name="portfolio_beta", dtype=ValueType.DOUBLE),
        Feature(name="portfolio_volatility", dtype=ValueType.DOUBLE),
        Feature(name="num_holdings", dtype=ValueType.INT64),
        Feature(name="concentration_hhi", dtype=ValueType.DOUBLE),
        Feature(name="equity_allocation", dtype=ValueType.DOUBLE),
        Feature(name="debt_allocation", dtype=ValueType.DOUBLE),
        Feature(name="gold_allocation", dtype=ValueType.DOUBLE),
        Feature(name="hybrid_allocation", dtype=ValueType.DOUBLE),
        Feature(name="large_cap_allocation", dtype=ValueType.DOUBLE),
        Feature(name="mid_cap_allocation", dtype=ValueType.DOUBLE),
        Feature(name="small_cap_allocation", dtype=ValueType.DOUBLE),
    ],
    online=True,
    source=portfolio_features_source,
    tags={"team": "ml", "category": "portfolio"}
)

# Market Features Data Source
market_features_source = FileSource(
    path="data/market_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Market Features View (no entity - global features)
market_features_view = FeatureView(
    name="market_features",
    entities=[],
    ttl=timedelta(minutes=15),
    features=[
        Feature(name="nifty_50_value", dtype=ValueType.DOUBLE),
        Feature(name="nifty_50_change", dtype=ValueType.DOUBLE),
        Feature(name="sensex_value", dtype=ValueType.DOUBLE),
        Feature(name="sensex_change", dtype=ValueType.DOUBLE),
        Feature(name="vix_value", dtype=ValueType.DOUBLE),
        Feature(name="market_regime", dtype=ValueType.STRING),
        Feature(name="inflation_rate", dtype=ValueType.DOUBLE),
        Feature(name="repo_rate", dtype=ValueType.DOUBLE),
        Feature(name="gdp_growth", dtype=ValueType.DOUBLE),
        Feature(name="market_sentiment", dtype=ValueType.DOUBLE),
    ],
    online=True,
    source=market_features_source,
    tags={"team": "ml", "category": "market"}
)

# On-demand feature views for derived features
request_source = RequestDataSource(
    name="request_source",
    schema={
        "user_id": ValueType.STRING,
        "fund_id": ValueType.STRING,
    }
)

@on_demand_feature_view(
    inputs={
        "user_features": user_features_view,
        "portfolio_features": portfolio_features_view,
    },
    features=[
        Feature(name="risk_capacity", dtype=ValueType.DOUBLE),
        Feature(name="investment_efficiency", dtype=ValueType.DOUBLE),
        Feature(name="portfolio_health_score", dtype=ValueType.DOUBLE),
    ]
)
def derived_user_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from base features
    """
    df = pd.DataFrame()
    
    # Risk capacity = (monthly_income * 12 * 0.2) / portfolio_value
    df["risk_capacity"] = (
        features_df["monthly_income"] * 12 * 0.2
    ) / features_df["portfolio_value"].clip(lower=1)
    
    # Investment efficiency = returns_percentage / volatility
    df["investment_efficiency"] = (
        features_df["returns_percentage"] / 
        features_df["portfolio_volatility"].clip(lower=0.01)
    )
    
    # Portfolio health score (0-100)
    df["portfolio_health_score"] = (
        (features_df["portfolio_sharpe"] * 20).clip(0, 40) +
        (features_df["num_holdings"] * 2).clip(0, 20) +
        ((1 - features_df["concentration_hhi"]) * 40).clip(0, 40)
    )
    
    return df

@on_demand_feature_view(
    inputs={
        "fund_features": fund_features_view,
        "market_features": market_features_view,
    },
    features=[
        Feature(name="fund_momentum", dtype=ValueType.DOUBLE),
        Feature(name="fund_quality_score", dtype=ValueType.DOUBLE),
        Feature(name="market_adjusted_return", dtype=ValueType.DOUBLE),
    ]
)
def derived_fund_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived fund features
    """
    df = pd.DataFrame()
    
    # Momentum = weighted average of recent returns
    df["fund_momentum"] = (
        features_df["return_1m"] * 0.4 +
        features_df["return_3m"] * 0.3 +
        features_df["return_6m"] * 0.2 +
        features_df["return_1y"] * 0.1
    )
    
    # Quality score based on Sharpe, alpha, and expense ratio
    df["fund_quality_score"] = (
        features_df["sharpe_ratio"] * 30 +
        features_df["alpha"] * 40 -
        features_df["expense_ratio"] * 10
    ).clip(0, 100)
    
    # Market-adjusted return
    df["market_adjusted_return"] = (
        features_df["return_1y"] - features_df["sensex_change"]
    )
    
    return df
