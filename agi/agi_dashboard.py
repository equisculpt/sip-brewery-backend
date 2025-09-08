import streamlit as st
import pandas as pd
from pymongo import MongoClient

st.set_page_config(page_title="AGI Retraining & Feedback Dashboard", layout="wide")
st.title("AGI Retraining, Feedback, and Audit Monitoring")

mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client['mf_data']

# Retraining audit
st.header("Retraining Audit Logs")
audit_col = db['retraining_audit']
audit_df = pd.DataFrame(list(audit_col.find().sort('timestamp', -1).limit(100)))
if not audit_df.empty:
    st.dataframe(audit_df[['scheme_name','model','last_nav','predicted_nav_30d','timestamp']])
else:
    st.info("No retraining audit logs found.")

# Feedback
st.header("User Feedback Scores")
feedback_col = db['user_feedback']
feedback_df = pd.DataFrame(list(feedback_col.aggregate([
    {"$group": {"_id": "$scheme_name", "avg_score": {"$avg": "$accuracy_score"}, "count": {"$sum": 1}}}
])))
if not feedback_df.empty:
    st.dataframe(feedback_df.rename(columns={"_id": "scheme_name"}))
else:
    st.info("No user feedback found.")

# Fund manager updates
st.header("Latest Fund Manager Intelligence")
manager_col = db['fund_managers']
manager_df = pd.DataFrame(list(manager_col.find().sort('last_updated', -1).limit(100)))
if not manager_df.empty:
    st.dataframe(manager_df[['scheme_name','manager_snippet','last_updated']])
else:
    st.info("No fund manager intelligence found.")

# RL Signals and Long-term Investment Analytics
st.header("RL Investment Signals (Long-term)")
rl_col = db['retraining_audit']
rl_df = pd.DataFrame(list(rl_col.find({'model': 'RL-DQN'}).sort('timestamp', -1).limit(200)))
if not rl_df.empty:
    if 'scheme_name' in rl_df.columns:
        st.dataframe(rl_df[['scheme_name', 'rl_signal', 'holding_period', 'timestamp', 'explanation']])
    elif 'symbol' in rl_df.columns:
        st.dataframe(rl_df[['symbol', 'rl_signal', 'holding_period', 'timestamp', 'explanation']])
    else:
        st.dataframe(rl_df)
else:
    st.info("No RL-based investment signals found.")

# Advanced Analytics: Drawdown, Sharpe Ratio, Asset Correlations, Rolling Volatility, User Feedback
st.header("Advanced Analytics")
import numpy as np
# Drawdown for mutual funds
if not audit_df.empty:
    navs = audit_df[['scheme_name','last_nav','predicted_nav_30d']].groupby('scheme_name').last()
    navs['drawdown'] = navs['predicted_nav_30d'] / navs['last_nav'] - 1
    st.subheader("Drawdown (Predicted 30d vs Last NAV)")
    st.bar_chart(navs['drawdown'])
    # Sharpe Ratio
    nav_returns = navs['predicted_nav_30d'] - navs['last_nav']
    sharpe = nav_returns.mean() / (nav_returns.std() + 1e-8)
    st.metric("Mutual Fund Sharpe Ratio (predicted)", f"{sharpe:.2f}")
    # Rolling volatility
    st.subheader("Rolling Volatility (30d)")
    navs['rolling_vol'] = pd.Series(navs['last_nav']).rolling(30).std()
    st.line_chart(navs['rolling_vol'])
    # Max drawdown
    st.subheader("Max Drawdown")
    navs['cummax'] = navs['last_nav'].cummax()
    navs['max_drawdown'] = navs['cummax'] - navs['last_nav']
    st.bar_chart(navs['max_drawdown'])
# Asset correlations (stub/demo)
if not audit_df.empty:
    st.subheader("Asset Correlations (Predicted NAV)")
    corr = audit_df.pivot_table(index='timestamp', columns='scheme_name', values='predicted_nav_30d').corr()
    st.dataframe(corr)
# User feedback on RL explanations
st.header("User Feedback on RL Explanations")
rl_feedback = pd.DataFrame(list(db['user_feedback'].find({'model': {'$regex': 'RL'}})))
if not rl_feedback.empty:
    st.dataframe(rl_feedback[['symbol' if 'symbol' in rl_feedback.columns else 'scheme_name', 'explanation_score', 'timestamp']])
else:
    st.info("No user feedback on RL explanations found.")
# Sector/Industry Analytics for ETFs
st.header("ETF Sector/Industry Breakdown & Correlations")
etf_col = db['etfs']
etf_df = pd.DataFrame(list(etf_col.find({}, {'symbol': 1, 'sector': 1, 'industry': 1})))
if not etf_df.empty:
    st.dataframe(etf_df)
    sector_counts = etf_df['sector'].value_counts()
    st.bar_chart(sector_counts)
    # Sector correlation matrix (stub: extend with real returns by sector)
    st.subheader("Sector Correlation Matrix (ETF count)")
    sector_corr = pd.crosstab(etf_df['sector'], etf_df['industry'])
    st.dataframe(sector_corr)
else:
    st.info("No ETF sector/industry data found.")

# Prometheus/Alert Status (stub)
st.header("Production Monitoring & Alerts")
st.write("Prometheus metrics and alert status (stub/demo):")
st.code("agi_retrain_success{scheme=...} 1")

# Deeper Analytics: NAV Prediction Distributions, Feedback Trends
st.header("Analytics: NAV Prediction & Feedback Trends")
if not audit_df.empty:
    st.line_chart(audit_df.set_index('timestamp')['predicted_nav_30d'])
if not feedback_df.empty:
    st.bar_chart(feedback_df.set_index('scheme_name')['avg_score'])
