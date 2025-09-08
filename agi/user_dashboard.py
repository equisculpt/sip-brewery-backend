import streamlit as st
from pymongo import MongoClient
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Portfolio Planning Dashboard", layout="wide")
st.title("Your Autonomous Portfolio Plan & Progress")
st.markdown("""
<div style='background-color:#fff3cd; padding:10px; border-radius:6px; color:#856404; font-size:16px;'>
<b>Disclaimer:</b> This tool is for research and educational purposes only. It does not constitute investment advice. Please consult your SEBI-registered Investment Adviser before making investment decisions. As a SEBI/AMFI-registered Mutual Fund Distributor and Research Analyst, we do not provide personalized investment advice or portfolio suitability assessment.
</div>
""", unsafe_allow_html=True)
# --- Compliance Logging ---
from pymongo import MongoClient
compliance_log = MongoClient('mongodb://localhost:27017/')['agi_cache']['compliance_log']
from datetime import datetime
compliance_log.insert_one({
    'user_id': user_id,
    'timestamp': datetime.utcnow(),
    'action': 'dashboard_access',
    'plan_goal': plan['goal_amount'] if 'plan' in locals() else None,
    'goal_date': plan['goal_date'] if 'plan' in locals() else None
})

mongo_client = MongoClient('mongodb://localhost:27017/')
plans_db = mongo_client['agi_cache']

# --- User selection ---
user_id = st.text_input("User ID", "user123")
plan_docs = list(plans_db['user_plans'].find({'user_id': user_id}))
if not plan_docs:
    st.warning("No plan found for this user. Submit a new plan via the AGI API.")
    st.stop()

selected_plan = st.selectbox("Select a plan/goal", [f"Goal: {d['goal_amount']} by {d['goal_date']}" for d in plan_docs])
plan = plan_docs[[f"Goal: {d['goal_amount']} by {d['goal_date']}" for d in plan_docs].index(selected_plan)]

# --- Mutual Fund Scheme Portfolio Analysis Button ---
st.header("Mutual Fund Scheme Portfolio Analysis")
scheme_code = st.text_input("Enter Mutual Fund Scheme Code", "118834")
if st.button("Show Scheme Portfolio Analysis"):
    from fund_training_pipeline import run_training_for_scheme
    result = run_training_for_scheme(scheme_code)
    st.subheader("Portfolio Analysis (Explainable)")
    st.write(f"Fund Score: {result['score']:.3f}")
    st.write(f"Detected Anomalies (NAV outliers): {result['anomalies']}")
    st.write("Top Holdings:")
    st.table(result['portfolio_analysis']['top_holdings'])
    st.write("Sector Exposure:")
    st.json(result['portfolio_analysis']['sector_exposure'])
    st.write("Recent NAVs:")
    st.dataframe(result['nav_tail'])
    if result['bench_tail']:
        st.write("Recent Benchmark Data:")
        st.dataframe(result['bench_tail'])
    st.info("This analysis is based on the latest available NAV and portfolio holding data, using clustering, anomaly detection, and scoring. For educational and research purposes only.")

# --- Roadmap Visualization ---
st.header("Investment Roadmap & Progress Bands")
roadmap = pd.DataFrame(plan['roadmap'])
progress = pd.DataFrame(plan['progress_checkpoints'])
st.line_chart(roadmap.set_index('month')['projected_value'], height=300)
st.area_chart(progress.set_index('month')[['min','median','max']], height=200)

# --- Milestones ---
st.subheader("Yearly Milestones")
st.dataframe(plan['milestones'])

# --- Constraints & RL Results ---
st.subheader("Constraints & RL Optimization")
st.write(f"Max Drawdown (95th pct): {plan.get('max_drawdown', 'N/A'):.2%}")
if plan.get('rl_results'):
    st.json(plan['rl_results'])

# --- User Controls ---
st.header("Re-Planning & Constraint Editing")
if st.button("Trigger Re-Planning Now"):
    from agi_microservice import trigger_replanning
    trigger_replanning(user_id, plan['goal_amount'], plan['goal_date'])
    st.success("Re-planning triggered!")

st.subheader("Edit Constraints")
max_dd = st.number_input("Max Drawdown (%)", value=plan.get('max_drawdown', -0.2)*100, min_value=-100.0, max_value=0.0) / 100
liquidity = st.text_area("Liquidity Needs (JSON: {month: amount})", value=str(plan.get('liquidity_needs', {})))
use_rl = st.checkbox("Use RL Optimization", value=plan.get('rl_results') is not None)
if st.button("Update Constraints & Re-Plan"):
    from agi_microservice import autonomous_plan, PlanningRequest
    import json
    req = PlanningRequest(
        user_id=user_id,
        goal_amount=plan['goal_amount'],
        goal_date=plan['goal_date'],
        start_amount=plan['roadmap'][0]['projected_value'],
        monthly_contribution=plan['roadmap'][1]['projected_value']-plan['roadmap'][0]['projected_value'],
        risk_tolerance=plan.get('risk_tolerance', 'moderate'),
        max_drawdown=max_dd,
        liquidity_needs=json.loads(liquidity),
        use_rl=use_rl
    )
    autonomous_plan(req)
    st.success("Constraints updated and plan re-computed!")

# --- Progress Alerts ---
st.header("Progress Alerts")
alerts = []
for i, row in progress.iterrows():
    if row['median'] < plan['goal_amount'] * (i / len(progress)):
        alerts.append(f"Warning: At month {row['month']}, projected median value is behind linear goal trajectory.")
if alerts:
    for a in alerts:
        st.warning(a)
else:
    st.success("On track to reach your goal!")
