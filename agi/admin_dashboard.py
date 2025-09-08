import streamlit as st
from pymongo import MongoClient
from datetime import datetime

st.set_page_config(page_title="AGI Admin Cache/Event Dashboard", layout="wide")
st.title("AGI Cache & Major Event Management Dashboard")
st.markdown("""
<div style='background-color:#fff3cd; padding:10px; border-radius:6px; color:#856404; font-size:16px;'>
<b>Disclaimer:</b> This dashboard is for research and compliance monitoring only. It does not constitute investment advice. Ensure all outputs comply with SEBI/AMFI guidelines for Mutual Fund Distributors and Research Analysts.
</div>
""", unsafe_allow_html=True)
# --- Compliance Logging ---
from pymongo import MongoClient
compliance_log = MongoClient('mongodb://localhost:27017/')['agi_cache']['compliance_log']
from datetime import datetime
compliance_log.insert_one({
    'admin_id': st.session_state.get('admin_id', 'unknown'),
    'timestamp': datetime.utcnow(),
    'action': 'admin_dashboard_access'
})

mongo_client = MongoClient('mongodb://localhost:27017/')
cache_db = mongo_client['agi_cache']

st.header("Major Events Log")
major_events = list(cache_db['major_events'].find().sort('timestamp', -1).limit(100))
if major_events:
    st.dataframe([{**e, 'timestamp': e['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} for e in major_events])
else:
    st.info("No major events logged.")

# --- Fund Analytics Controls ---
st.header("Fund Analytics Controls & Audit")
from fund_training_pipeline import run_batch_training
if st.button("Force Refresh Analytics (All Funds)"):
    run_batch_training()
    st.success("Batch analytics refresh triggered. All funds will be re-analyzed.")

if st.button("Export Compliance/Audit Log"):
    logs = list(cache_db['compliance_log'].find().sort('timestamp', -1))
    df = pd.DataFrame(logs)
    st.download_button("Download Audit Log CSV", df.to_csv(index=False), file_name="compliance_audit_log.csv")

# --- Show Fund Clustering (with style metadata) ---
st.header("Fund Clustering & Style Analysis")
fund_analytics = list(cache_db['fund_analytics'].find().limit(100))
if fund_analytics:
    df = pd.DataFrame(fund_analytics)
    if 'style' in df.columns:
        st.dataframe(df[['scheme_code', 'score', 'last_trained', 'style']])
    else:
        st.dataframe(df[['scheme_code', 'score', 'last_trained']])
else:
    st.info("No fund analytics data found.")

# --- Advanced Scenario/Regime/Forecast Visualization ---
st.header("Scenario Simulation, Regime Detection & Probabilistic Forecast")
if fund_analytics:
    selected_scheme = st.selectbox("Select scheme to view scenario/regime analytics", [f['scheme_code'] for f in fund_analytics])
    fund = next((f for f in fund_analytics if f['scheme_code'] == selected_scheme), None)
    if fund:
        # Regime history
        st.subheader("Market Regime History (last 180 days)")
        regime = fund.get('regime_history', [])
        if regime:
            regime_df = pd.DataFrame(regime)
            st.line_chart(regime_df.set_index('date')['regime'].replace({'bull':2,'sideways':1,'bear':0}))
        else:
            st.info("No regime history data.")
        # Scenario simulation
        st.subheader("Scenario Simulation (12mo Monte Carlo)")
        scenario = fund.get('scenario_simulation', {})
        if scenario:
            for k, v in scenario.items():
                st.markdown(f"**{k.title()} Regime:** Median={v['median']:.2f}, 5th%={v['min']:.2f}, 95th%={v['max']:.2f}")
                st.line_chart(pd.DataFrame(v['paths']).T)
        else:
            st.info("No scenario simulation data.")
        # Probabilistic forecast
        st.subheader("Probabilistic Forecast (NAV thresholds)")
        probf = fund.get('probabilistic_forecast', {})
        if probf:
            st.json(probf)
        else:
            st.info("No probabilistic forecast data.")

# --- Metadata Ingestion Audit/Transparency ---
st.header("Metadata Ingestion Audit Log")
meta_logs = list(cache_db['metadata_audit_log'].find().sort('timestamp', -1).limit(100))
if meta_logs:
    for log in meta_logs:
        st.markdown(f"**Scheme:** {log.get('scheme_code', '')} | **Timestamp:** {log.get('timestamp', '')}")
        st.write(f"Final Metadata: {log.get('final_metadata', {})}")
        st.write(f"Sources: {log.get('sources', [])}")
        st.markdown("---")
else:
    st.info("No metadata ingestion audit logs found.")

# --- Show Anomaly/Data Issue Alerts ---
st.header("Fund Anomaly & Data Issue Alerts")
anomaly_funds = [f for f in fund_analytics if f.get('anomalies') and len(f['anomalies']) > 5]
if anomaly_funds:
    for f in anomaly_funds:
        st.warning(f"Fund {f['scheme_code']} has {len(f['anomalies'])} NAV anomalies detected. Please review.")
        # Optionally, notify admin via WhatsApp/email here
else:
    st.success("No major fund anomalies detected in last batch.")

st.header("All User Plans & Progress (Admin View)")
user_plans = list(cache_db['user_plans'].find().sort('last_updated', -1))
if user_plans:
    user_ids = sorted(set([p['user_id'] for p in user_plans]))
    selected_user = st.selectbox("Filter by User", ["All"] + user_ids)
    filtered = [p for p in user_plans if selected_user == "All" or p['user_id'] == selected_user]
    for plan in filtered:
        st.subheader(f"User: {plan['user_id']} | Goal: {plan['goal_amount']} by {plan['goal_date']}")
        st.write(f"Success Probability: {plan.get('success_probability', 0):.2%}")
        st.write(f"Max Drawdown: {plan.get('max_drawdown', 'N/A')}")
        st.write(f"RL Results: {plan.get('rl_results', {})}")
        st.line_chart(pd.DataFrame(plan['roadmap']).set_index('month')['projected_value'])
        st.area_chart(pd.DataFrame(plan['progress_checkpoints']).set_index('month')[['min','median','max']])
        st.dataframe(plan['milestones'])
        if 'alerts' in plan:
            for alert in plan['alerts']:
                st.warning(alert)
        st.markdown("---")
else:
    st.info("No user plans found.")

st.header("Progress Alert Preview (WhatsApp/Web)")
if user_plans:
    sample = user_plans[0]
    for ms in sample['milestones']:
        st.code(f"WhatsApp/Web: Your portfolio plan for {ms['year']} projects value: {ms['projected_value']:.2f}")

st.header("Analysis Snapshots (Cache)")
if st.button("Clear ALL Snapshots (All Types)"):
    for col in cache_db.list_collection_names():
        if col.endswith('_analysis'):
            cache_db[col].delete_many({})
    st.success("All snapshots for all types cleared.")
analysis_types = [col for col in cache_db.list_collection_names() if col.endswith('_analysis')]
selected_type = st.selectbox("Select analysis type", analysis_types)
if selected_type:
    snapshots = list(cache_db[selected_type].find().sort('timestamp', -1).limit(100))
    if snapshots:
        st.dataframe([{**s, 'timestamp': s['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} for s in snapshots])
        if st.button("Clear All Snapshots for This Type"):
            cache_db[selected_type].delete_many({})
            st.success("All snapshots cleared.")
    else:
        st.info("No snapshots found for this analysis type.")

st.header("Admin Notification Controls & Log")
event_types = sorted(list(set([e['type'] for e in major_events])))
admin_id = st.text_input("Admin ID for notifications", "admin")
channel = st.selectbox("Notification channel", ["web", "whatsapp", "email"])
subscribe_event = st.selectbox("Event type to subscribe", event_types)
if st.button("Subscribe to Event Notifications"):
    cache_db['admin_notify'].update_one({'admin_id': admin_id, 'event_type': subscribe_event}, {"$set": {'channel': channel}}, upsert=True)
    st.success(f"Subscribed {admin_id} to {subscribe_event} events.")
unsubscribe_event = st.selectbox("Event type to unsubscribe", event_types)
if st.button("Unsubscribe from Event Notifications"):
    cache_db['admin_notify'].delete_one({'admin_id': admin_id, 'event_type': unsubscribe_event})
    st.success(f"Unsubscribed {admin_id} from {unsubscribe_event} events.")
st.subheader("Notification Log")
notif_log = list(cache_db['notification_log'].find().sort('timestamp', -1).limit(100))
if notif_log:
    st.dataframe([{**n, 'timestamp': n['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} for n in notif_log])
else:
    st.info("No notifications sent yet.")
