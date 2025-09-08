import requests
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from datetime import datetime
import time
import threading
from pymongo import MongoClient

# --- CONFIG ---
NAV_API = "https://api.mfapi.in/mf/{scheme_code}"
BENCHMARK_API = "https://api.example.com/benchmarks/{benchmark_code}"  # Placeholder
PORTFOLIO_WEB_SCRAPE = True
ALL_SCHEMES_API = "https://api.mfapi.in/mf"
AUTO_REFRESH_INTERVAL_HOURS = 24

# --- DATA INGESTION ---
def fetch_nav_history(scheme_code):
    url = NAV_API.format(scheme_code=scheme_code)
    r = requests.get(url)
    data = r.json()
    nav_df = pd.DataFrame(data['data'])
    nav_df['date'] = pd.to_datetime(nav_df['date'])
    nav_df['nav'] = pd.to_numeric(nav_df['nav'], errors='coerce')
    return nav_df.sort_values('date')

def fetch_benchmark(benchmark_code):
    url = BENCHMARK_API.format(benchmark_code=benchmark_code)
    r = requests.get(url)
    data = r.json()
    bench_df = pd.DataFrame(data['data'])
    bench_df['date'] = pd.to_datetime(bench_df['date'])
    bench_df['value'] = pd.to_numeric(bench_df['value'], errors='coerce')
    return bench_df.sort_values('date')

# --- PORTFOLIO SCRAPING (Robust, fallback sources) ---
def fetch_portfolio(scheme_code):
    try:
        # Try AGI research agent (web search)
        from agi_research_agent import get_fund_portfolio
        portfolio = get_fund_portfolio(scheme_code)
        if portfolio and len(portfolio) > 0:
            return portfolio
    except Exception:
        pass
    # Fallback: try scraping from public AMCs, ValueResearch, Moneycontrol, etc.
    # (Pseudo-code, replace with real scraping logic)
    try:
        import requests
        url = f"https://www.valueresearchonline.com/funds/newsnapshot.asp?schemecode={scheme_code}"
        # ...parse page for portfolio table...
        return []  # Placeholder
    except Exception:
        return []

# --- METADATA INGESTION (Automated via AGI/web/AMC) ---
def get_fund_metadata(scheme_code):
    # Try AGI research agent (web/AMC search)
    try:
        from agi_research_agent import get_fund_metadata as agi_get_fund_metadata
        meta = agi_get_fund_metadata(scheme_code)
        if meta:
            return meta
    except Exception:
        pass
    # Fallback: scrape trusted sites (ValueResearch, Moneycontrol, AMC direct)
    try:
        import requests
        # ValueResearch
        url = f"https://www.valueresearchonline.com/funds/newsnapshot.asp?schemecode={scheme_code}"
        resp = requests.get(url)
        # ...parse HTML for style, AMC, liquidity, ESG...
        # Placeholder: return dummy values
        return {'style': 'equity', 'liquidity': 0.05, 'amc': 'HDFC', 'esg_score': 0.7}
    except Exception:
        return {'style': 'other', 'liquidity': 0, 'amc': 'unknown', 'esg_score': 0.5}

# --- MULTI-FACTOR CLUSTERING ---
def cluster_funds(nav_histories, fund_metadata=None, n_clusters=5):
    # nav_histories: dict of {scheme_code: nav_df}
    # fund_metadata: dict of {scheme_code: {style, liquidity, amc, esg_score, ...}}
    features = []
    codes = []
    for code, nav in nav_histories.items():
        meta = fund_metadata.get(code, {}) if fund_metadata else get_fund_metadata(code)
        returns = nav['nav'].pct_change().fillna(0).values[-365:]
        if len(returns) == 365:
            risk = np.std(returns)
            drawdown = np.min((nav['nav'] / nav['nav'].cummax() - 1).values[-365:])
            mean_return = np.mean(returns)
            # Style: one-hot encode or ordinal (equity=0, debt=1, hybrid=2, etc.)
            style_map = {'equity': 0, 'debt': 1, 'hybrid': 2, 'sector': 3, 'other': 4}
            style_feat = style_map.get(meta.get('style', 'other').lower(), 4)
            # Liquidity: e.g., % in cash or liquid assets
            liquidity = float(meta.get('liquidity', 0))
            # AMC: one-hot or ordinal (hash to int for clustering)
            amc = meta.get('amc', 'unknown')
            amc_feat = hash(amc) % 1000 / 1000  # Normalize to [0,1]
            # ESG: normalized ESG score (0-1)
            esg = float(meta.get('esg_score', 0.5))
            features.append([mean_return, risk, drawdown, style_feat, liquidity, amc_feat, esg])
            codes.append(code)
    X = np.stack(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    return dict(zip(codes, kmeans.labels_))

# --- ANOMALY DETECTION ---
def detect_anomalies(nav_histories):
    anomalies = {}
    for code, nav in nav_histories.items():
        returns = nav['nav'].pct_change().fillna(0).values[-365:]
        if len(returns) == 365:
            model = IsolationForest(contamination=0.01, random_state=42)
            model.fit(returns.reshape(-1, 1))
            scores = model.decision_function(returns.reshape(-1, 1))
            anomalies[code] = np.where(scores < 0)[0].tolist()
    return anomalies

# --- FUND SCORING ---
def score_fund(nav, bench=None):
    nav = nav.copy()
    nav['returns'] = nav['nav'].pct_change().fillna(0)
    score = nav['returns'].mean() / nav['returns'].std()
    if bench is not None:
        merged = pd.merge(nav, bench, on='date', suffixes=('', '_bench'))
        merged['alpha'] = merged['returns'] - merged['value'].pct_change().fillna(0)
        score += merged['alpha'].mean()
    return score

# --- FUND PORTFOLIO ANALYSIS ---
def analyze_portfolio(portfolio):
    df = pd.DataFrame(portfolio)
    sector_exposure = df.groupby('sector')['value'].sum().to_dict() if 'sector' in df else {}
    top_holdings = df.sort_values('value', ascending=False).head(10).to_dict('records')
    return {'sector_exposure': sector_exposure, 'top_holdings': top_holdings}

# --- MACROECONOMIC FEED INTEGRATION ---
def fetch_macro_triggers():
    # Fetch macro triggers using AGI web search and LLM-based extraction for real-time global macro
    macro = {}
    try:
        from agi_research_agent import get_macro_data_via_websearch
        macro = get_macro_data_via_websearch()
    except Exception:
        # Fallback: static defaults
        macro = {'rate_hike': False, 'rbi_hike': False, 'VIX': 18, 'high_vol': False}
    return macro

# Example AGI research agent function (to be implemented in agi_research_agent.py):
# def get_macro_data_via_websearch():
#     """
#     Use web search and LLM extraction to pull latest macro data (rates, inflation, VIX, etc)
#     """
#     ...

# --- SCENARIO SIMULATION, REGIME DETECTION, PROBABILISTIC FORECASTING ---
def detect_market_regime_transformer(nav):
    # Transformer-based regime detection (deep learning)
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import MinMaxScaler
        # Prepare data
        nav = nav.copy()
        nav['returns'] = nav['nav'].pct_change().fillna(0)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(nav['returns'].values.reshape(-1,1))
        seq_len = 30
        X_seq = np.array([X[i-seq_len:i] for i in range(seq_len, len(X))])
        # Transformer model definition
        class RegimeTransformer(nn.Module):
            def __init__(self, d_model=16, nhead=2, num_layers=2):
                super().__init__()
                self.embedding = nn.Linear(1, d_model)
                encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.fc = nn.Linear(d_model, 3)
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                out = self.fc(x[:, -1, :])
                return out
        model = RegimeTransformer()
        # NOTE: In production, load pre-trained weights
        with torch.no_grad():
            logits = model(torch.tensor(X_seq, dtype=torch.float32))
            preds = torch.argmax(logits, dim=1).numpy()
        regime_map = {0:'bear', 1:'sideways', 2:'bull'}
        regimes = [regime_map[p] for p in preds]
        nav = nav.iloc[seq_len:].copy()
        nav['regime'] = regimes
        return nav[['date','regime']].tail(180).to_dict('records')
    except Exception as e:
        # Fallback to LSTM/HMM/rolling
        try:
            return detect_market_regime_lstm(nav)
        except Exception:
            return detect_market_regime(nav)

def simulate_scenarios(nav, bench=None, n_sim=1000, future_months=12):
    # Monte Carlo simulation for future NAVs under different regimes
    nav = nav.copy()
    returns = nav['nav'].pct_change().fillna(0).values[-365:]
    scenarios = {}
    for regime, mu, sigma in [
        ('bull', np.mean(returns)+0.01, np.std(returns)*0.8),
        ('bear', np.mean(returns)-0.01, np.std(returns)*1.2),
        ('sideways', np.mean(returns), np.std(returns))
    ]:
        sims = []
        for _ in range(n_sim):
            path = [nav['nav'].iloc[-1]]
            for _ in range(future_months):
                r = np.random.normal(mu, sigma)
                path.append(path[-1]*(1+r))
            sims.append(path)
        sims = np.array(sims)
        scenarios[regime] = {
            'median': sims[:, -1].mean(),
            'min': np.percentile(sims[:, -1], 5),
            'max': np.percentile(sims[:, -1], 95),
            'paths': sims[:, ::future_months//4].tolist()  # sample for dashboard
        }
    return scenarios

def simulate_custom_scenario(nav, event, n_sim=1000, future_months=12):
    # User-definable macro event: e.g. {'type': 'rate_hike', 'shock': -0.02, 'vol_mult': 1.5}
    nav = nav.copy()
    returns = nav['nav'].pct_change().fillna(0).values[-365:]
    mu = np.mean(returns)
    sigma = np.std(returns)
    shock = event.get('shock', 0)
    vol_mult = event.get('vol_mult', 1.0)
    mu += shock
    sigma *= vol_mult
    sims = []
    for _ in range(n_sim):
        path = [nav['nav'].iloc[-1]]
        for _ in range(future_months):
            r = np.random.normal(mu, sigma)
            path.append(path[-1]*(1+r))
        sims.append(path)
    sims = np.array(sims)
    return {
        'event': event,
        'median': sims[:, -1].mean(),
        'min': np.percentile(sims[:, -1], 5),
        'max': np.percentile(sims[:, -1], 95),
        'paths': sims[:, ::future_months//4].tolist()
    }

def probabilistic_forecast(nav, bench=None, future_months=12):
    # Output probability of NAV being above/below thresholds in future
    nav = nav.copy()
    returns = nav['nav'].pct_change().fillna(0).values[-365:]
    mu, sigma = np.mean(returns), np.std(returns)
    last_nav = nav['nav'].iloc[-1]
    thresholds = [last_nav*1.1, last_nav*0.9]
    prob_results = {}
    for t in thresholds:
        # Analytical: probability final NAV > t
        z = (np.log(t/last_nav) - future_months*mu) / (np.sqrt(future_months)*sigma)
        from scipy.stats import norm
        prob = 1 - norm.cdf(z)
        prob_results[f'prob_above_{t:.2f}'] = prob
    return prob_results

# --- PIPELINE ENTRY POINT ---
def run_training_for_scheme(scheme_code, benchmark_code=None, store_result=True):
    from prediction_error_log import log_prediction
    from feedback_learning import get_feedback
    from causal_inference import explain_error_causality
    from natural_language_postmortem import generate_postmortem
    nav = fetch_nav_history(scheme_code)
    bench = fetch_benchmark(benchmark_code) if benchmark_code else None
    portfolio = fetch_portfolio(scheme_code)
    anomaly_points = detect_anomalies({scheme_code: nav})[scheme_code]
    score = score_fund(nav, bench)
    portfolio_analysis = analyze_portfolio(portfolio)
    # --- Advanced analytics ---
    regime_history = detect_market_regime(nav, bench)
    scenario_sim = simulate_scenarios(nav, bench)
    prob_forecast = probabilistic_forecast(nav, bench)
    # --- Log predictions for continual learning ---
    context = {
        'macro': fetch_macro_triggers(),
        'features': portfolio_analysis,
        'model_version': 'v1',
    }
    pred_docs = []
    pred_docs.append(log_prediction(scheme_code, 'regime', regime_history, context))
    pred_docs.append(log_prediction(scheme_code, 'scenario', scenario_sim, context))
    pred_docs.append(log_prediction(scheme_code, 'nav_forecast', prob_forecast, context))
    # --- Schedule causal analysis, feedback review, post-mortem generation ---
    feedbacks = get_feedback(limit=10)
    for fb in feedbacks:
        # Causal analysis for flagged errors
        causal = explain_error_causality(fb['scheme_code'], fb['prediction_id'])
        # Generate post-mortem
        generate_postmortem(fb['prediction_id'], adaptation_details=causal)
    result = {
        'scheme_code': scheme_code,
        'score': score,
        'anomalies': anomaly_points,
        'portfolio_analysis': portfolio_analysis,
        'nav_tail': nav.tail(10).to_dict('records'),
        'bench_tail': bench.tail(10).to_dict('records') if bench is not None else None,
        'regime_history': regime_history,
        'scenario_simulation': scenario_sim,
        'probabilistic_forecast': prob_forecast,
        'last_trained': datetime.utcnow()
    }
    if store_result:
        MongoClient('mongodb://localhost:27017/')['agi_cache']['fund_analytics'].update_one({'scheme_code': scheme_code}, {"$set": result}, upsert=True)
    return result

# --- BATCH/AUTOMATIC TRAINING ---
def run_batch_training():
    r = requests.get(ALL_SCHEMES_API)
    all_schemes = r.json()
    codes = [f['schemeCode'] for f in all_schemes['data']]
    for code in codes:
        try:
            run_training_for_scheme(code)
        except Exception as e:
            print(f"Error training scheme {code}: {e}")

# --- PERIODIC AUTO-REFRESH (runs in background thread) ---
def auto_refresh_loop():
    while True:
        print(f"[Auto-Refresh] Starting batch training at {datetime.utcnow()}")
        run_batch_training()
        print(f"[Auto-Refresh] Sleeping {AUTO_REFRESH_INTERVAL_HOURS} hours...")
        time.sleep(AUTO_REFRESH_INTERVAL_HOURS * 3600)

def start_auto_training():
    t = threading.Thread(target=auto_refresh_loop, daemon=True)
    t.start()

# --- Start auto-training on import ---
start_auto_training()

if __name__ == "__main__":
    # Manual trigger for test/debug
    run_batch_training()
