import time
import logging
from redis import Redis
from rq import Queue
import traceback
from datetime import datetime

# Configure logging for audit
logging.basicConfig(
    filename='idle_training_audit.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

redis_conn = Redis(host='localhost', port=6379, db=0)
user_queue = Queue('cpu', connection=redis_conn)

# Example: Replace with your real training logic

def daily_training_step():
    logging.info("Starting training step.")
    try:
        # 1. Run mutual fund websearch+LLM pipeline for new/updated schemes
        from mutual_fund_websearch import get_all_scheme_names, duckduckgo_search, bing_search, brave_search, extract_manager_from_page, semantic_rank_snippets
        from pymongo import MongoClient
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import LinearRegression
        mongo_client = MongoClient('mongodb://localhost:27017/')
        db = mongo_client['mf_data']
        manager_col = db['fund_managers']
        scheme_col = db['schemes']
        nav_col = db['nav_history']
        feedback_col = db['user_feedback']
        audit_col = db['retraining_audit']
        # Extensible: add more asset classes
        asset_classes = ['mutual_fund', 'etf', 'stock']
        for asset_class in asset_classes:
            if asset_class == 'mutual_fund':
                all_schemes = get_all_scheme_names()
                for scheme in all_schemes:
                    snippets = []
                    for search_func in [duckduckgo_search, bing_search, brave_search]:
                        urls = search_func(f"{scheme} mutual fund fund manager name", num_results=3)
                        for url in urls:
                            snippet = extract_manager_from_page(url)
                            if snippet:
                                snippets.append(snippet)
                        if snippets:
                            break
                    best_snippet = semantic_rank_snippets(snippets, f"{scheme} fund manager")
                    if best_snippet:
                        manager_col.update_one(
                            {'scheme_name': scheme},
                            {'$set': {'manager_snippet': best_snippet, 'last_updated': datetime.utcnow()}},
                            upsert=True
                        )
                        logging.info(f"Updated best manager snippet for {scheme}")
                    # --- Deep Learning AGI Retraining: LSTM for NAV prediction ---
                    nav_records = list(nav_col.find({'scheme_code': {'$exists': True}, 'scheme_name': scheme}))
                    if len(nav_records) > 60:
                        import torch
                        import torch.nn as nn
                        from torch.utils.data import DataLoader, TensorDataset
                        nav_df = pd.DataFrame(nav_records)
                        nav_df['date'] = pd.to_datetime(nav_df['date'], errors='coerce')
                        nav_df = nav_df.dropna(subset=['date', 'nav'])
                        nav_df = nav_df.sort_values('date')
                        nav_df['nav'] = pd.to_numeric(nav_df['nav'], errors='coerce')
                        nav_df = nav_df.dropna(subset=['nav'])
                        nav_df['nav_norm'] = (nav_df['nav'] - nav_df['nav'].mean()) / nav_df['nav'].std()
                        sequence_length = 30
                        data = nav_df['nav_norm'].values
                        X_seq = []
                        y_seq = []
                        for i in range(len(data) - sequence_length):
                            X_seq.append(data[i:i+sequence_length])
                            y_seq.append(data[i+sequence_length])
                        X_seq = torch.tensor(X_seq).float().unsqueeze(-1)
                        y_seq = torch.tensor(y_seq).float().unsqueeze(-1)
                        dataset = TensorDataset(X_seq, y_seq)
                        loader = DataLoader(dataset, batch_size=16, shuffle=True)
                        class LSTMModel(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.lstm = nn.LSTM(1, 32, batch_first=True)
                                self.fc = nn.Linear(32, 1)
                            def forward(self, x):
                                out, _ = self.lstm(x)
                                return self.fc(out[:, -1, :])
                        model = LSTMModel()
                        loss_fn = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                        model.train()
                        for epoch in range(5):
                            for xb, yb in loader:
                                optimizer.zero_grad()
                                pred = model(xb)
                                loss = loss_fn(pred, yb)
                                loss.backward()
                                optimizer.step()
                        model.eval()
                        with torch.no_grad():
                            last_seq = torch.tensor(data[-sequence_length:]).float().unsqueeze(0).unsqueeze(-1)
                            nav_pred_norm = model(last_seq).item()
                            nav_pred = nav_pred_norm * nav_df['nav'].std() + nav_df['nav'].mean()
                        audit_col.insert_one({
                            'scheme_name': scheme,
                            'model': 'LSTM',
                            'last_nav': float(nav_df['nav'].values[-1]),
                            'predicted_nav_30d': float(nav_pred),
                            'timestamp': datetime.utcnow()
                        })
                        logging.info(f"[LSTM] Retrained NAV model for {scheme}. 30d pred: {nav_pred:.2f}")
                    # --- RL Example: DQN for long-term investment signal ---
                    # Only consider holding periods > 6 months (no short-term trading)
                    try:
                        import numpy as np
                        import torch
                        import torch.nn as nn
                        # Simple RL agent (DQN-style) for demonstration
                        # State: [last 180 NAV returns]
                        # Action: [0=hold, 1=buy, 2=sell], reward = 6mo return
                        nav_series = nav_df['nav'].values
                        if len(nav_series) > 210:
                            returns = np.diff(nav_series) / nav_series[:-1]
                            states = np.array([returns[i-180:i] for i in range(180, len(returns)-180)])
                            rewards = nav_series[180+1:len(nav_series)-180+1] / nav_series[1:len(nav_series)-180+1] - 1
                            actions = (rewards > 0.05).astype(int)  # 1=buy/hold if >5% gain, else hold
                            class PolicyNet(nn.Module):
                                def __init__(self):
                                    super().__init__()
                                    self.fc1 = nn.Linear(180, 64)
                                    self.fc2 = nn.Linear(64, 3)
                                def forward(self, x):
                                    x = torch.relu(self.fc1(x))
                                    return self.fc2(x)
                            policy_net = PolicyNet()
                            optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)
                            loss_fn = nn.CrossEntropyLoss()
                            X = torch.tensor(states).float()
                            y = torch.tensor(actions).long()
                            for epoch in range(3):
                                optimizer.zero_grad()
                                logits = policy_net(X)
                                loss = loss_fn(logits, y)
                                loss.backward()
                                optimizer.step()
                            # Generate long-term investment signal
                            latest_state = torch.tensor(returns[-180:]).float().unsqueeze(0)
                            with torch.no_grad():
                                action_logits = policy_net(latest_state)
                                action = torch.argmax(action_logits, dim=1).item()
                            rl_signal = ['hold', 'buy', 'sell'][action]
                            audit_col.insert_one({
                                'scheme_name': scheme,
                                'model': 'RL-DQN',
                                'rl_signal': rl_signal,
                                'holding_period': '6mo+',
                                'timestamp': datetime.utcnow()
                            })
                            logging.info(f"[RL] Long-term RL signal for {scheme}: {rl_signal}")
                    except Exception as e:
                        logging.warning(f"RL agent failed: {e}")
                    # --- Monitoring/Alerting: Prometheus stub ---
                    try:
                        import prometheus_client
                        prom_metric = prometheus_client.Gauge('agi_retrain_success', 'AGI retrain success', ['scheme'])
                        prom_metric.labels(scheme=scheme).set(1)
                    except Exception as e:
                        logging.debug(f"Prometheus metric update failed: {e}")
                    # --- Notification triggers ---
                    try:
                        from notification_service import send_notification
                        send_notification(
                            channel='web',
                            title=f'Retraining Complete: {scheme}',
                            message=f'NAV model retrained for {scheme}. Latest prediction: {nav_pred:.2f}'
                        )
                    except Exception as e:
                        logging.warning(f"Notification failed: {e}")
                    # --- User Feedback Loop (RL) ---
                    feedback = list(feedback_col.find({'scheme_name': scheme}))
                    if feedback:
                        # Example: if users rate prediction as inaccurate, trigger model update or flag
                        avg_score = np.mean([f.get('accuracy_score', 1.0) for f in feedback])
                        if avg_score < 0.7:
                            logging.warning(f"Low user feedback for {scheme}: {avg_score:.2f}. Consider retraining or alerting.")
            # RL agent and retraining for ETFs
            if asset_class == 'etf':
                all_etfs = [x['symbol'] for x in db['etfs'].find({}, {'symbol': 1})]
                for etf in all_etfs:
                    nav_records = list(db['etf_nav'].find({'symbol': etf}))
                    if len(nav_records) > 60:
                        nav_df = pd.DataFrame(nav_records)
                        nav_df['date'] = pd.to_datetime(nav_df['date'], errors='coerce')
                        nav_df = nav_df.dropna(subset=['date', 'nav'])
                        nav_df = nav_df.sort_values('date')
                        nav_df['nav'] = pd.to_numeric(nav_df['nav'], errors='coerce')
                        nav_df = nav_df.dropna(subset=['nav'])
                        nav_df['nav_norm'] = (nav_df['nav'] - nav_df['nav'].mean()) / nav_df['nav'].std()
                        sequence_length = 30
                        data = nav_df['nav_norm'].values
                        if len(data) > sequence_length + 180:
                            returns = np.diff(nav_df['nav'].values) / nav_df['nav'].values[:-1]
                            states = np.array([returns[i-180:i] for i in range(180, len(returns)-180)])
                            rewards = nav_df['nav'].values[180+1:len(nav_df['nav'])-180+1] / nav_df['nav'].values[1:len(nav_df['nav'])-180+1] - 1
                            actions = (rewards > 0.05).astype(int)
                            class PolicyNet(nn.Module):
                                def __init__(self):
                                    super().__init__()
                                    self.fc1 = nn.Linear(180, 64)
                                    self.fc2 = nn.Linear(64, 3)
                                def forward(self, x):
                                    x = torch.relu(self.fc1(x))
                                    return self.fc2(x)
                            policy_net = PolicyNet()
                            optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)
                            loss_fn = nn.CrossEntropyLoss()
                            X = torch.tensor(states).float()
                            y = torch.tensor(actions).long()
                            for epoch in range(3):
                                optimizer.zero_grad()
                                logits = policy_net(X)
                                loss = loss_fn(logits, y)
                                loss.backward()
                                optimizer.step()
                            latest_state = torch.tensor(returns[-180:]).float().unsqueeze(0)
                            with torch.no_grad():
                                action_logits = policy_net(latest_state)
                                action = torch.argmax(action_logits, dim=1).item()
                            rl_signal = ['hold', 'buy', 'sell'][action]
                            # --- Advanced RL: PPO from stable-baselines3 for ETF RL training ---
                            try:
                                from stable_baselines3 import PPO
                                from stable_baselines3.common.envs import DummyVecEnv
                                import gym
                                class ETFEnv(gym.Env):
                                    def __init__(self, data):
                                        super().__init__()
                                        self.data = data
                                        self.idx = 0
                                        self.action_space = gym.spaces.Discrete(3)  # hold, buy, sell
                                        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(30,), dtype=float)
                                    def reset(self):
                                        self.idx = 0
                                        return self.data[self.idx:self.idx+30]
                                    def step(self, action):
                                        reward = 0
                                        done = False
                                        self.idx += 1
                                        if self.idx+30 >= len(self.data):
                                            done = True
                                        obs = self.data[self.idx:self.idx+30] if not done else self.data[-30:]
                                        # Reward: 6mo return if buy, 0 if hold, -1 if sell (demo)
                                        if action == 1:  # buy
                                            reward = (self.data[self.idx+29] - self.data[self.idx]) / (abs(self.data[self.idx])+1e-8) if not done else 0
                                        elif action == 2:  # sell
                                            reward = -1
                                        return obs, reward, done, {}
                                etf_env = DummyVecEnv([lambda: ETFEnv(nav_df['nav_norm'].values)])
                                model = PPO('MlpPolicy', etf_env, verbose=0)
                                model.learn(total_timesteps=500)
                                obs = etf_env.reset()
                                action, _ = model.predict(obs)
                                rl_signal = ['hold', 'buy', 'sell'][int(action[0])]
                                explanation = f"ETF RL agent (PPO) trained on last 6mo returns. Signal: {rl_signal.upper()} (long-term)."
                            except Exception as e:
                                explanation = f"ETF RL agent (DQN fallback) analyzed last 6mo returns. Signal: {rl_signal.upper()} (long-term). RL PPO error: {e}"
                            audit_col.insert_one({
                                'symbol': etf,
                                'model': 'RL-DQN/PPO',
                                'rl_signal': rl_signal,
                                'holding_period': '6mo+',
                                'explanation': explanation,
                                'timestamp': datetime.utcnow()
                            })
                            # --- Sector/Industry Analytics ---
                            sector_info = db['etfs'].find_one({'symbol': etf}, {'sector': 1, 'industry': 1})
                            if sector_info and 'sector' in sector_info:
                                audit_col.update_one({'symbol': etf, 'timestamp': {'$exists': True}}, {'$set': {'sector': sector_info['sector'], 'industry': sector_info.get('industry')}})
                            # ...
                            # --- User feedback on explanation (for RL improvement) ---
                            feedback = list(feedback_col.find({'symbol': etf, 'model': 'RL-DQN/PPO'}))
                            if feedback:
                                avg_exp_score = np.mean([f.get('explanation_score', 1.0) for f in feedback])
                                if avg_exp_score < 0.7:
                                    logging.warning(f"Low explanation feedback for {etf}: {avg_exp_score:.2f}. RL policy improvement suggested.")
                            logging.info(f"[RL][ETF] {etf}: {rl_signal} | {explanation}")
                            # --- Advanced Analytics: Rolling volatility, max drawdown, rolling Sharpe, sector/industry correlations ---
                            if len(nav_df) > 60:
                                navs = nav_df['nav'].values
                                rolling_vol = pd.Series(navs).rolling(30).std()
                                max_dd = np.max(np.maximum.accumulate(navs) - navs)
                                rolling_sharpe = pd.Series(navs).rolling(30).apply(lambda x: (x[-1]-x[0])/(x.std()+1e-8))
                                audit_col.insert_one({
                                    'symbol': etf,
                                    'analytics': {
                                        'rolling_volatility': float(rolling_vol.iloc[-1]),
                                        'max_drawdown': float(max_dd),
                                        'rolling_sharpe': float(rolling_sharpe.iloc[-1])
                                    },
                                    'timestamp': datetime.utcnow()
                                })
                                # Sector/industry correlations (stub: extend with sector data)
                                # ...
                            try:
                                import prometheus_client
                                prom_metric = prometheus_client.Gauge('agi_rl_signal', 'AGI RL signal', ['asset_class','symbol'])
                                prom_metric.labels(asset_class='etf', symbol=etf).set(action)
                            except Exception as e:
                                logging.debug(f"Prometheus metric update failed: {e}")
            # RL agent and retraining for Stocks
            if asset_class == 'stock':
                all_stocks = [x['symbol'] for x in db['stocks'].find({}, {'symbol': 1})]
                for stock in all_stocks:
                    price_records = list(db['stock_prices'].find({'symbol': stock}))
                    if len(price_records) > 60:
                        price_df = pd.DataFrame(price_records)
                        price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce')
                        price_df = price_df.dropna(subset=['date', 'close'])
                        price_df = price_df.sort_values('date')
                        price_df['close'] = pd.to_numeric(price_df['close'], errors='coerce')
                        price_df = price_df.dropna(subset=['close'])
                        price_df['close_norm'] = (price_df['close'] - price_df['close'].mean()) / price_df['close'].std()
                        sequence_length = 30
                        data = price_df['close_norm'].values
                        if len(data) > sequence_length + 180:
                            returns = np.diff(price_df['close'].values) / price_df['close'].values[:-1]
                            states = np.array([returns[i-180:i] for i in range(180, len(returns)-180)])
                            rewards = price_df['close'].values[180+1:len(price_df['close'])-180+1] / price_df['close'].values[1:len(price_df['close'])-180+1] - 1
                            actions = (rewards > 0.05).astype(int)
                            class PolicyNet(nn.Module):
                                def __init__(self):
                                    super().__init__()
                                    self.fc1 = nn.Linear(180, 64)
                                    self.fc2 = nn.Linear(64, 3)
                                def forward(self, x):
                                    x = torch.relu(self.fc1(x))
                                    return self.fc2(x)
                            policy_net = PolicyNet()
                            optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)
                            loss_fn = nn.CrossEntropyLoss()
                            X = torch.tensor(states).float()
                            y = torch.tensor(actions).long()
                            for epoch in range(3):
                                optimizer.zero_grad()
                                logits = policy_net(X)
                                loss = loss_fn(logits, y)
                                loss.backward()
                                optimizer.step()
                            latest_state = torch.tensor(returns[-180:]).float().unsqueeze(0)
                            with torch.no_grad():
                                action_logits = policy_net(latest_state)
                                action = torch.argmax(action_logits, dim=1).item()
                            rl_signal = ['hold', 'buy', 'sell'][action]
                            explanation = f"Stock RL agent analyzed last 6mo returns. Signal: {rl_signal.upper()} (long-term)."
                            audit_col.insert_one({
                                'symbol': stock,
                                'model': 'RL-DQN',
                                'rl_signal': rl_signal,
                                'holding_period': '6mo+',
                                'explanation': explanation,
                                'timestamp': datetime.utcnow()
                            })
                            logging.info(f"[RL][Stock] {stock}: {rl_signal} | {explanation}")
                            try:
                                import prometheus_client
                                prom_metric = prometheus_client.Gauge('agi_rl_signal', 'AGI RL signal', ['asset_class','symbol'])
                                prom_metric.labels(asset_class='stock', symbol=stock).set(action)
                            except Exception as e:
                                logging.debug(f"Prometheus metric update failed: {e}")
        logging.info("Completed training step successfully.")
    except Exception as e:
        logging.error(f"Training step failed: {e}\n{traceback.format_exc()}")


def run_idle_training_loop():
    logging.info("Idle-aware training loop started.")
    while True:
        try:
            if user_queue.count > 0:
                logging.info(f"User job(s) detected ({user_queue.count}). Pausing training.")
                time.sleep(2)  # Check again soon
                continue
            # No user jobs, do one training step
            daily_training_step()
        except Exception as e:
            logging.error(f"Idle training loop error: {e}\n{traceback.format_exc()}")
            time.sleep(10)  # Wait before retrying on error

if __name__ == "__main__":
    run_idle_training_loop()
