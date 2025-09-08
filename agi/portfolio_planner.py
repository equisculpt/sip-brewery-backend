from datetime import datetime, timedelta
import numpy as np

class PlanningResult:
    def __init__(self, roadmap, milestones, rationale, projected_value, success_prob):
        self.roadmap = roadmap
        self.milestones = milestones
        self.rationale = rationale
        self.projected_value = projected_value
        self.success_prob = success_prob


def plan_portfolio(goal_amount, goal_date, start_amount, monthly_contribution, risk_tolerance, returns_mean=0.12, returns_std=0.18, max_years=30, max_drawdown=None, liquidity_needs=None, use_rl=False):
    """
    Advanced planner: supports constraints (max drawdown, liquidity), RL-based optimization, and returns progress checkpoints.
    Event-driven re-planning can be triggered externally by calling this function with updated params.
    """
    today = datetime.utcnow()
    months = max(1, (goal_date.year - today.year) * 12 + (goal_date.month - today.month))
    steps = months
    simulations = 1000
    projections = []
    drawdowns = []
    for _ in range(simulations):
        value = start_amount
        history = [value]
        peak = value
        dd = 0
        for m in range(steps):
            monthly_return = np.random.normal(returns_mean/12, returns_std/np.sqrt(12))
            value = value * (1 + monthly_return) + monthly_contribution
            peak = max(peak, value)
            dd = min(dd, (value - peak) / peak)
            # Liquidity constraint: simulate required cash withdrawals
            if liquidity_needs and m in liquidity_needs:
                value -= liquidity_needs[m]
            history.append(value)
        projections.append(history)
        drawdowns.append(dd)
    projections = np.array(projections)
    projected_value = np.median(projections[:,-1])
    prob_success = float((projections[:,-1] >= goal_amount).mean())
    max_dd = float(np.percentile(drawdowns, 95))
    # RL-based optimization using stable-baselines3 PPO and a real Gym environment
    if use_rl:
        try:
            from stable_baselines3 import PPO
            import gym
            import numpy as np
            class PortfolioPlanningEnv(gym.Env):
                def __init__(self, goal_amount, steps, start_amount, monthly_contribution, returns_mean, returns_std, max_drawdown=None, liquidity_needs=None):
                    super().__init__()
                    self.goal_amount = goal_amount
                    self.steps = steps
                    self.start_amount = start_amount
                    self.returns_mean = returns_mean
                    self.returns_std = returns_std
                    self.max_drawdown = max_drawdown
                    self.liquidity_needs = liquidity_needs or {}
                    self.action_space = gym.spaces.Box(low=0, high=2*monthly_contribution, shape=(1,), dtype=np.float32)  # Action: monthly contribution
                    self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)  # [current_value, step, peak]
                    self.reset()
                def reset(self):
                    self.value = self.start_amount
                    self.step_idx = 0
                    self.peak = self.value
                    self.done = False
                    return np.array([self.value, self.step_idx, self.peak], dtype=np.float32)
                def step(self, action):
                    contrib = float(np.clip(action[0], 0, 2*monthly_contribution))
                    monthly_return = np.random.normal(self.returns_mean/12, self.returns_std/np.sqrt(12))
                    self.value = self.value * (1 + monthly_return) + contrib
                    if self.step_idx in self.liquidity_needs:
                        self.value -= self.liquidity_needs[self.step_idx]
                    self.peak = max(self.peak, self.value)
                    dd = (self.value - self.peak) / self.peak
                    reward = -abs(self.goal_amount - self.value) / self.goal_amount
                    if self.max_drawdown is not None and dd < self.max_drawdown:
                        reward -= 1  # Penalize for exceeding drawdown
                    self.step_idx += 1
                    done = self.step_idx >= self.steps
                    obs = np.array([self.value, self.step_idx, self.peak], dtype=np.float32)
                    return obs, reward, done, {}
            env = PortfolioPlanningEnv(goal_amount, steps, start_amount, monthly_contribution, returns_mean, returns_std, max_drawdown, liquidity_needs)
            model = PPO('MlpPolicy', env, verbose=0)
            model.learn(total_timesteps=5000)
            obs = env.reset()
            rl_roadmap = []
            for i in range(steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                rl_roadmap.append({'month': i+1, 'projected_value': float(obs[0]), 'action_contribution': float(action[0])})
                if done:
                    break
            rl_results = {'rl_roadmap': rl_roadmap, 'final_value': float(obs[0]), 'reward': float(reward)}
        except Exception as e:
            rl_results = {'error': str(e)}
    else:
        rl_results = None
    # Milestones: every year
    milestones = []
    for y in range(1, (steps//12)+1):
        milestone = {
            'year': today.year + y,
            'projected_value': float(np.median(projections[:,y*12]))
        }
        milestones.append(milestone)
    roadmap = [
        {'month': i+1, 'projected_value': float(np.median(projections[:,i]))}
        for i in range(0, steps+1, max(1, steps//10))
    ]
    progress_checkpoints = [
        {'month': i+1, 'min': float(np.percentile(projections[:,i], 5)), 'median': float(np.percentile(projections[:,i], 50)), 'max': float(np.percentile(projections[:,i], 95))}
        for i in range(0, steps+1, max(1, steps//12))
    ]
    rationale = f"Plan assumes mean annual return {returns_mean*100:.1f}%, volatility {returns_std*100:.1f}%, monthly contribution {monthly_contribution}, starting from {start_amount}. Max drawdown constraint: {max_drawdown}, liquidity needs: {liquidity_needs}, RL: {use_rl}."
    return PlanningResult(roadmap, milestones, rationale, float(projected_value), float(prob_success)), progress_checkpoints, max_dd, rl_results
