"""
Background self-improving loop for AGI++ mutual fund platform.
Runs continual learning, backtesting, feedback review, causal analysis, and post-mortem generation whenever CPU is idle.
"""
import threading
import time
import logging
from continual_learning import retrain_models_from_errors
from backtesting_adaptation import backtest_and_adapt
from feedback_learning import get_feedback
from causal_inference import explain_error_causality
from natural_language_postmortem import generate_postmortem
from pymongo import MongoClient

logger = logging.getLogger("agi_self_improving_loop")

mongo_client = MongoClient('mongodb://localhost:27017/')
error_log = mongo_client['agi_cache']['prediction_error_log']

# Dummy CPU idle check (replace with actual system check)
def cpu_is_idle():
    import psutil
    return psutil.cpu_percent(interval=1) < 25  # Considered idle if CPU usage is below 25%

def self_improving_loop():
    while True:
        if cpu_is_idle():
            logger.info("CPU idle. Running self-improving loop.")
            # 1. Continual learning from errors
            retrain_models_from_errors()
            # 2. Backtesting and strategy adaptation
            backtest_and_adapt({}, {})  # Pass actual models and data
            # 3. Feedback review and causal/post-mortem generation
            feedbacks = get_feedback(limit=10)
            for fb in feedbacks:
                causal = explain_error_causality(fb['scheme_code'], fb['prediction_id'])
                generate_postmortem(fb['prediction_id'], adaptation_details=causal)
            logger.info("Self-improving loop completed. Sleeping...")
            time.sleep(120)  # Sleep for 2 minutes before next run
        else:
            logger.info("CPU busy. Waiting...")
            time.sleep(60)  # Check again in 1 minute

def start_background_loop():
    t = threading.Thread(target=self_improving_loop, daemon=True)
    t.start()
