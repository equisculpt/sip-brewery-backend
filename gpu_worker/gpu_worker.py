import os
import sys
import logging
from rq import Worker, Queue, Connection
from redis import Redis
from agi import idle_training_worker

redis_conn = Redis(host='localhost', port=6379)
logging.basicConfig(level=logging.INFO)

# RL/deep learning job to be called from queue
def run_rl_training(asset_type, symbol):
    logging.info(f'Running RL training for {asset_type} {symbol} on GPU')
    # Call the RL training logic from idle_training_worker or similar
    # idle_training_worker.train_rl(asset_type, symbol)
    # ...
    logging.info(f'RL training complete for {asset_type} {symbol}')

if __name__ == '__main__':
    with Connection(redis_conn):
        worker = Worker(['agi-jobs'])
        worker.work()
