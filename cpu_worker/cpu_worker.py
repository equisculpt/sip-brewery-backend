import os
import sys
import time
import logging
from rq import Queue
from redis import Redis
from agi import mutual_fund_ingest, notification_service

# Setup Redis connection for job queue
redis_conn = Redis(host='localhost', port=6379)
queue = Queue('agi-jobs', connection=redis_conn)

logging.basicConfig(level=logging.INFO)

import psutil
import threading

def run_on_cpu_or_gpu(job_func, *args, **kwargs):
    """Try to run job on CPU, monitor progress/system, offload to GPU if needed."""
    result = {'done': False, 'success': False}
    def job_wrapper():
        try:
            job_func(*args, **kwargs)
            result['success'] = True
        except Exception as e:
            logging.error(f'Job failed on CPU: {e}')
        finally:
            result['done'] = True
    t = threading.Thread(target=job_wrapper)
    t.start()
    # Monitor for time and CPU load
    start = time.time()
    timeout = 600  # 10 min per job
    while not result['done']:
        time.sleep(5)
        elapsed = time.time() - start
        cpu_load = psutil.cpu_percent(interval=1)
        if elapsed > timeout or cpu_load > 90:
            logging.warning(f'Job slow/hanging/overloaded (elapsed={elapsed:.1f}s, cpu={cpu_load}%), requeueing to GPU')
            # Requeue to GPU
            queue.enqueue('gpu_worker.run_rl_training', *args, **kwargs)
            return
    if result['success']:
        logging.info('Job completed on CPU')
    else:
        logging.warning('Job failed on CPU, requeueing to GPU')
        queue.enqueue('gpu_worker.run_rl_training', *args, **kwargs)

def main():
    while True:
        # Ingestion, web search, UI always on CPU
        schemes = mutual_fund_ingest.get_all_scheme_names()
        for scheme in schemes:
            # Try to train on CPU, offload to GPU if needed
            run_on_cpu_or_gpu(mutual_fund_ingest.train_rl, asset_type='mutual_fund', symbol=scheme)
        # Monitor results, update dashboard, send notifications
        # ...
        time.sleep(3600)  # Run every hour

if __name__ == '__main__':
    main()
