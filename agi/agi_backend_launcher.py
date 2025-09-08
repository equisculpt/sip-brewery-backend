import subprocess
import sys
import os
import time
import signal

# Paths to scripts
WORKER = os.path.join(os.path.dirname(__file__), 'idle_training_worker.py')
DASHBOARD = os.path.join(os.path.dirname(__file__), 'agi_dashboard.py')

processes = []

try:
    # Start idle training worker (RL, retraining, feedback, analytics)
    p_worker = subprocess.Popen([sys.executable, WORKER])
    processes.append(p_worker)
    # Start dashboard (Streamlit) as subprocess
    p_dashboard = subprocess.Popen(['streamlit', 'run', DASHBOARD, '--server.headless=true'])
    processes.append(p_dashboard)
    # (Optional) Start Prometheus/Grafana exporters here if needed
    # e.g., subprocess.Popen(['prometheus', ...])
    print('All AGI backend services started. Press Ctrl+C to stop.')
    while True:
        time.sleep(10)
        # Monitor and restart if any process dies
        for i, p in enumerate(processes):
            if p.poll() is not None:
                print(f'Process {i} died, restarting...')
                if i == 0:
                    processes[i] = subprocess.Popen([sys.executable, WORKER])
                elif i == 1:
                    processes[i] = subprocess.Popen(['streamlit', 'run', DASHBOARD, '--server.headless=true'])
except KeyboardInterrupt:
    print('Shutting down all AGI backend services...')
    for p in processes:
        try:
            p.terminate()
        except Exception:
            pass
    sys.exit(0)
