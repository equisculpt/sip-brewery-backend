from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from redis import Redis
from rq import Queue
import time

app = FastAPI()
redis_conn = Redis()
cpu_queue = Queue('cpu', connection=redis_conn)
gpu_queue = Queue('gpu', connection=redis_conn)

INVESTMENT_KEYWORDS = ["top fund", "best fund", "sip", "invest", "new sip", "buy", "purchase", "lump sum", "execute"]

class AIRequest(BaseModel):
    user_id: str
    message: str

def classify_request(message: str):
    if any(word in message.lower() for word in INVESTMENT_KEYWORDS):
        return "investment"
    return "analysis"

def get_eta(req_type: str, cpu_queue_len: int):
    if req_type == "investment":
        return min(5, 1 + cpu_queue_len // 2)
    return min(15, 5 + cpu_queue_len)

def notify_user(user_id: str, msg: str):
    # TODO: Integrate with WhatsApp, Twilio, or your notification system
    print(f"Notify {user_id}: {msg}")

def process_on_cpu(message: str):
    # TODO: Replace with real logic (simulate slow CPU work)
    time.sleep(60)  # Simulate 1 min processing
    return f"CPU result for: {message}"

def process_on_gpu(message: str):
    # TODO: Replace with real logic (simulate fast GPU work)
    time.sleep(10)  # Simulate 10 sec processing
    return f"GPU result for: {message}"

def escalate_to_gpu(user_id, message):
    notify_user(user_id, "This is taking longer than usual, we’re accelerating your analysis for a faster response.")
    gpu_job = gpu_queue.enqueue(process_on_gpu, message)
    result = gpu_job.result or gpu_job.wait(timeout=300)  # Wait up to 5 min
    return result

def cpu_worker(user_id, message, deadline):
    start = time.time()
    try:
        result = process_on_cpu(message)
        elapsed = time.time() - start
        if elapsed > deadline:
            return escalate_to_gpu(user_id, message)
        return result
    except Exception as e:
        return escalate_to_gpu(user_id, message)

@app.post("/ai")
def handle_ai_request(req: AIRequest, background_tasks: BackgroundTasks):
    req_type = classify_request(req.message)
    cpu_queue_len = len(cpu_queue.jobs)
    eta = get_eta(req_type, cpu_queue_len)
    deadline = 300 if req_type == "investment" else 900  # 5 min or 15 min

    notify_user(req.user_id, f"We have received your request and SIPBrew AGI is working on it. You will get a response in approx {eta} minutes.")
    job = cpu_queue.enqueue(cpu_worker, req.user_id, req.message, deadline, job_timeout=deadline+60, result_ttl=deadline+300)
    return {"status": "queued", "eta_minutes": eta, "job_id": job.id}
