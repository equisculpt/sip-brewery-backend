# AGI Microservice: Open Source, Continual Learning, Web Research Agent
# Requirements: fastapi, ray, uvicorn, transformers, sentence-transformers, weaviate-client, scrapy, beautifulsoup4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agi_microservice")

llm_pipeline = None
llm_model_name = None
embedder = None

from pymongo import MongoClient
mongo_client = MongoClient('mongodb://localhost:27017/')
aud_db = mongo_client['agi_audit']
aud_collection = aud_db['inference_logs']

def audit_log(action, user=None, model=None, input_data=None, output_data=None, error=None, note=None):
    from datetime import datetime
    aud = {
        "timestamp": datetime.utcnow(),
        "action": action,
        "user": user,
        "model": model,
        "input": input_data,
        "output": output_data,
        "error": error,
        "note": note
    }
    try:
        aud_collection.insert_one(aud)
    except Exception as e:
        logger.error(f"Audit log DB error: {e}")

# Unified notification interface stub
# Extend this to integrate with WhatsApp, web/app (websockets), email/SMS, etc.
from notification_service import notify_user as unified_notify_user
from portfolio_planner import plan_portfolio
from datetime import datetime
from pydantic import BaseModel

class PlanningRequest(BaseModel):
    user_id: str
    goal_amount: float
    goal_date: str  # ISO8601 date
    start_amount: float
    monthly_contribution: float
    risk_tolerance: str = "moderate"
    max_drawdown: float = None
    liquidity_needs: dict = None  # {month: amount}
    use_rl: bool = False

@app.post("/plan")
def autonomous_plan(req: PlanningRequest):
    """ASI-grade autonomous planning: event-driven, constraint-aware, RL-optional, notifies user, stores for dashboard."""
    goal_date = datetime.fromisoformat(req.goal_date)
    risk_map = {
        "conservative": (0.07, 0.08),
        "moderate": (0.12, 0.18),
        "aggressive": (0.16, 0.25)
    }
    mean, std = risk_map.get(req.risk_tolerance, (0.12, 0.18))
    plan, progress_checkpoints, max_dd, rl_results = plan_portfolio(
        goal_amount=req.goal_amount,
        goal_date=goal_date,
        start_amount=req.start_amount,
        monthly_contribution=req.monthly_contribution,
        risk_tolerance=req.risk_tolerance,
        returns_mean=mean,
        returns_std=std,
        max_drawdown=req.max_drawdown,
        liquidity_needs=req.liquidity_needs,
        use_rl=req.use_rl
    )
    # Store plan/progress for dashboard
    mongo_client['agi_cache']['user_plans'].update_one(
        {'user_id': req.user_id, 'goal_amount': req.goal_amount, 'goal_date': req.goal_date},
        {"$set": {
            'roadmap': plan.roadmap,
            'milestones': plan.milestones,
            'rationale': plan.rationale,
            'projected_value': plan.projected_value,
            'success_probability': plan.success_prob,
            'progress_checkpoints': progress_checkpoints,
            'max_drawdown': max_dd,
            'rl_results': rl_results,
            'last_updated': datetime.utcnow()
        }}, upsert=True)
    # Notify user on major milestones
    notify_user = unified_notify_user
    for ms in plan.milestones:
        if ms['year'] == datetime.utcnow().year:
            notify_user(req.user_id, 'web', f"Your portfolio plan for {ms['year']} projects value: {ms['projected_value']:.2f}")
    # --- Compliance Disclaimer and Logging ---
    disclaimer = ("This output is for research and educational purposes only. It is not investment advice. "
                  "Please consult your SEBI-registered Investment Adviser before making investment decisions. "
                  "As a SEBI/AMFI-registered Mutual Fund Distributor and Research Analyst, we do not provide personalized investment advice or portfolio suitability assessment.")
    mongo_client['agi_cache']['compliance_log'].insert_one({
        'user_id': req.user_id,
        'timestamp': datetime.utcnow(),
        'action': 'plan_api_response',
        'goal_amount': req.goal_amount,
        'goal_date': req.goal_date
    })
    return {
        "roadmap": plan.roadmap,
        "milestones": plan.milestones,
        "rationale": plan.rationale,
        "projected_value": plan.projected_value,
        "success_probability": plan.success_prob,
        "progress_checkpoints": progress_checkpoints,
        "max_drawdown": max_dd,
        "rl_results": rl_results,
        "disclaimer": disclaimer
    }

# --- Event-driven re-planning trigger ---
def trigger_replanning(user_id, goal_amount, goal_date):
    """Call this on major event (market/user change) to auto-update the plan."""
    user_plan = mongo_client['agi_cache']['user_plans'].find_one({'user_id': user_id, 'goal_amount': goal_amount, 'goal_date': goal_date})
    if user_plan:
        req = PlanningRequest(**{k: user_plan.get(k) for k in PlanningRequest.__fields__ if k in user_plan})
        autonomous_plan(req)

# --- Major Event Triggers ---
def trigger_major_event(event_type, user_id=None, meta=None):
    major_event_col = mongo_client['agi_cache']['major_events']
    event = {
        'type': event_type,
        'timestamp': datetime.utcnow(),
    }
    if user_id:
        event['user_id'] = user_id
    if meta:
        event['meta'] = meta
    major_event_col.insert_one(event)
    logger.info(f"Major event triggered: {event_type} for {user_id or 'global'}")
    # --- Event-based notification automation ---
    admin_notify = mongo_client['agi_cache']['admin_notify']
    subs = list(admin_notify.find({'event_type': event_type}))
    for sub in subs:
        try:
            notify_user(sub['admin_id'], sub.get('channel', 'web'), f"Major event: {event_type} for {user_id or 'global'} at {event['timestamp']}", meta)
        except Exception as e:
            logger.error(f"Admin notification failed: {e}")
    # --- User-facing event subscriptions ---
    user_subs = list(mongo_client['agi_cache']['user_notify'].find({'event_type': event_type}))
    for sub in user_subs:
        try:
            notify_user(sub['user_id'], sub.get('channel', 'web'), f"Update: {event_type} event for your account at {event['timestamp']}", meta)
        except Exception as e:
            logger.error(f"User notification failed: {e}")
    # Log notification
    notif_log = mongo_client['agi_cache']['notification_log']
    notif_log.insert_one({
        'event': event,
        'notified_admins': [s['admin_id'] for s in subs],
        'notified_users': [s['user_id'] for s in user_subs],
        'timestamp': datetime.utcnow()
    })
    # --- Anomaly detection and auto-escalation ---
    anomaly_log = mongo_client['agi_cache']['anomaly_log']
    # Example anomaly: too many events in short time (flood), repeated failures, or analytics outlier
    recent_events = list(major_event_col.find({'type': event_type, 'timestamp': {'$gte': datetime.utcnow().replace(microsecond=0, second=0, minute=datetime.utcnow().minute-10)}}))
    if len(recent_events) > 10:
        anomaly_log.insert_one({'event_type': event_type, 'level': 'warning', 'msg': 'High frequency of events', 'count': len(recent_events), 'timestamp': datetime.utcnow()})
        # Auto-escalate to super-admins
        super_admins = list(admin_notify.find({'event_type': event_type, 'role': 'super-admin'}))
        for sa in super_admins:
            try:
                notify_user(sa['admin_id'], sa.get('channel', 'web'), f"AUTO-ESCALATION: Anomaly detected for {event_type} (flood)", meta)
            except Exception as e:
                logger.error(f"Super-admin escalation failed: {e}")

def notify_user(user_id, channel, message, meta=None):
    try:
        unified_notify_user(user_id, channel, message, meta or {})
        logger.info(f"Notification sent to {user_id} on {channel}")
    except Exception as e:
        logger.error(f"Notification delivery failed for {user_id} on {channel}: {e}")

def load_llm(model_choice="distilgpt2"):
    global llm_pipeline, llm_model_name
    if llm_pipeline is not None and llm_model_name == model_choice:
        return True
    try:
        from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
        if model_choice == "mistralai/Mistral-7B-Instruct-v0.2":
            model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            llm = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            llm_pipeline = pipeline("text-generation", model=llm, tokenizer=tokenizer, device_map="auto")
            llm_model_name = model_choice
            logger.info("LLM loaded successfully (Mistral 7B).")
        else:
            model_name = "distilgpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            llm = AutoModelForCausalLM.from_pretrained(model_name)
            llm_pipeline = pipeline("text-generation", model=llm, tokenizer=tokenizer)
            llm_model_name = model_choice
            logger.info("LLM loaded successfully (distilgpt2).")
        return True
    except Exception as e:
        logger.error(f"LLM load failed: {e}")
        audit_log("load_llm_fail", model=model_choice, error=str(e))
        return False

def load_embedder():
    global embedder
    if embedder is not None:
        return True
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedder loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Embedder load failed: {e}")
        return False

@app.get("/health")
def health():
    # Prometheus metrics stub (extend with real counters/gauges as needed)
    return {"status": "ok", "uptime": "TODO", "metrics": {"inference_count": "TODO"}}

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 64

class EmbedRequest(BaseModel):
    texts: List[str]

import threading

@app.post("/inference")
def inference(req: InferenceRequest):
    from agi_research_agent import explain_recommendation
    if not load_llm():
        audit_log("inference_fail", input_data=req.prompt, error="LLM not available")
        return JSONResponse(status_code=503, content={"error": "LLM not available. Check logs for details."})
    result_holder = {}
    def run_inference():
        try:
            result = llm_pipeline(req.prompt, max_length=min(req.max_tokens, 64), do_sample=True)[0]["generated_text"]
            result_holder["result"] = result
        except Exception as e:
            result_holder["error"] = str(e)

    thread = threading.Thread(target=run_inference)
    thread.start()
    thread.join(timeout=10)
    def rule_based_fallback(prompt):
        prompt_lc = prompt.lower()
        if "sip" in prompt_lc and "mutual" in prompt_lc:
            return "A Systematic Investment Plan (SIP) is a way to invest in mutual funds by contributing small, regular amounts rather than a lump sum."
        if "mutual fund" in prompt_lc:
            return "A mutual fund pools money from many investors to invest in stocks, bonds, or other assets."
        if "portfolio" in prompt_lc:
            return "A portfolio is a collection of financial assets such as stocks, bonds, and mutual funds owned by an investor."
        if "nse" in prompt_lc or "bse" in prompt_lc:
            return "NSE (National Stock Exchange) and BSE (Bombay Stock Exchange) are major stock exchanges in India."
        if "lumpsum" in prompt_lc:
            return "A lump sum investment is a single, large investment made at one time, as opposed to smaller, regular investments like SIP."
        return "I'm an open-source finance assistant. Please ask about SIPs, mutual funds, portfolios, or stock markets!"

    rationale = None
    user_id = getattr(req, 'user_id', None)
    meta = getattr(req, 'meta', None)
    notify_channels = ["whatsapp", "web", "email"]

    if thread.is_alive():
        logger.error("LLM inference timed out (hung) for prompt: %s", req.prompt)
        canned = rule_based_fallback(req.prompt)
        rationale = explain_recommendation(
            user_id=user_id,
            prompt=req.prompt,
            model=llm_model_name,
            output=canned
        )
    if "error" in result_holder:
        logger.error(f"LLM inference failed: {result_holder['error']}")
        canned = rule_based_fallback(req.prompt)
        rationale = explain_recommendation(
            user_id=user_id,
            prompt=req.prompt,
            model=llm_model_name,
            output=canned
        )
        audit_log("inference_fail", input_data=req.prompt, output_data=canned, error=result_holder["error"], note="Rule-based fallback used.")
        for channel in notify_channels:
            notify_user(user_id, channel, f"Your AGI request failed. Fallback: {canned}\nRationale: {rationale.get('rationale', '')}", meta)
        return {"result": canned, "rationale": rationale.get("rationale", ""), "note": "LLM inference failed. Rule-based fallback used."}
    result = result_holder.get("result", "")
    rationale = explain_recommendation(
        user_id=user_id,
        prompt=req.prompt,
        model=llm_model_name,
        output=result
    )
    audit_log("inference", input_data=req.prompt, output_data=result, note="Success")
    for channel in notify_channels:
        notify_user(user_id, channel, f"Your AGI result: {result}\nRationale: {rationale.get('rationale', '')}", meta)
    return {"result": result, "rationale": rationale.get("rationale", "")}

@app.post("/embed")
def embed(req: EmbedRequest):
    if not load_embedder():
        audit_log("embed_fail", input_data=req.texts, error="Embedder not available")
        return JSONResponse(status_code=503, content={"error": "Embedder not available. Check logs for details."})
    try:
        vectors = embedder.encode(req.texts).tolist()
        audit_log("embed", input_data=req.texts, output_data="vectors", note="Success")
        return {"vectors": vectors}
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        audit_log("embed_fail", input_data=req.texts, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/admin/force_refresh_snapshot")
def force_refresh_snapshot(analysis_type: str, user_id: str, prompt: str, max_tokens: int = 128):
    """Admin endpoint to force-refresh snapshot (deletes cache for given analysis)."""
    cache_key = hashlib.sha256((analysis_type+prompt+user_id+str(max_tokens)).encode()).hexdigest()
    cache_col = mongo_client['agi_cache'][f'{analysis_type}_analysis']
    cache_col.delete_one({'_id': cache_key})
    return {"status": "ok", "msg": f"Snapshot for {analysis_type} deleted."}

from agi_self_improving_loop import start_background_loop

if __name__ == "__main__":
    start_background_loop()
    import uvicorn
    uvicorn.run("agi_microservice:app", host="0.0.0.0", port=8000, reload=True)
