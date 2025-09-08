"""
Self-improving AGI research agent for mutual fund platform.
Learns from past extraction successes/failures to adapt source selection and parsing logic.
"""
from pymongo import MongoClient
from datetime import datetime

mongo_client = MongoClient('mongodb://localhost:27017/')
agent_log = mongo_client['agi_cache']['agi_agent_log']

# Log every extraction attempt, source, and result
def log_extraction_attempt(scheme_code, source, success, details=None):
    agent_log.insert_one({
        'scheme_code': scheme_code,
        'source': source,
        'success': success,
        'details': details,
        'timestamp': datetime.utcnow()
    })

# Adapt source selection based on past performance
def choose_best_source(scheme_code):
    # Aggregate past success rates
    pipeline = [
        {'$match': {'scheme_code': scheme_code}},
        {'$group': {'_id': '$source', 'success_rate': {'$avg': {'$cond': ['$success', 1, 0]}}}},
        {'$sort': {'success_rate': -1}}
    ]
    results = list(agent_log.aggregate(pipeline))
    if results:
        return results[0]['_id']
    return None
