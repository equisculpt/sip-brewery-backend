"""
Prediction, outcome, and error logging for AGI++ continual learning.
Logs every forecast, regime call, scenario simulation, and actual outcome for self-improving AI.
"""
from datetime import datetime
from pymongo import MongoClient

mongo_client = MongoClient('mongodb://localhost:27017/')
error_log = mongo_client['agi_cache']['prediction_error_log']

def log_prediction(scheme_code, prediction_type, prediction, context):
    doc = {
        'scheme_code': scheme_code,
        'prediction_type': prediction_type,  # e.g., 'nav_forecast', 'regime', 'scenario'
        'prediction': prediction,
        'context': context,  # macro, features, model version, etc.
        'timestamp': datetime.utcnow(),
        'actual': None,
        'error': None,
        'evaluated': False
    }
    error_log.insert_one(doc)
    return doc

def log_outcome(scheme_code, prediction_id, actual):
    doc = error_log.find_one({'_id': prediction_id})
    if not doc:
        return None
    error = None
    if doc['prediction_type'] == 'nav_forecast':
        error = actual - doc['prediction']
    elif doc['prediction_type'] == 'regime':
        error = actual != doc['prediction']
    # Add more error types as needed
    error_log.update_one({'_id': prediction_id}, {'$set': {'actual': actual, 'error': error, 'evaluated': True}})
    return error

def get_recent_errors(limit=100):
    return list(error_log.find({'evaluated': True}).sort('timestamp', -1).limit(limit))
