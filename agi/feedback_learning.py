"""
User/admin feedback for continual learning in AGI++ mutual fund platform.
Allows feedback on errors, surprises, and feeds it into the learning loop.
"""
from pymongo import MongoClient
from datetime import datetime

mongo_client = MongoClient('mongodb://localhost:27017/')
feedback_log = mongo_client['agi_cache']['user_feedback_log']

# Feedback can be 'error', 'surprise', 'missed_opportunity', etc.
def log_feedback(scheme_code, prediction_id, feedback_type, comment, user_id=None):
    feedback_log.insert_one({
        'scheme_code': scheme_code,
        'prediction_id': prediction_id,
        'feedback_type': feedback_type,
        'comment': comment,
        'user_id': user_id,
        'timestamp': datetime.utcnow()
    })
    return True

def get_feedback(limit=100):
    return list(feedback_log.find().sort('timestamp', -1).limit(limit))
