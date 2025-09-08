"""
Natural-language post-mortem generator for AGI++ mutual fund platform.
Uses LLM to create trader-style explanations for major errors or model adaptations.
"""
from datetime import datetime
from pymongo import MongoClient
import logging

logger = logging.getLogger("postmortem")

try:
    from transformers import pipeline
    summarizer = pipeline("summarization")
except ImportError:
    summarizer = None

mongo_client = MongoClient('mongodb://localhost:27017/')
error_log = mongo_client['agi_cache']['prediction_error_log']
postmortem_log = mongo_client['agi_cache']['postmortem_log']

# Generate a post-mortem for a given error or adaptation event
def generate_postmortem(error_id, adaptation_details=None):
    doc = error_log.find_one({'_id': error_id})
    if not doc:
        return None
    # Compose context for explanation
    context = f"Prediction type: {doc['prediction_type']}. Prediction: {doc['prediction']}. Actual: {doc['actual']}. Error: {doc['error']}. Context: {doc['context']}."
    if adaptation_details:
        context += f" Adaptation details: {adaptation_details}."
    # Use LLM to generate explanation
    if summarizer:
        explanation = summarizer(context, max_length=120, min_length=40, do_sample=False)[0]['summary_text']
    else:
        explanation = f"Post-mortem: {context}"
    postmortem_log.insert_one({
        'error_id': error_id,
        'explanation': explanation,
        'timestamp': datetime.utcnow()
    })
    logger.info(f"Generated postmortem for error {error_id}.")
    return explanation
