"""
Causal inference engine for AGI++ mutual fund platform.
Explains not just 'what' but 'why' (root cause analysis for errors and outcomes).
Uses DoWhy or CausalNex for causal analysis.
"""
from pymongo import MongoClient
from datetime import datetime
import logging

logger = logging.getLogger("causal_inference")

try:
    import dowhy
    from dowhy import CausalModel
except ImportError:
    dowhy = None

mongo_client = MongoClient('mongodb://localhost:27017/')
error_log = mongo_client['agi_cache']['prediction_error_log']

# Example: root cause analysis for NAV forecast errors
def explain_error_causality(scheme_code, error_id):
    doc = error_log.find_one({'_id': error_id})
    if not doc:
        return None
    # Placeholder: fetch relevant data, build model
    # In production, use actual data and graph structure
    if dowhy:
        import pandas as pd
        # Example: outcome = nav_error, treatment = regime, confounders = macro, features
        data = pd.DataFrame([{
            'nav_error': doc.get('error', 0),
            'regime': doc['context'].get('regime', 'unknown'),
            'macro': str(doc['context'].get('macro', {})),
            'features': str(doc['context'].get('features', {})),
        }])
        model = CausalModel(
            data=data,
            treatment='regime',
            outcome='nav_error',
            common_causes=['macro', 'features']
        )
        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_matching")
        logger.info(f"Causal estimate: {estimate.value}")
        return {'causal_estimate': estimate.value, 'explanation': str(estimate)}
    else:
        return {'explanation': 'DoWhy not installed, causal analysis not available.'}
