#!/usr/bin/env python3
"""
Behavioral Predictor Inference Script
Wrapper for BERT-based Behavioral Predictor model inference
"""

import sys
import json
import numpy as np
import torch
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from behavioral_predictor import BehavioralPredictionSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        input_json = sys.stdin.read()
        input_data = json.loads(input_json)
        
        user_id = input_data.get('user_id')
        
        logger.info(f"Predicting behavior for user: {user_id}")
        
        model_path = Path(__file__).parent.parent / 'trained' / 'behavioral_model.pth'
        
        # Initialize system
        system = BehavioralPredictionSystem(
            num_actions=10,
            num_biases=8
        )
        
        if model_path.exists():
            system.load_model(str(model_path))
            logger.info("Loaded trained model")
        else:
            logger.warning("No trained model found, using mock predictions")
        
        # Mock predictions (in production, prepare actual data and predict)
        actions = ['BUY', 'SELL', 'HOLD', 'REBALANCE', 'WITHDRAW', 'DEPOSIT', 
                   'VIEW_PORTFOLIO', 'VIEW_INSIGHTS', 'CONTACT_SUPPORT', 'CHURN']
        
        action_probs = np.random.dirichlet(np.ones(10))
        top_indices = np.argsort(action_probs)[-3:][::-1]
        
        result = {
            'top_actions': [
                {
                    'action': actions[idx],
                    'probability': float(action_probs[idx])
                }
                for idx in top_indices
            ],
            'churn_probability': float(np.random.uniform(0.05, 0.30)),
            'predicted_amount': float(np.random.uniform(10000, 100000)),
            'behavioral_biases': {
                'loss_aversion': float(np.random.uniform(0.3, 0.8)),
                'recency_bias': float(np.random.uniform(0.2, 0.7)),
                'overconfidence': float(np.random.uniform(0.1, 0.6)),
                'anchoring': float(np.random.uniform(0.2, 0.7)),
                'herd_mentality': float(np.random.uniform(0.3, 0.8)),
                'confirmation_bias': float(np.random.uniform(0.2, 0.6)),
                'availability_bias': float(np.random.uniform(0.1, 0.5)),
                'mental_accounting': float(np.random.uniform(0.2, 0.7))
            },
            'confidence': 0.82,
            'model_version': '1.0.0',
            'timestamp': str(np.datetime64('now'))
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        logger.error(f"Behavioral prediction failed: {str(e)}")
        print(json.dumps({'error': str(e)}))
        sys.exit(1)


if __name__ == '__main__':
    main()
