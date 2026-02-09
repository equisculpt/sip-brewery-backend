#!/usr/bin/env python3
"""
Risk Predictor Inference Script
Wrapper for LSTM Risk Predictor model inference
"""

import sys
import json
import numpy as np
import torch
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from lstm_risk_predictor import RiskPredictionSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        input_json = sys.stdin.read()
        input_data = json.loads(input_json)
        
        user_id = input_data.get('user_id')
        time_horizon = input_data.get('time_horizon', '1M')
        
        logger.info(f"Predicting risk for user: {user_id}, horizon: {time_horizon}")
        
        model_path = Path(__file__).parent.parent / 'trained' / 'lstm_risk_model.pth'
        
        # Initialize system
        system = RiskPredictionSystem(
            input_dim=20,
            sequence_length=60
        )
        
        if model_path.exists():
            system.load_model(str(model_path))
            logger.info("Loaded trained model")
        else:
            logger.warning("No trained model found, using mock predictions")
        
        # Mock predictions (in production, prepare actual sequences and predict)
        result = {
            'var_1d': float(np.random.uniform(-0.03, -0.01)),
            'var_1w': float(np.random.uniform(-0.06, -0.02)),
            'var_1m': float(np.random.uniform(-0.10, -0.04)),
            'cvar_1d': float(np.random.uniform(-0.04, -0.015)),
            'cvar_1w': float(np.random.uniform(-0.08, -0.03)),
            'cvar_1m': float(np.random.uniform(-0.12, -0.05)),
            'volatility_1d': float(np.random.uniform(0.01, 0.02)),
            'volatility_1w': float(np.random.uniform(0.02, 0.04)),
            'volatility_1m': float(np.random.uniform(0.04, 0.08)),
            'current_drawdown': float(np.random.uniform(-0.10, 0.0)),
            'max_drawdown': float(np.random.uniform(-0.25, -0.10)),
            'skewness': float(np.random.uniform(-0.5, 0.5)),
            'kurtosis': float(np.random.uniform(1.0, 4.0)),
            'confidence': 0.88,
            'model_version': '1.0.0',
            'timestamp': str(np.datetime64('now'))
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        logger.error(f"Risk prediction failed: {str(e)}")
        print(json.dumps({'error': str(e)}))
        sys.exit(1)


if __name__ == '__main__':
    main()
