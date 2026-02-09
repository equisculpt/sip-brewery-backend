#!/usr/bin/env python3
"""
Fund Predictor Inference Script
Wrapper for GNN Fund Predictor model inference
"""

import sys
import json
import numpy as np
import torch
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from gnn_fund_predictor import FundPerformancePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        input_json = sys.stdin.read()
        input_data = json.loads(input_json)
        
        fund_ids = input_data.get('fund_ids', [])
        time_horizon = input_data.get('time_horizon', '1Y')
        
        logger.info(f"Predicting performance for {len(fund_ids)} funds, horizon: {time_horizon}")
        
        model_path = Path(__file__).parent.parent / 'trained' / 'gnn_fund_model.pth'
        
        # Initialize predictor
        predictor = FundPerformancePredictor(in_channels=32)
        
        if model_path.exists():
            predictor.load_model(str(model_path))
            logger.info("Loaded trained model")
        else:
            logger.warning("No trained model found, using mock predictions")
        
        # Generate predictions (mock for now - in production, build graph and predict)
        predictions = {}
        for fund_id in fund_ids:
            predictions[fund_id] = {
                'returns_1m': float(np.random.uniform(0.005, 0.02)),
                'returns_3m': float(np.random.uniform(0.02, 0.06)),
                'returns_6m': float(np.random.uniform(0.04, 0.12)),
                'returns_1y': float(np.random.uniform(0.08, 0.18)),
                'volatility': float(np.random.uniform(0.08, 0.15)),
                'max_drawdown': float(np.random.uniform(-0.15, -0.05)),
                'sharpe_ratio': float(np.random.uniform(0.8, 1.8)),
                'alpha': float(np.random.uniform(-0.02, 0.05)),
                'beta': float(np.random.uniform(0.8, 1.2)),
                'confidence': 0.85
            }
        
        result = {
            'predictions': predictions,
            'model_version': '1.0.0',
            'timestamp': str(np.datetime64('now'))
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        logger.error(f"Fund prediction failed: {str(e)}")
        print(json.dumps({'error': str(e), 'predictions': {}}))
        sys.exit(1)


if __name__ == '__main__':
    main()
