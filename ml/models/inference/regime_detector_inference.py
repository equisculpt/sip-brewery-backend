#!/usr/bin/env python3
"""
Market Regime Detector Inference Script
Wrapper for HMM + NN Market Regime Detector model inference
"""

import sys
import json
import numpy as np
import torch
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from market_regime_detector import MarketRegimeDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        input_json = sys.stdin.read()
        input_data = json.loads(input_json) if input_json.strip() else {}
        
        logger.info("Detecting current market regime")
        
        model_path = Path(__file__).parent.parent / 'trained' / 'regime_detector_model.pth'
        
        # Initialize detector
        detector = MarketRegimeDetector(input_dim=20)
        
        if model_path.exists():
            detector.load_model(str(model_path))
            logger.info("Loaded trained model")
        else:
            logger.warning("No trained model found, using mock predictions")
        
        # Mock prediction (in production, fetch real market data and predict)
        regimes = ['BULL', 'BEAR', 'SIDEWAYS', 'VOLATILE']
        regime_probs = np.random.dirichlet(np.ones(4))
        current_regime_idx = np.argmax(regime_probs)
        
        # Generate transition probabilities
        transition_probs = {}
        for regime in regimes:
            transition_probs[regime] = float(np.random.uniform(0.1, 0.4))
        # Normalize
        total = sum(transition_probs.values())
        transition_probs = {k: v/total for k, v in transition_probs.items()}
        
        result = {
            'regime': regimes[current_regime_idx],
            'regime_id': int(current_regime_idx),
            'confidence': float(regime_probs[current_regime_idx]),
            'probabilities': {
                regimes[i]: float(regime_probs[i])
                for i in range(4)
            },
            'expected_duration_days': float(np.random.uniform(20, 90)),
            'transition_probabilities': transition_probs,
            'model_version': '1.0.0',
            'timestamp': str(np.datetime64('now'))
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        logger.error(f"Regime detection failed: {str(e)}")
        print(json.dumps({'error': str(e), 'regime': 'UNKNOWN'}))
        sys.exit(1)


if __name__ == '__main__':
    main()
