/**
 * 🤖 AI ROUTES
 * 
 * Advanced AI endpoints for mutual fund analysis, predictions, and recommendations
 * Integrates with continuous learning engine and real-time data
 * 
 * @author AI Founder with 100+ years team experience
 * @version 1.0.0 - Financial ASI
 */

const express = require('express');
const router = express.Router();
const aiController = require('../controllers/aiController');
const { authenticate } = require('../middleware/auth');
const { validateRequest } = require('../middleware/validation');
const { rateLimit } = require('../middleware/rateLimit');

// Apply authentication to all AI routes
router.use(authenticate);

// Apply rate limiting for AI endpoints
const aiRateLimit = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 30, // 30 requests per minute
  message: 'Too many AI requests, please try again later'
});

router.use(aiRateLimit);

// ===== FUND ANALYSIS ENDPOINTS =====

// POST /api/ai/analyze/fund - Analyze single mutual fund
router.post('/analyze/fund', 
  validateRequest({
    body: {
      fundCode: { type: 'string', required: true },
      includeHistory: { type: 'boolean', default: true },
      analysisType: { type: 'string', enum: ['basic', 'comprehensive'], default: 'comprehensive' }
    }
  }),
  aiController.analyzeSingleFund
);

// POST /api/ai/analyze/funds - Analyze multiple mutual funds
router.post('/analyze/funds',
  validateRequest({
    body: {
      fundCodes: { type: 'array', required: true, minLength: 1, maxLength: 20 },
      includeHistory: { type: 'boolean', default: true },
      analysisType: { type: 'string', enum: ['basic', 'comprehensive'], default: 'basic' }
    }
  }),
  aiController.analyzeMultipleFunds
);

// POST /api/ai/analyze/portfolio - Analyze portfolio composition
router.post('/analyze/portfolio',
  validateRequest({
    body: {
      portfolioId: { type: 'string', required: true },
      funds: { type: 'array', required: true },
      userProfile: { type: 'object', required: true }
    }
  }),
  aiController.analyzePortfolio
);

// ===== PREDICTION ENDPOINTS =====

// POST /api/ai/predict/nav - Predict NAV for fund
router.post('/predict/nav',
  validateRequest({
    body: {
      fundCode: { type: 'string', required: true },
      timeHorizon: { type: 'string', enum: ['1M', '3M', '6M', '1Y'], default: '3M' },
      includeConfidence: { type: 'boolean', default: true }
    }
  }),
  aiController.predictNAV
);

// POST /api/ai/predict/performance - Predict fund performance
router.post('/predict/performance',
  validateRequest({
    body: {
      fundCode: { type: 'string', required: true },
      timeHorizon: { type: 'string', enum: ['1M', '3M', '6M', '1Y', '3Y', '5Y'], default: '1Y' },
      marketConditions: { type: 'object' }
    }
  }),
  aiController.predictPerformance
);

// POST /api/ai/predict/risk - Predict risk assessment
router.post('/predict/risk',
  validateRequest({
    body: {
      fundCode: { type: 'string', required: true },
      userRiskProfile: { type: 'string', enum: ['conservative', 'moderate', 'balanced', 'aggressive', 'veryAggressive'] }
    }
  }),
  aiController.predictRisk
);

// ===== RECOMMENDATION ENDPOINTS =====

// POST /api/ai/recommend/funds - Get fund recommendations
router.post('/recommend/funds',
  validateRequest({
    body: {
      userProfile: { type: 'object', required: true },
      investmentAmount: { type: 'number', required: true, min: 500 },
      investmentGoal: { type: 'string', required: true },
      timeHorizon: { type: 'string', required: true },
      maxRecommendations: { type: 'number', default: 10, max: 20 }
    }
  }),
  aiController.recommendFunds
);

// POST /api/ai/recommend/portfolio - Get portfolio recommendations
router.post('/recommend/portfolio',
  validateRequest({
    body: {
      userProfile: { type: 'object', required: true },
      currentPortfolio: { type: 'array' },
      targetAmount: { type: 'number', required: true },
      rebalanceOnly: { type: 'boolean', default: false }
    }
  }),
  aiController.recommendPortfolio
);

// POST /api/ai/recommend/sip - Get SIP recommendations
router.post('/recommend/sip',
  validateRequest({
    body: {
      fundCode: { type: 'string', required: true },
      monthlyAmount: { type: 'number', required: true, min: 500 },
      duration: { type: 'number', required: true, min: 12 },
      userProfile: { type: 'object', required: true }
    }
  }),
  aiController.recommendSIP
);

// ===== MARKET ANALYSIS ENDPOINTS =====

// GET /api/ai/market/sentiment - Get current market sentiment
router.get('/market/sentiment', aiController.getMarketSentiment);

// GET /api/ai/market/timing - Get market timing analysis
router.get('/market/timing', aiController.getMarketTiming);

// GET /api/ai/market/insights - Get AI-generated market insights
router.get('/market/insights', aiController.getMarketInsights);

// GET /api/ai/market/trends - Get market trend analysis
router.get('/market/trends', aiController.getMarketTrends);

// ===== LEARNING & MONITORING ENDPOINTS =====

// GET /api/ai/learning/status - Get continuous learning status
router.get('/learning/status', aiController.getLearningStatus);

// GET /api/ai/learning/metrics - Get learning performance metrics
router.get('/learning/metrics', aiController.getLearningMetrics);

// POST /api/ai/learning/feedback - Provide feedback for learning
router.post('/learning/feedback',
  validateRequest({
    body: {
      predictionId: { type: 'string', required: true },
      actualOutcome: { type: 'object', required: true },
      feedback: { type: 'string', enum: ['accurate', 'inaccurate', 'partially_accurate'] }
    }
  }),
  aiController.provideLearningFeedback
);

// ===== SYSTEM ENDPOINTS =====

// GET /api/ai/health - Get comprehensive AI service health
router.get('/health', aiController.getAIHealth);

// GET /api/ai/metrics - Get AI service metrics
router.get('/metrics', aiController.getAIMetrics);

// GET /api/ai/status - Get AI service status
router.get('/status', aiController.getAIStatus);

// ===== TESTING & DEVELOPMENT ENDPOINTS =====

// GET /api/ai/test/fund/:schemeCode - Test fund data fetching
router.get('/test/fund/:schemeCode', aiController.testFundData);

// POST /api/ai/test/prediction - Test prediction accuracy
router.post('/test/prediction',
  validateRequest({
    body: {
      testType: { type: 'string', required: true },
      parameters: { type: 'object', required: true }
    }
  }),
  aiController.testPrediction
);

// GET /api/ai/test/models - Test AI model performance
router.get('/test/models', aiController.testModels);

// Backtesting Routes
router.post('/backtest/run', 
  authenticate,
  aiRateLimit,
  aiController.runBacktest
);

router.get('/backtest/results/:strategyName',
  authenticate,
  aiRateLimit,
  aiController.getBacktestResults
);

// Performance Dashboard Routes
router.get('/performance/dashboard',
  authenticate,
  aiRateLimit,
  aiController.getPerformanceDashboard
);

router.get('/performance/models/compare',
  authenticate,
  aiRateLimit,
  aiController.compareModels
);

router.get('/performance/alerts',
  authenticate,
  aiRateLimit,
  aiController.getPerformanceAlerts
);

// Historical Data Routes
router.get('/data/historical/:symbol',
  authenticate,
  aiRateLimit,
  aiController.getHistoricalData
);

router.get('/data/realtime/market',
  authenticate,
  aiRateLimit,
  aiController.getRealTimeMarketData
);

// Model Management Routes
router.post('/models/register',
  authenticate,
  aiRateLimit,
  aiController.registerModel
);

router.get('/models/performance/:modelName',
  authenticate,
  aiRateLimit,
  aiController.getModelPerformance
);

// Testing and Development Routes
router.get('/test/health', aiController.testHealth);
router.post('/test/prediction', aiController.testPrediction);
router.get('/test/models', aiController.testModels);
router.post('/test/backtest', aiController.testBacktest);
router.get('/test/performance', aiController.testPerformance);

module.exports = router;