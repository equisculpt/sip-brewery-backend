/**
 * 🚀 UNIFIED ASI ROUTES
 * 
 * Single set of routes that handles ALL intelligence requests
 * Everything goes through ASI Master Engine
 * Replaces separate AI/AGI/ASI route files
 * 
 * @author Universe-Class ASI Architect
 * @version 1.0.0 - Unified ASI Architecture
 */

const express = require('express');
const router = express.Router();
const { UnifiedASIController } = require('../controllers/unifiedASIController');
const { authenticateUser } = require('../middleware/auth');
const { validateRequest } = require('../middleware/validation');
const { rateLimiter } = require('../middleware/rateLimiter');

// Initialize controller
const asiController = new UnifiedASIController();

// Initialize ASI on first route access
router.use(async (req, res, next) => {
  try {
    await asiController.ensureInitialized();
    next();
  } catch (error) {
    res.status(503).json({
      success: false,
      message: 'ASI system is initializing, please try again',
      error: error.message
    });
  }
});

/**
 * 🧠 UNIVERSAL ASI ENDPOINT
 * Single endpoint that can handle ANY intelligence request
 */
router.post('/process', 
  authenticateUser,
  rateLimiter({ windowMs: 60000, max: 100 }), // 100 requests per minute
  validateRequest([
    { field: 'type', type: 'string', required: false },
    { field: 'data', type: 'object', required: false },
    { field: 'parameters', type: 'object', required: false },
    { field: 'urgency', type: 'string', enum: ['relaxed', 'normal', 'urgent', 'critical'], required: false },
    { field: 'precision', type: 'string', enum: ['low', 'standard', 'high', 'critical'], required: false }
  ]),
  asiController.processAnyRequest.bind(asiController)
);

/**
 * 📊 FUND ANALYSIS ENDPOINTS
 */

// Analyze single fund
router.get('/fund/:fundCode/analyze',
  authenticateUser,
  rateLimiter({ windowMs: 60000, max: 50 }),
  asiController.analyzeFund.bind(asiController)
);

// Analyze fund with custom parameters
router.post('/fund/analyze',
  authenticateUser,
  rateLimiter({ windowMs: 60000, max: 50 }),
  validateRequest([
    { field: 'fundCode', type: 'string', required: true },
    { field: 'includeHistory', type: 'boolean', required: false },
    { field: 'depth', type: 'string', enum: ['basic', 'standard', 'comprehensive'], required: false },
    { field: 'includeRecommendations', type: 'boolean', required: false }
  ]),
  asiController.analyzeFund.bind(asiController)
);

/**
 * 🔮 PREDICTION ENDPOINTS
 */

// General prediction endpoint
router.post('/predict',
  authenticateUser,
  rateLimiter({ windowMs: 60000, max: 30 }),
  validateRequest([
    { field: 'predictionType', type: 'string', enum: ['nav_prediction', 'performance', 'risk', 'market_movement'], required: true },
    { field: 'data', type: 'object', required: true },
    { field: 'horizon', type: 'string', required: false },
    { field: 'confidence', type: 'number', min: 0.5, max: 0.99, required: false }
  ]),
  asiController.predict.bind(asiController)
);

// NAV prediction
router.post('/predict/nav',
  authenticateUser,
  rateLimiter({ windowMs: 60000, max: 30 }),
  validateRequest([
    { field: 'fundCode', type: 'string', required: true },
    { field: 'history', type: 'array', required: false },
    { field: 'horizon', type: 'string', required: false }
  ]),
  (req, res) => {
    req.body.predictionType = 'nav_prediction';
    req.body.data = {
      fundCode: req.body.fundCode,
      history: req.body.history
    };
    return asiController.predict(req, res);
  }
);

/**
 * 🎯 PORTFOLIO OPTIMIZATION ENDPOINTS
 */

// Portfolio optimization
router.post('/portfolio/optimize',
  authenticateUser,
  rateLimiter({ windowMs: 300000, max: 10 }), // 10 requests per 5 minutes (computationally intensive)
  validateRequest([
    { field: 'funds', type: 'array', required: true },
    { field: 'constraints', type: 'object', required: false },
    { field: 'riskProfile', type: 'string', enum: ['conservative', 'moderate', 'aggressive'], required: false },
    { field: 'method', type: 'string', enum: ['modern_portfolio', 'black_litterman', 'quantum_inspired'], required: false }
  ]),
  asiController.optimizePortfolio.bind(asiController)
);

// SIP optimization
router.post('/sip/optimize',
  authenticateUser,
  rateLimiter({ windowMs: 300000, max: 15 }),
  validateRequest([
    { field: 'amount', type: 'number', min: 500, required: true },
    { field: 'duration', type: 'string', required: true },
    { field: 'riskProfile', type: 'string', required: false },
    { field: 'goals', type: 'array', required: false }
  ]),
  (req, res) => {
    req.body.type = 'sip_optimization';
    req.body.data = {
      amount: req.body.amount,
      duration: req.body.duration,
      riskProfile: req.body.riskProfile,
      goals: req.body.goals
    };
    return asiController.processAnyRequest(req, res);
  }
);

/**
 * 🧠 BEHAVIORAL ANALYSIS ENDPOINTS
 */

// Behavioral analysis
router.post('/behavior/analyze',
  authenticateUser,
  rateLimiter({ windowMs: 60000, max: 20 }),
  validateRequest([
    { field: 'marketData', type: 'object', required: false },
    { field: 'userProfile', type: 'object', required: false },
    { field: 'transactionHistory', type: 'array', required: false },
    { field: 'includeBiasDetection', type: 'boolean', required: false },
    { field: 'culturalFactors', type: 'string', required: false }
  ]),
  asiController.analyzeBehavior.bind(asiController)
);

// Market sentiment analysis
router.post('/sentiment/analyze',
  authenticateUser,
  rateLimiter({ windowMs: 60000, max: 30 }),
  validateRequest([
    { field: 'timeframe', type: 'string', required: false },
    { field: 'sectors', type: 'array', required: false }
  ]),
  (req, res) => {
    req.body.type = 'market_sentiment';
    req.body.data = {
      timeframe: req.body.timeframe,
      sectors: req.body.sectors
    };
    return asiController.processAnyRequest(req, res);
  }
);

/**
 * ⚡ QUANTUM OPTIMIZATION ENDPOINTS
 */

// Quantum portfolio optimization
router.post('/quantum/optimize',
  authenticateUser,
  rateLimiter({ windowMs: 600000, max: 5 }), // 5 requests per 10 minutes (very computationally intensive)
  validateRequest([
    { field: 'expectedReturns', type: 'array', required: true },
    { field: 'covarianceMatrix', type: 'array', required: true },
    { field: 'constraints', type: 'object', required: false },
    { field: 'algorithm', type: 'string', enum: ['QAOA', 'VQE', 'quantum_annealing'], required: false }
  ]),
  asiController.quantumOptimize.bind(asiController)
);

// Quantum vs Classical comparison
router.post('/quantum/compare',
  authenticateUser,
  rateLimiter({ windowMs: 600000, max: 3 }),
  validateRequest([
    { field: 'expectedReturns', type: 'array', required: true },
    { field: 'covarianceMatrix', type: 'array', required: true },
    { field: 'constraints', type: 'object', required: false }
  ]),
  (req, res) => {
    req.body.type = 'quantum_comparison';
    req.body.parameters = { compareWithClassical: true };
    return asiController.quantumOptimize(req, res);
  }
);

/**
 * 🎓 AUTONOMOUS LEARNING ENDPOINTS
 */

// Autonomous learning
router.post('/learn/autonomous',
  authenticateUser,
  rateLimiter({ windowMs: 300000, max: 5 }),
  validateRequest([
    { field: 'data', type: 'object', required: false },
    { field: 'learningType', type: 'string', enum: ['meta_learning', 'curriculum_learning', 'active_learning'], required: false }
  ]),
  asiController.autonomousLearn.bind(asiController)
);

// Meta learning
router.post('/learn/meta',
  authenticateUser,
  rateLimiter({ windowMs: 300000, max: 5 }),
  validateRequest([
    { field: 'tasks', type: 'array', required: true }
  ]),
  (req, res) => {
    req.body.type = 'meta_learning';
    req.body.data = { tasks: req.body.tasks };
    return asiController.processAnyRequest(req, res);
  }
);

/**
 * 📈 MARKET ANALYSIS ENDPOINTS
 */

// Comprehensive market analysis
router.post('/market/analyze',
  authenticateUser,
  rateLimiter({ windowMs: 60000, max: 20 }),
  validateRequest([
    { field: 'marketData', type: 'object', required: false },
    { field: 'timeframe', type: 'string', required: false },
    { field: 'sectors', type: 'array', required: false }
  ]),
  asiController.analyzeMarket.bind(asiController)
);

// Market timing analysis
router.post('/market/timing',
  authenticateUser,
  rateLimiter({ windowMs: 60000, max: 15 }),
  validateRequest([
    { field: 'strategy', type: 'string', required: false },
    { field: 'riskTolerance', type: 'string', required: false }
  ]),
  (req, res) => {
    req.body.type = 'market_timing';
    req.body.data = {
      strategy: req.body.strategy,
      riskTolerance: req.body.riskTolerance
    };
    return asiController.processAnyRequest(req, res);
  }
);

/**
 * 🔄 BACKTESTING ENDPOINTS
 */

// Strategy backtesting
router.post('/backtest/strategy',
  authenticateUser,
  rateLimiter({ windowMs: 300000, max: 10 }),
  validateRequest([
    { field: 'strategy', type: 'object', required: true },
    { field: 'timeframe', type: 'string', required: false },
    { field: 'monteCarloRuns', type: 'number', min: 100, max: 5000, required: false }
  ]),
  asiController.backtestStrategy.bind(asiController)
);

// Walk-forward analysis
router.post('/backtest/walkforward',
  authenticateUser,
  rateLimiter({ windowMs: 300000, max: 5 }),
  validateRequest([
    { field: 'strategy', type: 'object', required: true },
    { field: 'windowSize', type: 'string', required: false },
    { field: 'stepSize', type: 'string', required: false }
  ]),
  (req, res) => {
    req.body.type = 'walkforward_analysis';
    req.body.data = {
      strategy: req.body.strategy,
      windowSize: req.body.windowSize,
      stepSize: req.body.stepSize
    };
    return asiController.processAnyRequest(req, res);
  }
);

/**
 * 📊 MONITORING AND METRICS ENDPOINTS
 */

// ASI metrics
router.get('/metrics',
  authenticateUser,
  rateLimiter({ windowMs: 60000, max: 60 }),
  asiController.getMetrics.bind(asiController)
);

// ASI health check
router.get('/health',
  rateLimiter({ windowMs: 60000, max: 120 }),
  asiController.getHealthStatus.bind(asiController)
);

// Performance dashboard
router.get('/dashboard',
  authenticateUser,
  rateLimiter({ windowMs: 60000, max: 30 }),
  (req, res) => {
    req.body.type = 'performance_dashboard';
    return asiController.processAnyRequest(req, res);
  }
);

/**
 * 🔧 DEVELOPMENT AND TESTING ENDPOINTS
 */

// Test ASI system
router.post('/test',
  authenticateUser,
  rateLimiter({ windowMs: 300000, max: 10 }),
  asiController.testASI.bind(asiController)
);

// Reset learning state (development only)
router.post('/reset',
  authenticateUser,
  rateLimiter({ windowMs: 3600000, max: 1 }), // 1 request per hour
  (req, res, next) => {
    if (process.env.NODE_ENV === 'production') {
      return res.status(403).json({
        success: false,
        message: 'Reset endpoint not available in production'
      });
    }
    next();
  },
  asiController.resetLearningState.bind(asiController)
);

/**
 * 🔄 LEGACY COMPATIBILITY ENDPOINTS
 * These redirect old AI/AGI/ASI endpoints to the unified ASI system
 */

// Legacy AI endpoints
router.post('/ai/analyze', (req, res) => {
  req.body.type = 'fund_analysis';
  return asiController.processAnyRequest(req, res);
});

router.post('/ai/predict', (req, res) => {
  req.body.type = req.body.predictionType || 'nav_prediction';
  return asiController.processAnyRequest(req, res);
});

// Legacy AGI endpoints
router.post('/agi/reason', (req, res) => {
  req.body.type = 'cross_domain_analysis';
  return asiController.processAnyRequest(req, res);
});

router.post('/agi/analyze', (req, res) => {
  req.body.type = 'behavioral_analysis';
  return asiController.processAnyRequest(req, res);
});

// Legacy ASI endpoints
router.post('/asi/optimize', (req, res) => {
  req.body.type = 'portfolio_optimization';
  return asiController.processAnyRequest(req, res);
});

router.post('/asi/quantum', (req, res) => {
  req.body.type = 'quantum_optimization';
  return asiController.processAnyRequest(req, res);
});

/**
 * 📚 API DOCUMENTATION ENDPOINT
 */
router.get('/docs', (req, res) => {
  res.json({
    title: '🚀 Unified ASI API Documentation',
    version: '1.0.0',
    description: 'Single API that handles all intelligence requests through ASI Master Engine',
    
    endpoints: {
      universal: {
        'POST /asi/process': 'Universal endpoint - handles any intelligence request',
      },
      
      fund_analysis: {
        'GET /asi/fund/:fundCode/analyze': 'Analyze single fund',
        'POST /asi/fund/analyze': 'Analyze fund with custom parameters'
      },
      
      predictions: {
        'POST /asi/predict': 'General prediction endpoint',
        'POST /asi/predict/nav': 'NAV prediction'
      },
      
      optimization: {
        'POST /asi/portfolio/optimize': 'Portfolio optimization',
        'POST /asi/sip/optimize': 'SIP optimization',
        'POST /asi/quantum/optimize': 'Quantum optimization',
        'POST /asi/quantum/compare': 'Quantum vs Classical comparison'
      },
      
      behavioral: {
        'POST /asi/behavior/analyze': 'Behavioral analysis',
        'POST /asi/sentiment/analyze': 'Market sentiment analysis'
      },
      
      learning: {
        'POST /asi/learn/autonomous': 'Autonomous learning',
        'POST /asi/learn/meta': 'Meta learning'
      },
      
      market: {
        'POST /asi/market/analyze': 'Market analysis',
        'POST /asi/market/timing': 'Market timing analysis'
      },
      
      backtesting: {
        'POST /asi/backtest/strategy': 'Strategy backtesting',
        'POST /asi/backtest/walkforward': 'Walk-forward analysis'
      },
      
      monitoring: {
        'GET /asi/metrics': 'ASI performance metrics',
        'GET /asi/health': 'ASI health status',
        'GET /asi/dashboard': 'Performance dashboard'
      },
      
      development: {
        'POST /asi/test': 'Test ASI system',
        'POST /asi/reset': 'Reset learning state (dev only)'
      }
    },
    
    capabilities: {
      basic: 'Simple ML tasks - fund analysis, NAV prediction',
      general: 'Cross-domain intelligence - behavioral analysis, market sentiment',
      super: 'Advanced optimization - portfolio optimization, autonomous learning',
      quantum: 'Maximum capability - quantum optimization, advanced algorithms'
    },
    
    parameters: {
      urgency: ['relaxed', 'normal', 'urgent', 'critical'],
      precision: ['low', 'standard', 'high', 'critical'],
      riskProfile: ['conservative', 'moderate', 'aggressive']
    }
  });
});

module.exports = router;
