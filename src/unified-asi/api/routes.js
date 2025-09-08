/**
 * 🚀 UNIFIED ASI API ROUTES
 * 
 * Single API endpoint for complete Finance ASI system
 * All AI/AGI/ASI capabilities accessible through unified interface
 * 
 * @author Universe-Class ASI Architect
 * @version 1.0.0 - Unified Finance ASI
 */

const express = require('express');
const router = express.Router();
const { ASIMasterController } = require('../core/ASIMasterController');
const logger = require('../../utils/logger');

// Initialize ASI Master Controller
const asiController = new ASIMasterController();
let isInitialized = false;

// Middleware to ensure ASI system is initialized
const ensureInitialized = async (req, res, next) => {
  if (!isInitialized) {
    try {
      await asiController.initialize();
      isInitialized = true;
      logger.info('✅ ASI System initialized for first request');
    } catch (error) {
      logger.error('❌ ASI System initialization failed:', error);
      return res.status(503).json({
        success: false,
        error: 'ASI System initialization failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  }
  next();
};

// Apply middleware to all routes
router.use(ensureInitialized);

/**
 * 🎯 UNIFIED ASI ENDPOINT
 * Single endpoint for all ASI capabilities
 */
router.post('/process', async (req, res) => {
  try {
    const { type, data, options = {} } = req.body;
    
    // Validate request
    if (!type) {
      return res.status(400).json({
        success: false,
        error: 'Request type is required',
        availableTypes: [
          'portfolio_analysis',
          'risk_assessment', 
          'prediction',
          'market_analysis',
          'fund_comparison',
          'backtesting',
          'optimization'
        ],
        timestamp: new Date().toISOString()
      });
    }
    
    // Process request through ASI system
    const result = await asiController.processRequest({
      type,
      data,
      options,
      userId: req.user?.id,
      sessionId: req.sessionID
    });
    
    res.json(result);
    
  } catch (error) {
    logger.error('❌ ASI processing error:', error);
    res.status(500).json({
      success: false,
      error: 'ASI processing failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * 📊 PORTFOLIO ANALYSIS
 * Comprehensive portfolio analysis using ASI
 */
router.post('/portfolio/analyze', async (req, res) => {
  try {
    const { symbols, amounts, timeHorizon = 365, riskProfile = 'moderate' } = req.body;
    
    const result = await asiController.processRequest({
      type: 'portfolio_analysis',
      data: {
        symbols,
        amounts,
        timeHorizon,
        riskProfile,
        analysisType: 'comprehensive'
      }
    });
    
    res.json(result);
    
  } catch (error) {
    logger.error('❌ Portfolio analysis error:', error);
    res.status(500).json({
      success: false,
      error: 'Portfolio analysis failed',
      message: error.message
    });
  }
});

/**
 * 🔮 PREDICTIONS
 * Advanced ML/AI predictions
 */
router.post('/predict', async (req, res) => {
  try {
    const { symbols, predictionType = 'price', timeHorizon = 30, confidence = 0.95 } = req.body;
    
    const result = await asiController.processRequest({
      type: 'prediction',
      data: {
        symbols,
        predictionType,
        timeHorizon,
        confidence,
        includeUncertainty: true
      }
    });
    
    res.json(result);
    
  } catch (error) {
    logger.error('❌ Prediction error:', error);
    res.status(500).json({
      success: false,
      error: 'Prediction failed',
      message: error.message
    });
  }
});

/**
 * ⚖️ RISK ASSESSMENT
 * Comprehensive risk analysis
 */
router.post('/risk/assess', async (req, res) => {
  try {
    const { portfolio, riskMetrics = ['var', 'cvar', 'sharpe', 'sortino'] } = req.body;
    
    const result = await asiController.processRequest({
      type: 'risk_assessment',
      data: {
        portfolio,
        riskMetrics,
        confidenceLevel: 0.95,
        timeHorizon: 252 // 1 year trading days
      }
    });
    
    res.json(result);
    
  } catch (error) {
    logger.error('❌ Risk assessment error:', error);
    res.status(500).json({
      success: false,
      error: 'Risk assessment failed',
      message: error.message
    });
  }
});

/**
 * 🌍 MARKET ANALYSIS
 * Global market intelligence
 */
router.post('/market/analyze', async (req, res) => {
  try {
    const { markets = ['indian'], assetClasses = ['equity'], depth = 'comprehensive' } = req.body;
    
    const result = await asiController.processRequest({
      type: 'market_analysis',
      data: {
        markets,
        assetClasses,
        depth,
        includeAlternativeData: true,
        includeSentiment: true
      }
    });
    
    res.json(result);
    
  } catch (error) {
    logger.error('❌ Market analysis error:', error);
    res.status(500).json({
      success: false,
      error: 'Market analysis failed',
      message: error.message
    });
  }
});

/**
 * 🔄 FUND COMPARISON
 * Advanced mutual fund comparison
 */
router.post('/funds/compare', async (req, res) => {
  try {
    const { fundCodes, comparisonMetrics = ['returns', 'risk', 'ratios', 'portfolio'] } = req.body;
    
    const result = await asiController.processRequest({
      type: 'fund_comparison',
      data: {
        fundCodes,
        comparisonMetrics,
        timeHorizon: [30, 90, 365, 1095], // 1M, 3M, 1Y, 3Y
        includeRanking: true
      }
    });
    
    res.json(result);
    
  } catch (error) {
    logger.error('❌ Fund comparison error:', error);
    res.status(500).json({
      success: false,
      error: 'Fund comparison failed',
      message: error.message
    });
  }
});

/**
 * 📈 BACKTESTING
 * Strategy backtesting with ASI
 */
router.post('/backtest', async (req, res) => {
  try {
    const { strategy, startDate, endDate, initialCapital = 100000 } = req.body;
    
    const result = await asiController.processRequest({
      type: 'backtesting',
      data: {
        strategy,
        startDate,
        endDate,
        initialCapital,
        includeMetrics: true,
        includeBenchmark: true
      }
    });
    
    res.json(result);
    
  } catch (error) {
    logger.error('❌ Backtesting error:', error);
    res.status(500).json({
      success: false,
      error: 'Backtesting failed',
      message: error.message
    });
  }
});

/**
 * ⚡ OPTIMIZATION
 * Portfolio optimization using ASI
 */
router.post('/optimize', async (req, res) => {
  try {
    const { 
      universe, 
      constraints = {}, 
      objective = 'max_sharpe',
      optimizationType = 'quantum_inspired'
    } = req.body;
    
    const result = await asiController.processRequest({
      type: 'optimization',
      data: {
        universe,
        constraints,
        objective,
        optimizationType,
        includeAlternatives: true
      }
    });
    
    res.json(result);
    
  } catch (error) {
    logger.error('❌ Optimization error:', error);
    res.status(500).json({
      success: false,
      error: 'Optimization failed',
      message: error.message
    });
  }
});

/**
 * 🏥 SYSTEM HEALTH
 * ASI system health and status
 */
router.get('/health', async (req, res) => {
  try {
    const health = await asiController.performHealthCheck();
    
    const statusCode = health.systemHealth === 'HEALTHY' ? 200 : 
                      health.systemHealth === 'DEGRADED' ? 206 : 503;
    
    res.status(statusCode).json(health);
    
  } catch (error) {
    logger.error('❌ Health check error:', error);
    res.status(503).json({
      systemHealth: 'UNHEALTHY',
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * 📊 SYSTEM STATUS
 * Detailed system status and metrics
 */
router.get('/status', async (req, res) => {
  try {
    const status = await asiController.getSystemStatus();
    res.json(status);
    
  } catch (error) {
    logger.error('❌ Status check error:', error);
    res.status(500).json({
      error: 'Status check failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * 🎯 CAPABILITIES
 * List all ASI capabilities
 */
router.get('/capabilities', (req, res) => {
  res.json({
    systemName: 'Unified Finance ASI',
    version: '1.0.0',
    targetRating: '9+/10',
    capabilities: {
      'Portfolio Analysis': {
        endpoint: '/portfolio/analyze',
        description: 'Comprehensive portfolio analysis with ASI',
        features: ['Risk assessment', 'Performance attribution', 'Optimization suggestions']
      },
      'Predictions': {
        endpoint: '/predict',
        description: 'Advanced ML/AI predictions',
        features: ['Price predictions', 'Return forecasting', 'Uncertainty quantification']
      },
      'Risk Assessment': {
        endpoint: '/risk/assess',
        description: 'Comprehensive risk analysis',
        features: ['VaR calculation', 'Stress testing', 'Risk attribution']
      },
      'Market Analysis': {
        endpoint: '/market/analyze',
        description: 'Global market intelligence',
        features: ['Market trends', 'Sentiment analysis', 'Alternative data']
      },
      'Fund Comparison': {
        endpoint: '/funds/compare',
        description: 'Advanced mutual fund comparison',
        features: ['Performance comparison', 'Risk-adjusted returns', 'Portfolio analysis']
      },
      'Backtesting': {
        endpoint: '/backtest',
        description: 'Strategy backtesting with ASI',
        features: ['Historical simulation', 'Performance metrics', 'Risk analysis']
      },
      'Optimization': {
        endpoint: '/optimize',
        description: 'Portfolio optimization using ASI',
        features: ['Quantum-inspired algorithms', 'Multi-objective optimization', 'Constraint handling']
      }
    },
    integrations: {
      'Python ASI': 'Advanced ML models and LLMs',
      'Real-time Data': 'Live market data feeds',
      'Alternative Data': 'Satellite intelligence, sentiment data',
      'Quantum Computing': 'Quantum-inspired optimization'
    },
    performance: {
      'Target Response Time': '< 500ms',
      'Target Accuracy': '85%+',
      'Concurrent Requests': '100+',
      'Uptime Target': '99.9%'
    }
  });
});

/**
 * 📚 API DOCUMENTATION
 * Interactive API documentation
 */
router.get('/docs', (req, res) => {
  res.json({
    title: 'Unified Finance ASI API Documentation',
    version: '1.0.0',
    description: 'Complete Financial Artificial Superintelligence System API',
    baseUrl: '/api/unified-asi',
    endpoints: {
      'POST /process': {
        description: 'Universal ASI processing endpoint',
        parameters: {
          type: 'string (required) - Request type',
          data: 'object (required) - Request data',
          options: 'object (optional) - Processing options'
        },
        example: {
          type: 'portfolio_analysis',
          data: {
            symbols: ['RELIANCE', 'TCS'],
            amounts: [50000, 50000],
            timeHorizon: 365
          }
        }
      },
      'POST /portfolio/analyze': {
        description: 'Portfolio analysis endpoint',
        parameters: {
          symbols: 'array (required) - Stock/fund symbols',
          amounts: 'array (required) - Investment amounts',
          timeHorizon: 'number (optional) - Analysis period in days',
          riskProfile: 'string (optional) - Risk profile'
        }
      },
      'POST /predict': {
        description: 'Prediction endpoint',
        parameters: {
          symbols: 'array (required) - Symbols to predict',
          predictionType: 'string (optional) - Type of prediction',
          timeHorizon: 'number (optional) - Prediction horizon',
          confidence: 'number (optional) - Confidence level'
        }
      },
      'GET /health': {
        description: 'System health check',
        response: 'Health status and component status'
      },
      'GET /status': {
        description: 'Detailed system status',
        response: 'Complete system metrics and status'
      },
      'GET /capabilities': {
        description: 'List all ASI capabilities',
        response: 'Available capabilities and features'
      }
    },
    examples: {
      'Portfolio Analysis': {
        request: {
          symbols: ['RELIANCE', 'TCS', 'INFY'],
          amounts: [40000, 30000, 30000],
          timeHorizon: 365,
          riskProfile: 'moderate'
        }
      },
      'Prediction': {
        request: {
          symbols: ['NIFTY50'],
          predictionType: 'price',
          timeHorizon: 30,
          confidence: 0.95
        }
      },
      'Risk Assessment': {
        request: {
          portfolio: {
            'RELIANCE': 0.4,
            'TCS': 0.3,
            'INFY': 0.3
          },
          riskMetrics: ['var', 'cvar', 'sharpe']
        }
      }
    }
  });
});

module.exports = router;
