/**
 * 🚀 UNIFIED ASI CONTROLLER
 * 
 * Single controller that handles ALL intelligence requests
 * Routes everything through ASI Master Engine
 * Replaces the need for separate AI/AGI/ASI controllers
 * 
 * @author Universe-Class ASI Architect
 * @version 1.0.0 - Unified ASI Architecture
 */

const { PythonASIBridge } = require('../services/PythonASIBridge');
const { ASIMasterEngine } = require('../asi/ASIMasterEngine');
const { createBreaker } = require('../utils/circuitBreaker');
const { auditLog } = require('../utils/auditLogger');
const response = require('../utils/response');
const logger = require('../utils/logger');

class UnifiedASIController {
  constructor() {
    this.pythonASIBridge = null;
    this.asiMasterEngine = null;
    this.isInitialized = false;
    this.initializationPromise = null;
  }

  /**
   * Initialize ASI Master Engine
   */
  async initialize() {
    if (this.initializationPromise) {
      return this.initializationPromise;
    }

    this.initializationPromise = this._performInitialization();
    return this.initializationPromise;
  }

  async _performInitialization() {
    try {
      logger.info('🚀 Initializing Unified ASI Controller with Node.js ASIMasterEngine...');

      // Initialize Node.js ASIMasterEngine
      this.asiMasterEngine = new ASIMasterEngine();
      await this.asiMasterEngine.initialize();

      // Initialize Python ASI Bridge (fallback only)
      this.pythonASIBridge = new PythonASIBridge({
        // Python ASI service configuration
        pythonASIUrl: process.env.PYTHON_ASI_URL || 'http://localhost:8001',
        timeout: 30000,
        retries: 2,
        
        // Fallback services
        fallbackServices: {
          agi: process.env.AGI_SERVICE_URL || 'http://localhost:8000',
          analytics: process.env.ANALYTICS_SERVICE_URL || 'http://localhost:5001'
        },
        
        // Health monitoring
        healthCheckInterval: 30000,
        enableFallback: true
      });

      // Circuit breaker for Python ASI
      this.pythonBreaker = createBreaker(
        (request) => this.pythonASIBridge.processRequest(request),
        {
          timeout: 35000,
          errorThresholdPercentage: 50,
          resetTimeout: 60000
        }
      );
      // Audit circuit breaker events
      this.pythonBreaker.on('open', () => auditLog('python_asi_breaker_open', null, { reason: 'Too many failures' }));
      this.pythonBreaker.on('close', () => auditLog('python_asi_breaker_close', null, {}));
      this.pythonBreaker.on('halfOpen', () => auditLog('python_asi_breaker_halfopen', null, {}));

      await this.pythonASIBridge.initialize();
      this.isInitialized = true;

      logger.info('✅ Unified ASI Controller initialized with Node.js ASIMasterEngine and Python fallback');

    } catch (error) {
      logger.error('❌ Unified ASI Controller initialization failed:', error);
      throw error;
    }
  }

  /**
   * Ensure ASI is initialized before processing
   */
  async ensureInitialized() {
    if (!this.isInitialized) {
      await this.initialize();
    }
  }

  /**
   * 🧠 UNIVERSAL INTELLIGENCE ENDPOINT
   * Single endpoint that handles ALL requests
   */
  async processAnyRequest(req, res) {
    try {
      await this.ensureInitialized();

      const request = {
        type: req.body.type || 'general_analysis',
        data: req.body.data || {},
        parameters: req.body.parameters || {},
        user: req.user,
        urgency: req.body.urgency || 'normal',
        precision: req.body.precision || 'standard',
        metadata: {
          userAgent: req.get('User-Agent'),
          ip: req.ip,
          timestamp: new Date().toISOString()
        }
      };

      logger.info(`🧠 Processing ${request.type} request for user ${req.user?.id || 'anonymous'}`);

      // Node.js ASI handles everything; fallback to Python only on error
      let result;
      try {
        result = await this.asiMasterEngine.processRequest(request);
      } catch (asiErr) {
        logger.error('Node.js ASIMasterEngine failed, falling back to Python ASI:', asiErr);
        auditLog('node_asi_fallback_to_python', req.user, { error: asiErr.message, requestType: request.type });
        try {
          result = await this.pythonBreaker.fire(request);
        } catch (cbErr) {
          auditLog('python_asi_breaker_fallback', req.user, { error: cbErr.message, requestType: request.type });
          throw cbErr;
        }
      }

      return response.success(res, 'ASI processing completed successfully', {
        result: result.result,
        metadata: {
          requestId: result.requestId,
          capability: result.capability,
          quality: result.quality,
          processingTime: result.processingTime,
          complexity: result.metadata?.complexity,
          retries: result.metadata?.retries
        }
      });

    } catch (error) {
      logger.error('❌ ASI processing failed:', error);
      return response.error(res, 'ASI processing failed', error.message, 500);
    }
  }

  /**
   * 📊 FUND ANALYSIS - Routed through ASI
   */
  async analyzeFund(req, res) {
    try {
      await this.ensureInitialized();

      const request = {
        type: 'fund_analysis',
        data: {
          fundCode: req.params.fundCode || req.body.fundCode,
          includeHistory: req.body.includeHistory !== false
        },
        parameters: {
          depth: req.body.depth || 'comprehensive',
          includeRecommendations: req.body.includeRecommendations !== false
        },
        user: req.user,
        urgency: req.body.urgency || 'normal',
        precision: req.body.precision || 'high'
      };

      let result;
      try {
        result = await this.asiMasterEngine.processRequest(request);
      } catch (asiErr) {
        logger.error('Node.js ASIMasterEngine failed, falling back to Python ASI:', asiErr);
        auditLog('node_asi_fallback_to_python', req.user, { error: asiErr.message, requestType: request.type });
        try {
          result = await this.pythonBreaker.fire(request);
        } catch (cbErr) {
          auditLog('python_asi_breaker_fallback', req.user, { error: cbErr.message, requestType: request.type });
          throw cbErr;
        }
      }

      return response.success(res, 'Fund analysis completed', result.result);

    } catch (error) {
      logger.error('❌ Fund analysis failed:', error);
      return response.error(res, 'Fund analysis failed', error.message, 500);
    }
  }

  /**
   * 🔮 PREDICTION - Routed through ASI
   */
  async predict(req, res) {
    try {
      await this.ensureInitialized();

      const request = {
        type: req.body.predictionType || 'nav_prediction',
        data: req.body.data || {},
        parameters: {
          horizon: req.body.horizon || '30d',
          confidence: req.body.confidence || 0.95,
          includeUncertainty: req.body.includeUncertainty !== false
        },
        user: req.user,
        urgency: req.body.urgency || 'normal',
        precision: req.body.precision || 'high'
      };

      const result = await this.asiMasterEngine.processRequest(request);

      return response.success(res, 'Prediction completed', result.result);

    } catch (error) {
      logger.error('❌ Prediction failed:', error);
      return response.error(res, 'Prediction failed', error.message, 500);
    }
  }

  /**
   * 🎯 PORTFOLIO OPTIMIZATION - Routed through ASI
   */
  async optimizePortfolio(req, res) {
    try {
      await this.ensureInitialized();

      const request = {
        type: 'portfolio_optimization',
        data: {
          funds: req.body.funds || [],
          constraints: req.body.constraints || {},
          riskProfile: req.body.riskProfile || 'moderate'
        },
        parameters: {
          optimizationMethod: req.body.method || 'quantum_inspired',
          includeRebalancing: req.body.includeRebalancing !== false,
          timeHorizon: req.body.timeHorizon || '1y'
        },
        user: req.user,
        urgency: req.body.urgency || 'normal',
        precision: req.body.precision || 'critical'
      };

      const result = await this.asiMasterEngine.processRequest(request);

      return response.success(res, 'Portfolio optimization completed', result.result);

    } catch (error) {
      logger.error('❌ Portfolio optimization failed:', error);
      return response.error(res, 'Portfolio optimization failed', error.message, 500);
    }
  }

  /**
   * 🧠 BEHAVIORAL ANALYSIS - Routed through ASI
   */
  async analyzeBehavior(req, res) {
    try {
      await this.ensureInitialized();

      const request = {
        type: 'behavioral_analysis',
        data: {
          marketData: req.body.marketData || {},
          userProfile: req.body.userProfile || {},
          transactionHistory: req.body.transactionHistory || []
        },
        parameters: {
          includeBiasDetection: req.body.includeBiasDetection !== false,
          includeMarketSentiment: req.body.includeMarketSentiment !== false,
          culturalFactors: req.body.culturalFactors || 'indian_market'
        },
        user: req.user,
        urgency: req.body.urgency || 'normal',
        precision: req.body.precision || 'high'
      };

      const result = await this.asiMasterEngine.processRequest(request);

      return response.success(res, 'Behavioral analysis completed', result.result);

    } catch (error) {
      logger.error('❌ Behavioral analysis failed:', error);
      return response.error(res, 'Behavioral analysis failed', error.message, 500);
    }
  }

  /**
   * ⚡ QUANTUM OPTIMIZATION - Routed through ASI
   */
  async quantumOptimize(req, res) {
    try {
      await this.ensureInitialized();

      const request = {
        type: 'quantum_optimization',
        data: {
          expectedReturns: req.body.expectedReturns || [],
          covarianceMatrix: req.body.covarianceMatrix || [],
          constraints: req.body.constraints || {}
        },
        parameters: {
          algorithm: req.body.algorithm || 'QAOA',
          iterations: req.body.iterations || 100,
          compareWithClassical: req.body.compareWithClassical !== false
        },
        user: req.user,
        urgency: req.body.urgency || 'relaxed', // Quantum is computationally intensive
        precision: req.body.precision || 'critical'
      };

      const result = await this.asiMasterEngine.processRequest(request);

      return response.success(res, 'Quantum optimization completed', result.result);

    } catch (error) {
      logger.error('❌ Quantum optimization failed:', error);
      return response.error(res, 'Quantum optimization failed', error.message, 500);
    }
  }

  /**
   * 🎓 AUTONOMOUS LEARNING - Routed through ASI
   */
  async autonomousLearn(req, res) {
    try {
      await this.ensureInitialized();

      const request = {
        type: 'autonomous_learning',
        data: req.body.data || {},
        parameters: {
          learningType: req.body.learningType || 'meta_learning',
          adaptationRate: req.body.adaptationRate || 0.01,
          explorationRate: req.body.explorationRate || 0.1
        },
        user: req.user,
        urgency: req.body.urgency || 'relaxed',
        precision: req.body.precision || 'high'
      };

      const result = await this.asiMasterEngine.processRequest(request);

      return response.success(res, 'Autonomous learning completed', result.result);

    } catch (error) {
      logger.error('❌ Autonomous learning failed:', error);
      return response.error(res, 'Autonomous learning failed', error.message, 500);
    }
  }

  /**
   * 📈 MARKET ANALYSIS - Routed through ASI
   */
  async analyzeMarket(req, res) {
    try {
      await this.ensureInitialized();

      const request = {
        type: 'market_sentiment',
        data: {
          marketData: req.body.marketData || {},
          timeframe: req.body.timeframe || '1M',
          sectors: req.body.sectors || []
        },
        parameters: {
          includeSentiment: req.body.includeSentiment !== false,
          includeTechnical: req.body.includeTechnical !== false,
          includeMacroeconomic: req.body.includeMacroeconomic !== false
        },
        user: req.user,
        urgency: req.body.urgency || 'normal',
        precision: req.body.precision || 'high'
      };

      const result = await this.asiMasterEngine.processRequest(request);

      return response.success(res, 'Market analysis completed', result.result);

    } catch (error) {
      logger.error('❌ Market analysis failed:', error);
      return response.error(res, 'Market analysis failed', error.message, 500);
    }
  }

  /**
   * 🔄 BACKTEST STRATEGY - Routed through ASI
   */
  async backtestStrategy(req, res) {
    try {
      await this.ensureInitialized();

      const request = {
        type: 'backtest',
        data: {
          strategy: req.body.strategy || {},
          historicalData: req.body.historicalData || {},
          timeframe: req.body.timeframe || '5Y'
        },
        parameters: {
          monteCarloRuns: req.body.monteCarloRuns || 1000,
          includeTransactionCosts: req.body.includeTransactionCosts !== false,
          riskMetrics: req.body.riskMetrics || ['sharpe', 'sortino', 'calmar']
        },
        user: req.user,
        urgency: req.body.urgency || 'relaxed',
        precision: req.body.precision || 'high'
      };

      const result = await this.asiMasterEngine.processRequest(request);

      return response.success(res, 'Backtest completed', result.result);

    } catch (error) {
      logger.error('❌ Backtest failed:', error);
      return response.error(res, 'Backtest failed', error.message, 500);
    }
  }

  /**
   * 📊 ASI METRICS AND MONITORING
   */
  async getMetrics(req, res) {
    try {
      await this.ensureInitialized();

      const metrics = await this.pythonASIBridge.getMetrics();

      return response.success(res, 'ASI metrics retrieved', metrics);

    } catch (error) {
      logger.error('❌ Failed to get ASI metrics:', error);
      return response.error(res, 'Failed to get ASI metrics', error.message, 500);
    }
  }

  /**
   * ❤️ ASI HEALTH CHECK
   */
  async getHealthStatus(req, res) {
    try {
      let healthStatus;

      if (this.isInitialized) {
        healthStatus = await this.pythonASIBridge.getHealthStatus();
      } else {
        healthStatus = {
          status: 'initializing',
          message: 'Python ASI Bridge is initializing'
        };
      }

      const statusCode = healthStatus.status === 'healthy' ? 200 : 503;

      return response.success(res, 'ASI health status', healthStatus, statusCode);

    } catch (error) {
      logger.error('❌ Health check failed:', error);
      return response.error(res, 'Health check failed', error.message, 503);
    }
  }

  /**
   * 🔧 DEVELOPMENT AND TESTING ENDPOINTS
   */

  /**
   * Test ASI with sample data
   */
  async testASI(req, res) {
    try {
      await this.ensureInitialized();

      const testRequests = [
        {
          type: 'fund_analysis',
          data: { fundCode: 'TEST001' },
          parameters: { depth: 'basic' },
          urgency: 'normal',
          precision: 'standard'
        },
        {
          type: 'nav_prediction',
          data: { fundCode: 'TEST001', history: [] },
          parameters: { horizon: '7d' },
          urgency: 'normal',
          precision: 'standard'
        }
      ];

      const results = [];

      for (const testRequest of testRequests) {
        try {
          let result;
          try {
            result = await this.pythonBreaker.fire(testRequest);
          } catch (cbErr) {
            auditLog('python_asi_breaker_fallback', req.user, { error: cbErr.message, requestType: testRequest.type });
            throw cbErr;
          }
          results.push({
            request: testRequest.type,
            status: 'success',
            capability: result.capability,
            processingTime: result.processingTime
          });
        } catch (error) {
          results.push({
            request: testRequest.type,
            status: 'failed',
            error: error.message
          });
        }
      }

      return response.success(res, 'ASI test completed', { results });

    } catch (error) {
      logger.error('❌ ASI test failed:', error);
      return response.error(res, 'ASI test failed', error.message, 500);
    }
  }

  /**
   * Reset ASI learning state (development only)
   */
  async resetLearningState(req, res) {
    try {
      await this.ensureInitialized();

      // Reset learning components via Python ASI
      await this.pythonASIBridge.resetLearningState();

      logger.info('🔄 Python ASI learning state reset');

      return response.success(res, 'ASI learning state reset successfully');

    } catch (error) {
      logger.error('❌ Failed to reset ASI learning state:', error);
      return response.error(res, 'Failed to reset ASI learning state', error.message, 500);
    }
  }
}

module.exports = { UnifiedASIController };
