/**
 * 🚀 UNIFIED ASI SERVICE INTEGRATION
 * 
 * Service layer integration for Unified Finance ASI System
 * Connects unified ASI to main application service layer
 * 
 * @author Universe-Class ASI Architect
 * @version 1.0.0 - Unified Finance ASI
 */

const { UnifiedFinanceASI } = require('../unified-asi');
const logger = require('../utils/logger');

class UnifiedASIService {
  constructor() {
    this.asi = null;
    this.isInitialized = false;
    this.initializationPromise = null;
    this.metrics = {
      requestCount: 0,
      successCount: 0,
      errorCount: 0,
      averageResponseTime: 0,
      lastHealthCheck: null
    };
  }

  /**
   * Initialize the Unified ASI Service
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
      logger.info('🚀 Initializing Unified ASI Service...');

      // Create unified ASI instance
      this.asi = new UnifiedFinanceASI();

      // Initialize the ASI system
      const initResult = await this.asi.initialize();

      this.isInitialized = true;
      
      logger.info('✅ Unified ASI Service initialized successfully');
      logger.info(`🏆 ASI System Rating: ${initResult.rating.toFixed(1)}/10`);
      logger.info(`⏱️  Initialization Time: ${initResult.initializationTime}ms`);

      // Setup periodic health checks
      this.setupHealthChecks();

      return {
        success: true,
        service: 'UnifiedASIService',
        rating: initResult.rating,
        capabilities: initResult.capabilities
      };

    } catch (error) {
      logger.error('❌ Unified ASI Service initialization failed:', error);
      this.isInitialized = false;
      throw error;
    }
  }

  /**
   * Setup periodic health checks
   */
  setupHealthChecks() {
    setInterval(async () => {
      try {
        const health = await this.asi.getSystemHealth();
        this.metrics.lastHealthCheck = {
          timestamp: new Date().toISOString(),
          status: health.status,
          details: health
        };
      } catch (error) {
        logger.error('Health check failed:', error);
        this.metrics.lastHealthCheck = {
          timestamp: new Date().toISOString(),
          status: 'ERROR',
          error: error.message
        };
      }
    }, 30000); // Every 30 seconds
  }

  /**
   * Process ASI request
   */
  async processRequest(request) {
    if (!this.isInitialized) {
      throw new Error('Unified ASI Service not initialized');
    }

    const startTime = Date.now();
    this.metrics.requestCount++;

    try {
      const result = await this.asi.processRequest(request);
      
      this.metrics.successCount++;
      const responseTime = Date.now() - startTime;
      this.updateAverageResponseTime(responseTime);

      return result;

    } catch (error) {
      this.metrics.errorCount++;
      logger.error('ASI request processing failed:', error);
      throw error;
    }
  }

  /**
   * Get portfolio analysis
   */
  async analyzePortfolio(portfolioData) {
    return this.processRequest({
      type: 'portfolio_analysis',
      data: portfolioData
    });
  }

  /**
   * Get market predictions
   */
  async getPredictions(symbols, timeHorizon = 30) {
    return this.processRequest({
      type: 'prediction',
      data: { symbols, timeHorizon }
    });
  }

  /**
   * Perform risk assessment
   */
  async assessRisk(portfolioData) {
    return this.processRequest({
      type: 'risk_assessment',
      data: portfolioData
    });
  }

  /**
   * Get market analysis
   */
  async analyzeMarket(marketData) {
    return this.processRequest({
      type: 'market_analysis',
      data: marketData
    });
  }

  /**
   * Compare funds
   */
  async compareFunds(fundIds) {
    return this.processRequest({
      type: 'fund_comparison',
      data: { fundIds }
    });
  }

  /**
   * Optimize portfolio
   */
  async optimizePortfolio(portfolioData, objectives = ['max_sharpe']) {
    return this.processRequest({
      type: 'optimization',
      data: { ...portfolioData, objectives }
    });
  }

  /**
   * Perform backtesting
   */
  async backtest(strategy, period = '1Y') {
    return this.processRequest({
      type: 'backtesting',
      data: { strategy, period }
    });
  }

  /**
   * Get system status
   */
  async getSystemStatus() {
    if (!this.asi) {
      return {
        initialized: false,
        status: 'NOT_INITIALIZED'
      };
    }

    return await this.asi.getSystemStatus();
  }

  /**
   * Get system capabilities
   */
  async getCapabilities() {
    if (!this.asi) {
      return { error: 'ASI not initialized' };
    }

    return await this.asi.getSystemCapabilities();
  }

  /**
   * Get service health
   */
  async getHealth() {
    if (!this.isInitialized) {
      return {
        status: 'UNHEALTHY',
        reason: 'Service not initialized',
        metrics: this.metrics
      };
    }

    try {
      const asiHealth = await this.asi.getSystemHealth();
      
      return {
        status: asiHealth.status,
        service: 'UnifiedASIService',
        asi: asiHealth,
        metrics: this.metrics,
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      return {
        status: 'ERROR',
        error: error.message,
        metrics: this.metrics,
        timestamp: new Date().toISOString()
      };
    }
  }

  /**
   * Get service metrics
   */
  getMetrics() {
    const successRate = this.metrics.requestCount > 0 ? 
      (this.metrics.successCount / this.metrics.requestCount * 100).toFixed(2) : 0;

    return {
      ...this.metrics,
      successRate: `${successRate}%`,
      errorRate: `${((this.metrics.errorCount / this.metrics.requestCount * 100) || 0).toFixed(2)}%`
    };
  }

  /**
   * Update average response time
   */
  updateAverageResponseTime(responseTime) {
    if (this.metrics.successCount === 1) {
      this.metrics.averageResponseTime = responseTime;
    } else {
      this.metrics.averageResponseTime = 
        (this.metrics.averageResponseTime * (this.metrics.successCount - 1) + responseTime) / 
        this.metrics.successCount;
    }
  }

  /**
   * Shutdown the service
   */
  async shutdown() {
    try {
      logger.info('🛑 Shutting down Unified ASI Service...');

      if (this.asi) {
        await this.asi.shutdown();
      }

      this.isInitialized = false;
      logger.info('✅ Unified ASI Service shutdown complete');

    } catch (error) {
      logger.error('❌ Unified ASI Service shutdown error:', error);
      throw error;
    }
  }
}

// Create singleton instance
const unifiedASIService = new UnifiedASIService();

module.exports = {
  UnifiedASIService,
  unifiedASIService
};
