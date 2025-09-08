/**
 * 🐍 PYTHON ASI BRIDGE SERVICE
 * 
 * Bridge between Node.js backend and Python ASI Master Engine
 * Routes intelligence requests to real Python AI/ML services
 * 
 * @author Universe-Class ASI Architect (Corrected)
 * @version 2.0.0 - Python AI Integration
 */

const axios = require('axios');
const logger = require('../utils/logger');

class PythonASIBridge {
  constructor(config = {}) {
    this.config = {
      // Python ASI Master Engine
      asiMasterUrl: config.asiMasterUrl || 'http://localhost:8001',
      
      // Existing Python services
      agiServiceUrl: config.agiServiceUrl || 'http://localhost:8000',
      analyticsServiceUrl: config.analyticsServiceUrl || 'http://localhost:5001',
      
      // Timeouts and retries
      timeout: config.timeout || 30000,
      maxRetries: config.maxRetries || 3,
      retryDelay: config.retryDelay || 1000,
      
      // Health check intervals
      healthCheckInterval: config.healthCheckInterval || 60000,
      
      ...config
    };

    // Service health status
    this.serviceHealth = {
      asiMaster: false,
      agiService: false,
      analyticsService: false
    };

    // Request statistics
    this.stats = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      averageResponseTime: 0,
      serviceUsage: {
        asiMaster: 0,
        agiService: 0,
        analyticsService: 0
      }
    };

    // Start health monitoring
    this.startHealthMonitoring();
  }

  /**
   * 🧠 UNIVERSAL PYTHON ASI INTERFACE
   * Routes all intelligence requests to Python ASI Master Engine
   */
  async processRequest(request) {
    const startTime = Date.now();
    
    try {
      logger.info(`🐍 Routing request to Python ASI: ${request.type}`);

      // Ensure ASI Master is healthy
      if (!this.serviceHealth.asiMaster) {
        await this.checkASIMasterHealth();
        
        if (!this.serviceHealth.asiMaster) {
          throw new Error('Python ASI Master Engine is not available');
        }
      }

      // Route to Python ASI Master Engine
      const response = await this.callPythonASI('/asi/process', {
        type: request.type,
        data: request.data || {},
        parameters: request.parameters || {},
        user_id: request.user_id,
        urgency: request.urgency || 'normal',
        precision: request.precision || 'standard'
      });

      // Update statistics
      const responseTime = Date.now() - startTime;
      this.updateStats('asiMaster', responseTime, true);

      logger.info(`✅ Python ASI completed request: ${response.data.request_id}`);

      return {
        success: true,
        requestId: response.data.request_id,
        result: response.data.result,
        capability: response.data.capability,
        qualityScore: response.data.quality_score,
        processingTime: response.data.processing_time,
        modelUsed: response.data.model_used,
        confidence: response.data.confidence,
        source: 'python_asi_master'
      };

    } catch (error) {
      logger.error(`❌ Python ASI request failed: ${error.message}`);
      
      const responseTime = Date.now() - startTime;
      this.updateStats('asiMaster', responseTime, false);

      // Try fallback to existing Python services
      return await this.fallbackToExistingServices(request, error);
    }
  }

  /**
   * 🔄 FALLBACK TO EXISTING PYTHON SERVICES
   * If ASI Master fails, use existing AGI/Analytics services
   */
  async fallbackToExistingServices(request, originalError) {
    logger.info(`🔄 Attempting fallback to existing Python services...`);

    try {
      // Route to appropriate existing service based on request type
      if (this.isAGIRequest(request.type)) {
        return await this.routeToAGIService(request);
      } else if (this.isAnalyticsRequest(request.type)) {
        return await this.routeToAnalyticsService(request);
      } else {
        // Default to AGI service for general requests
        return await this.routeToAGIService(request);
      }

    } catch (fallbackError) {
      logger.error(`❌ All Python services failed. Original: ${originalError.message}, Fallback: ${fallbackError.message}`);
      
      throw new Error(`Python AI services unavailable. ASI: ${originalError.message}, Fallback: ${fallbackError.message}`);
    }
  }

  /**
   * 🤖 ROUTE TO AGI SERVICE
   * Use existing AGI microservice for general intelligence
   */
  async routeToAGIService(request) {
    const startTime = Date.now();

    try {
      logger.info(`🤖 Routing to AGI service: ${request.type}`);

      let endpoint, payload;

      switch (request.type) {
        case 'behavioral_analysis':
        case 'market_sentiment':
        case 'text_analysis':
          endpoint = '/inference';
          payload = {
            prompt: this.convertToPrompt(request),
            max_tokens: 128
          };
          break;

        case 'portfolio_optimization':
        case 'financial_planning':
          endpoint = '/plan';
          payload = {
            user_id: request.user_id || 'anonymous',
            goal_amount: request.data.goalAmount || 1000000,
            goal_date: request.data.goalDate || '2025-12-31',
            start_amount: request.data.startAmount || 100000,
            monthly_contribution: request.data.monthlyContribution || 10000,
            risk_tolerance: request.data.riskTolerance || 'moderate'
          };
          break;

        default:
          endpoint = '/inference';
          payload = {
            prompt: this.convertToPrompt(request),
            max_tokens: 64
          };
      }

      const response = await this.callPythonService(
        this.config.agiServiceUrl,
        endpoint,
        payload
      );

      const responseTime = Date.now() - startTime;
      this.updateStats('agiService', responseTime, true);

      return {
        success: true,
        result: response.data,
        source: 'agi_service',
        processingTime: responseTime,
        modelUsed: 'AGI_Microservice',
        confidence: 0.8
      };

    } catch (error) {
      const responseTime = Date.now() - startTime;
      this.updateStats('agiService', responseTime, false);
      throw error;
    }
  }

  /**
   * 📊 ROUTE TO ANALYTICS SERVICE
   * Use existing analytics ML service for statistical analysis
   */
  async routeToAnalyticsService(request) {
    const startTime = Date.now();

    try {
      logger.info(`📊 Routing to Analytics service: ${request.type}`);

      let endpoint, payload;

      switch (request.type) {
        case 'statistical_analysis':
          endpoint = '/statistical/mean';
          payload = { data: request.data.values || [] };
          break;

        case 'risk_assessment':
          endpoint = '/risk/var';
          payload = {
            data: request.data.returns || [],
            confidence_level: request.parameters?.confidenceLevel || 0.95
          };
          break;

        case 'nav_prediction':
        case 'price_prediction':
          endpoint = '/ml/lstm';
          payload = { series: request.data.history || [] };
          break;

        case 'explainability':
          endpoint = '/explain/shap';
          payload = { features: request.data.features || [] };
          break;

        default:
          endpoint = '/statistical/mean';
          payload = { data: request.data.values || [1, 2, 3, 4, 5] };
      }

      const response = await this.callPythonService(
        this.config.analyticsServiceUrl,
        endpoint,
        payload
      );

      const responseTime = Date.now() - startTime;
      this.updateStats('analyticsService', responseTime, true);

      return {
        success: true,
        result: response.data,
        source: 'analytics_service',
        processingTime: responseTime,
        modelUsed: 'Analytics_ML_Service',
        confidence: 0.75
      };

    } catch (error) {
      const responseTime = Date.now() - startTime;
      this.updateStats('analyticsService', responseTime, false);
      throw error;
    }
  }

  /**
   * 🔧 HELPER METHODS
   */

  isAGIRequest(requestType) {
    const agiTypes = [
      'behavioral_analysis',
      'market_sentiment',
      'text_analysis',
      'portfolio_optimization',
      'financial_planning',
      'reasoning',
      'inference'
    ];
    return agiTypes.includes(requestType);
  }

  isAnalyticsRequest(requestType) {
    const analyticsTypes = [
      'statistical_analysis',
      'risk_assessment',
      'nav_prediction',
      'price_prediction',
      'explainability'
    ];
    return analyticsTypes.includes(requestType);
  }

  convertToPrompt(request) {
    const { type, data } = request;
    
    switch (type) {
      case 'fund_analysis':
        return `Analyze mutual fund ${data.fundCode || 'UNKNOWN'} and provide investment recommendation`;
      
      case 'market_sentiment':
        return `Analyze current market sentiment and provide insights`;
      
      case 'behavioral_analysis':
        return `Analyze investor behavior and identify cognitive biases`;
      
      default:
        return `Provide financial analysis for ${type}`;
    }
  }

  /**
   * 🌐 HTTP CLIENT METHODS
   */

  async callPythonASI(endpoint, data) {
    return await this.callPythonService(this.config.asiMasterUrl, endpoint, data);
  }

  async callPythonService(baseUrl, endpoint, data) {
    const url = `${baseUrl}${endpoint}`;
    
    const config = {
      method: 'POST',
      url,
      data,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'NodeJS-ASI-Bridge/2.0.0'
      }
    };

    let lastError;
    
    for (let attempt = 1; attempt <= this.config.maxRetries; attempt++) {
      try {
        logger.debug(`🌐 Calling ${url} (attempt ${attempt})`);
        
        const response = await axios(config);
        
        if (response.status === 200 && response.data) {
          return response;
        } else {
          throw new Error(`Invalid response: ${response.status}`);
        }

      } catch (error) {
        lastError = error;
        
        if (attempt < this.config.maxRetries) {
          const delay = this.config.retryDelay * attempt;
          logger.warn(`⚠️ Request failed (attempt ${attempt}), retrying in ${delay}ms: ${error.message}`);
          await this.sleep(delay);
        }
      }
    }

    throw new Error(`Failed after ${this.config.maxRetries} attempts: ${lastError.message}`);
  }

  /**
   * ❤️ HEALTH MONITORING
   */

  startHealthMonitoring() {
    // Initial health check
    this.performHealthChecks();

    // Periodic health checks
    setInterval(() => {
      this.performHealthChecks();
    }, this.config.healthCheckInterval);

    logger.info('🏥 Health monitoring started for Python services');
  }

  async performHealthChecks() {
    await Promise.all([
      this.checkASIMasterHealth(),
      this.checkAGIServiceHealth(),
      this.checkAnalyticsServiceHealth()
    ]);
  }

  async checkASIMasterHealth() {
    try {
      const response = await axios.get(`${this.config.asiMasterUrl}/asi/health`, {
        timeout: 5000
      });
      
      this.serviceHealth.asiMaster = response.status === 200 && response.data.status === 'healthy';
      
      if (this.serviceHealth.asiMaster) {
        logger.debug('✅ Python ASI Master Engine is healthy');
      }

    } catch (error) {
      this.serviceHealth.asiMaster = false;
      logger.warn(`⚠️ Python ASI Master Engine health check failed: ${error.message}`);
    }
  }

  async checkAGIServiceHealth() {
    try {
      const response = await axios.get(`${this.config.agiServiceUrl}/health`, {
        timeout: 5000
      });
      
      this.serviceHealth.agiService = response.status === 200;
      
      if (this.serviceHealth.agiService) {
        logger.debug('✅ AGI Service is healthy');
      }

    } catch (error) {
      this.serviceHealth.agiService = false;
      logger.warn(`⚠️ AGI Service health check failed: ${error.message}`);
    }
  }

  async checkAnalyticsServiceHealth() {
    try {
      const response = await axios.get(`${this.config.analyticsServiceUrl}/health`, {
        timeout: 5000
      });
      
      this.serviceHealth.analyticsService = response.status === 200;
      
      if (this.serviceHealth.analyticsService) {
        logger.debug('✅ Analytics Service is healthy');
      }

    } catch (error) {
      this.serviceHealth.analyticsService = false;
      logger.warn(`⚠️ Analytics Service health check failed: ${error.message}`);
    }
  }

  /**
   * 📊 METRICS AND MONITORING
   */

  updateStats(service, responseTime, success) {
    this.stats.totalRequests++;
    this.stats.serviceUsage[service]++;

    if (success) {
      this.stats.successfulRequests++;
    } else {
      this.stats.failedRequests++;
    }

    // Update average response time
    const total = this.stats.totalRequests;
    const currentAvg = this.stats.averageResponseTime;
    this.stats.averageResponseTime = (currentAvg * (total - 1) + responseTime) / total;
  }

  getStats() {
    return {
      ...this.stats,
      serviceHealth: this.serviceHealth,
      successRate: this.stats.totalRequests > 0 
        ? this.stats.successfulRequests / this.stats.totalRequests 
        : 0
    };
  }

  getHealthStatus() {
    const healthyServices = Object.values(this.serviceHealth).filter(Boolean).length;
    const totalServices = Object.keys(this.serviceHealth).length;
    
    return {
      overall: healthyServices === totalServices ? 'healthy' : 
               healthyServices > 0 ? 'degraded' : 'unhealthy',
      services: this.serviceHealth,
      healthyServices,
      totalServices
    };
  }

  /**
   * 🔧 UTILITY METHODS
   */

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

module.exports = { PythonASIBridge };
