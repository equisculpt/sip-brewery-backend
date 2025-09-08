/**
 * 🚀 ASI MASTER ENGINE
 * 
 * Universal Intelligence System - Single point that handles ALL requests
 * Automatically determines optimal approach and capability level
 * Replaces the need for separate AI/AGI/ASI decision making
 * 
 * @author Universe-Class ASI Architect
 * @version 1.0.0 - Unified ASI Architecture
 * ASIMasterEngine.js - Core ASI Engine
 */

const tf = require('@tensorflow/tfjs-node-gpu');
const logger = require('../utils/logger');
const { WebResearchAgent } = require('./WebResearchAgent');

// Import existing components as capabilities
const { ContinuousLearningEngine } = require('../ai/ContinuousLearningEngine');
const { MutualFundAnalyzer } = require('../ai/MutualFundAnalyzer');
const { RealTimeDataFeeds } = require('../ai/RealTimeDataFeeds');
const { BacktestingFramework } = require('../ai/BacktestingFramework');
// Advanced Financial Models
const { AlphaModel } = require('./models/AlphaModel');
const { RiskModel } = require('./models/RiskModel');
const { DeepForecastingModel } = require('./models/DeepForecastingModel');
const { QuantumPortfolioOptimizer } = require('./models/QuantumPortfolioOptimizer');
const { ExplainabilityEngine } = require('./models/ExplainabilityEngine');
const { AGIEngine } = require('../services/agiEngine');
const { ReinforcementLearningEngine } = require('./ReinforcementLearningEngine');
const { QuantumInspiredOptimizer } = require('./QuantumInspiredOptimizer');
const { BehavioralFinanceEngine } = require('./BehavioralFinanceEngine');
const { AutonomousLearningSystem } = require('./AutonomousLearningSystem');

const { complianceCheck } = require('../middleware/complianceMiddleware');
const QuantumOptimizer = require('./QuantumOptimizer');
const PredictionMonitor = require('./PredictionMonitor');
const { selfHealLearningLoop } = require('./SelfHealLearningLoop');
const { generateExplainabilityReport } = require('./ExplainabilityReporter');
const schedule = require('node-cron');

// Schedule self-heal learning loop every hour
schedule.schedule('0 * * * *', async () => {
  await selfHealLearningLoop();
});

class ASIMasterEngine {
  constructor(options = {}) {
    this.config = {
      // Complexity thresholds
      basicThreshold: options.basicThreshold || 0.3,
      generalThreshold: options.generalThreshold || 0.6,
      superThreshold: options.superThreshold || 0.8,
      
      // Performance settings
      qualityThreshold: options.qualityThreshold || 0.8,
      maxRetries: options.maxRetries || 2,
      adaptiveLearning: options.adaptiveLearning !== false,
      
      // Resource management
      maxConcurrentRequests: options.maxConcurrentRequests || 10,
      timeoutMs: options.timeoutMs || 30000,
      
      ...options
    };

    // Initialize capability layers
    this.capabilities = {
      basic: null,      // Simple ML tasks
      general: null,    // Cross-domain reasoning
      super: null,      // Advanced optimization
      quantum: null     // Maximum capability
    };

    // Decision and routing components
    this.decisionRouter = null;
    this.capabilityManager = null;
    this.performanceMonitor = null;
    
    // Request processing
    this.activeRequests = new Map();
    this.requestHistory = [];
    this.performanceMetrics = new Map();
    
    // Learning and adaptation
    this.adaptationEngine = null;
    this.knowledgeBase = new Map();
    this.webResearchAgent = new WebResearchAgent();
    this.curriculum = null;

    // Initialize curriculum
    this.curriculum = { stages: [] };
  }

  /**
   * Initialize ASI Master Engine
   */
  async initialize() {
    try {
      logger.info('🚀 Initializing ASI Master Engine - Universal Intelligence System...');

      await tf.ready();

      // Initialize capability layers
      await this.initializeCapabilities();
      
      // Initialize decision routing
      await this.initializeDecisionRouter();
      
      // Initialize capability management
      await this.initializeCapabilityManager();
      
      // Initialize performance monitoring
      await this.initializePerformanceMonitor();
      
      // Initialize adaptation engine
      await this.initializeAdaptationEngine();

      logger.info('✅ ASI Master Engine initialized - Ready to handle all intelligence requests');

    } catch (error) {
      logger.error('❌ ASI Master Engine initialization failed:', error);
      throw error;
    }
  }

  /**
   * Initialize all capability layers
   */
  async initializeCapabilities() {
    logger.info('🧠 Initializing ASI capability layers...');

    // Basic capability - Simple ML and data processing
    this.capabilities.basic = {
      continuousLearning: new ContinuousLearningEngine(),
      mutualFundAnalyzer: new MutualFundAnalyzer(),
      realTimeDataFeeds: new RealTimeDataFeeds(),
      backtestingFramework: new BacktestingFramework(),
      alphaModel: new AlphaModel(),
      riskModel: new RiskModel(),
      deepForecasting: new DeepForecastingModel()
    };

    // General capability - Cross-domain intelligence
    this.capabilities.general = {
      agiEngine: new AGIEngine(),
      behavioralFinance: new BehavioralFinanceEngine(),
      explainability: new ExplainabilityEngine()
    };

    // Super capability - Advanced optimization
    this.capabilities.super = {
      reinforcementLearning: new ReinforcementLearningEngine(),
      autonomousLearning: new AutonomousLearningSystem(),
      portfolioOptimizer: new QuantumPortfolioOptimizer()
    };

    // Quantum capability - Maximum processing power
    this.capabilities.quantum = {
      quantumOptimizer: new QuantumInspiredOptimizer(),
      quantumPortfolioOptimizer: new QuantumPortfolioOptimizer()
    };

    // Initialize all capabilities
    await Promise.all([
      this.capabilities.basic.continuousLearning.initialize(),
      this.capabilities.basic.mutualFundAnalyzer.initialize(),
      this.capabilities.basic.realTimeDataFeeds.initialize(),
      this.capabilities.basic.backtestingFramework.initialize(),
      this.capabilities.basic.alphaModel.initialize?.(),
      this.capabilities.basic.riskModel.initialize?.(),
      this.capabilities.basic.deepForecasting.initialize?.(),
      this.capabilities.general.agiEngine.initialize(),
      this.capabilities.general.behavioralFinance.initialize(),
      this.capabilities.general.explainability.initialize?.(),
      this.capabilities.super.reinforcementLearning.initialize(),
      this.capabilities.super.autonomousLearning.initialize(),
      this.capabilities.super.portfolioOptimizer.initialize?.(),
      this.capabilities.quantum.quantumOptimizer.initialize(),
      this.capabilities.quantum.quantumPortfolioOptimizer.initialize?.()
    ]);

    logger.info('✅ All ASI capability layers initialized');
  }

  /**
   * Initialize decision routing system
   */
  async initializeDecisionRouter() {
    this.decisionRouter = {
      // Request analysis patterns
      patterns: new Map([
        ['fund_analysis', { complexity: 0.2, domains: ['finance'], optimal: 'basic' }],
        ['nav_prediction', { complexity: 0.3, domains: ['finance', 'ml'], optimal: 'basic' }],
        ['portfolio_optimization', { complexity: 0.7, domains: ['finance', 'optimization'], optimal: 'super' }],
        ['market_sentiment', { complexity: 0.5, domains: ['finance', 'psychology'], optimal: 'general' }],
        ['quantum_optimization', { complexity: 0.9, domains: ['finance', 'quantum'], optimal: 'quantum' }],
        ['behavioral_analysis', { complexity: 0.6, domains: ['finance', 'psychology'], optimal: 'general' }],
        ['autonomous_learning', { complexity: 0.8, domains: ['ml', 'meta-learning'], optimal: 'super' }]
      ]),
      
      // Dynamic complexity assessment
      complexityFactors: {
        dataSize: { small: 0.1, medium: 0.3, large: 0.5, huge: 0.7 },
        domains: { single: 0.1, dual: 0.3, multi: 0.5, cross: 0.7 },
        accuracy: { low: 0.1, medium: 0.3, high: 0.5, critical: 0.7 },
        time: { relaxed: 0.1, normal: 0.3, urgent: 0.5, critical: 0.7 }
      }
    };
  }

  /**
   * Initialize capability management
   */
  async initializeCapabilityManager() {
    this.capabilityManager = {
      // Capability hierarchy
      hierarchy: ['basic', 'general', 'super', 'quantum'],
      
      // Performance tracking per capability
      performance: new Map([
        ['basic', { successRate: 0.95, avgTime: 100, quality: 0.85 }],
        ['general', { successRate: 0.90, avgTime: 500, quality: 0.90 }],
        ['super', { successRate: 0.85, avgTime: 2000, quality: 0.95 }],
        ['quantum', { successRate: 0.80, avgTime: 5000, quality: 0.98 }]
      ]),
      
      // Capability selection logic
      selectionStrategy: 'adaptive' // adaptive, conservative, aggressive
    };
  }

  /**
   * Initialize performance monitoring
   */
  async initializePerformanceMonitor() {
    this.performanceMonitor = {
      metrics: {
        totalRequests: 0,
        successfulRequests: 0,
        failedRequests: 0,
        avgResponseTime: 0,
        capabilityUsage: new Map(),
        qualityScores: []
      },
      
      // Real-time monitoring
      activeMonitoring: true,
      alertThresholds: {
        failureRate: 0.1,
        responseTime: 10000,
        qualityScore: 0.7
      }
    };
  }

  /**
   * Initialize adaptation engine
   */
  async initializeAdaptationEngine() {
    this.adaptationEngine = {
      // Learning from request patterns
      patternLearning: true,
      
      // Capability performance optimization
      capabilityOptimization: true,
      
      // Dynamic threshold adjustment
      thresholdAdaptation: true,
      
      // Knowledge accumulation
      knowledgeAccumulation: true
    };
  }

  /**
   * UNIVERSAL INTELLIGENCE INTERFACE
   * Single method that handles ALL requests
   */
  async processRequest(request) {
    // 1. Compliance check for all requests
    const compliance = complianceCheck({ type: request.type, ...request });
    // Log compliance check for audit trail (for dashboard)
    try {
      const { logComplianceAudit } = require('./ComplianceAuditDashboard');
      logComplianceAudit({
        id: Date.now() + '-' + Math.random(),
        actionType: 'asi_request',
        user: request.user,
        path: request.path || request.type,
        method: request.method || 'ASI',
        result: compliance.compliant ? 'pass' : 'fail',
        violations: compliance.violations,
        timestamp: Date.now()
      });
    } catch (e) { logger.warn('Compliance audit log failed', e); }
    if (!compliance.compliant) {
      logger.error('❌ Compliance violation:', compliance.violations);
      throw new Error('Compliance violation: ' + compliance.violations.join(','));
    }

    // 2. Prediction/analysis: enforce backtest and prediction monitoring
    if ([
      'predict_fund_return', 'predict_stock_return', 'compare_funds', 'compare_stocks'
    ].includes(request.type)) {
      // Backtest before prediction
      const backtestResult = await this.backtestBeforePredict(request);
      if (backtestResult && backtestResult.accuracy < 0.8) {
        logger.warn('Prediction blocked: backtest accuracy below threshold');
        throw new Error('Backtest accuracy below threshold. Prediction not allowed.');
      }
      // Proceed with prediction and store
      const prediction = await this.predictAndStore(request);
      // Generate explainability report for prediction
      generateExplainabilityReport({
        id: prediction.id,
        type: request.type,
        input: request,
        output: prediction,
        model: 'ASI-Predictor',
        rationale: 'Prediction made after passing compliance and backtest checks.',
        compliance,
        timestamp: Date.now()
      });
      return prediction;
    }

    // 3. Quantum optimization for relevant types
    if ([
      'portfolio_optimization', 'quantum_optimization', 'schedule_optimization', 'crawl_priority'
    ].includes(request.type)) {
      const result = await this.quantumOptimize(request);
      // Generate explainability report for optimization
      generateExplainabilityReport({
        id: Date.now() + '-' + Math.random(),
        type: request.type,
        input: request,
        output: result,
        model: 'QuantumOptimizer',
        rationale: 'Optimization performed by quantum-inspired module.',
        compliance,
        timestamp: Date.now()
      });
      return result;
    }

    // 4. Default: legacy universal intelligence flow
    const result = await this.processUniversal(request);
    // Generate explainability report for generic actions
    generateExplainabilityReport({
      id: Date.now() + '-' + Math.random(),
      type: request.type,
      input: request,
      output: result,
      model: 'ASI-Universal',
      rationale: 'Processed by universal intelligence flow.',
      compliance,
      timestamp: Date.now()
    });
    return result;
  }
    const requestId = this.generateRequestId();
    const startTime = Date.now();

    try {
      logger.info(`🧠 ASI processing request ${requestId}: ${request.type}`);

      // Track active request
      this.activeRequests.set(requestId, { request, startTime });

      // Step 1: Analyze request complexity
      const complexity = await this.analyzeComplexity(request);
      
      // Step 2: Select optimal capability
      const capability = await this.selectOptimalCapability(complexity, request);
      
      // Step 3: Execute with selected capability
      const result = await this.executeWithCapability(request, capability, requestId);
      
      // Step 4: Validate result quality
      const quality = await this.validateResultQuality(result, request);
      
      // Step 5: Auto-retry with higher capability if needed
      const finalResult = await this.handleQualityValidation(result, quality, request, capability, requestId);
      
      // Step 6: Update performance metrics
      await this.updatePerformanceMetrics(requestId, capability, finalResult, startTime);
      
      // Step 7: Learn from this request
      await this.learnFromRequest(request, capability, finalResult, quality);

      // Clean up
      this.activeRequests.delete(requestId);

      logger.info(`✅ ASI completed request ${requestId} with ${capability} capability`);
      
      return {
        requestId,
        result: finalResult,
        capability: capability,
        quality: quality,
        processingTime: Date.now() - startTime,
        metadata: {
          complexity: complexity,
          retries: finalResult.retries || 0
        }
      };

    } catch (error) {
      logger.error(`❌ ASI request ${requestId} failed:`, error);
      
      // Clean up
      this.activeRequests.delete(requestId);
      
      // Update failure metrics
      this.performanceMonitor.metrics.failedRequests++;
      
      throw error;
    }
  }

  /**
   * Analyze request complexity using multiple factors
   */
  async analyzeComplexity(request) {
    const factors = {
      // Data complexity
      dataComplexity: this.assessDataComplexity(request.data),
      
      // Domain complexity
      domainComplexity: this.assessDomainComplexity(request.type),
      
      // Computational needs
      computationalNeeds: this.assessComputationalNeeds(request.parameters),
      
      // Accuracy requirements
      accuracyRequirements: this.assessAccuracyNeeds(request.precision),
      
      // Time constraints
      timeConstraints: this.assessTimeConstraints(request.urgency),
      
      // Historical patterns
      historicalPattern: this.getHistoricalComplexity(request.type)
    };

    // Calculate weighted complexity score
    const weights = {
      dataComplexity: 0.2,
      domainComplexity: 0.25,
      computationalNeeds: 0.2,
      accuracyRequirements: 0.15,
      timeConstraints: 0.1,
      historicalPattern: 0.1
    };

    const complexityScore = Object.keys(factors).reduce((score, factor) => {
      return score + (factors[factor] * weights[factor]);
    }, 0);

    return {
      score: Math.min(1.0, Math.max(0.0, complexityScore)),
      factors,
      recommendation: this.getComplexityRecommendation(complexityScore)
    };
  }

  /**
   * Select optimal capability based on complexity and constraints
   */
  async selectOptimalCapability(complexity, request) {
    const { score } = complexity;
    
    // Get pattern-based recommendation
    const patternRecommendation = this.decisionRouter.patterns.get(request.type);
    
    // Calculate optimal capability
    let optimalCapability;
    
    if (patternRecommendation) {
      optimalCapability = patternRecommendation.optimal;
    } else if (score < this.config.basicThreshold) {
      optimalCapability = 'basic';
    } else if (score < this.config.generalThreshold) {
      optimalCapability = 'general';
    } else if (score < this.config.superThreshold) {
      optimalCapability = 'super';
    } else {
      optimalCapability = 'quantum';
    }

    // Consider resource constraints
    if (request.urgency === 'critical' && optimalCapability === 'quantum') {
      optimalCapability = 'super'; // Quantum might be too slow for critical requests
    }

    // Consider accuracy requirements
    if (request.precision === 'critical' && optimalCapability === 'basic') {
      optimalCapability = 'general'; // Basic might not be accurate enough
    }

    return optimalCapability;
  }

  /**
   * Execute request with selected capability
   */
  async executeWithCapability(request, capability, requestId) {
    const capabilityEngine = this.capabilities[capability];
    
    logger.info(`🔧 Executing request ${requestId} with ${capability} capability`);

    let result;

    switch (capability) {
      case 'basic':
        result = await this.executeBasicCapability(request, capabilityEngine);
        break;
        
      case 'general':
        result = await this.executeGeneralCapability(request, capabilityEngine);
        break;
        
      case 'super':
        result = await this.executeSuperCapability(request, capabilityEngine);
        break;
        
      case 'quantum':
        result = await this.executeQuantumCapability(request, capabilityEngine);
        break;
        
      default:
        throw new Error(`Unknown capability: ${capability}`);
    }

    return { ...result, capability, requestId };
  }

  /**
   * Execute with basic capability
   */
  async executeBasicCapability(request, engines) {
    switch (request.type) {
      case 'chatbot':
        return await this.handleChatbot(request, engines);

      case 'fund_analysis':
        return await engines.mutualFundAnalyzer.analyzeFund(
          request.data.fundCode, 
          request.parameters?.includeHistory
        );
        
      case 'nav_prediction':
        return await engines.continuousLearning.predictNAV(
          request.data.fundCode, 
          request.data.history
        );
        
      case 'alpha_signal':
        return await engines.alphaModel.generateAlpha(request.data);
        
      case 'risk_analysis':
        return await engines.riskModel.computeRisk(request.data);
        
      case 'deep_forecast':
        return await engines.deepForecasting.predictSequence(request.data);
        
      case 'market_data':
        return await engines.realTimeDataFeeds.collectMarketData(request.parameters);
        
      case 'backtest':
        return await engines.backtestingFramework.runBacktest(
          request.data.strategy,
          request.parameters
        );
        
      default:
        // Default to fund analysis for unknown basic requests
        return await engines.mutualFundAnalyzer.analyzeFund(request.data.fundCode);
    }
  }

  /**
   * Execute with general capability
   */
  async executeGeneralCapability(request, engines) {
    switch (request.type) {
      case 'market_sentiment':
      case 'behavioral_analysis':
        return await engines.behavioralFinance.analyzeBehavioralFactors(
          request.data,
          request.parameters?.marketData,
          request.parameters?.userProfile
        );
        
      case 'cross_domain_analysis':
        return await engines.agiEngine.performCrossDomainAnalysis(request.data);
        
      case 'reasoning':
        return await engines.agiEngine.performReasoning(request.data, request.parameters);
        
      case 'explainability':
        return await engines.explainability.explain(request.data);
        
      case 'quantum_portfolio_optimization':
        return await engines.quantumOptimizer.optimizePortfolio(request.data);
        
      default:
        // Default to AGI engine for unknown general requests
        return await engines.agiEngine.processGeneralRequest(request);
    }
  }

  /**
   * Execute with super capability
   */
  async executeSuperCapability(request, engines) {
    switch (request.type) {
      case 'portfolio_optimization':
        return await engines.portfolioOptimizer.optimize(request.data);
        
      case 'strategy_optimization':
        return await engines.reinforcementLearning.trainMultiAgentSystem(
          request.parameters?.episodes || 1000
        );
        
      case 'autonomous_learning':
        return await engines.autonomousLearning.autonomousLearningLoop(
          request.data,
          request.parameters?.performanceMetrics
        );
        
      case 'meta_learning':
        return await engines.autonomousLearning.metaLearn(request.data.tasks);
        
      default:
        // Default to autonomous learning for unknown super requests
        return await engines.autonomousLearning.autonomousLearningLoop(
          request.data,
          request.parameters
        );
    }
  }

  /**
   * Execute with quantum capability
   */
  async executeQuantumCapability(request, engines) {
    switch (request.type) {
      case 'quantum_optimization':
        return await engines.quantumOptimizer.optimizeWithQAOA(
          request.data.expectedReturns,
          request.data.covarianceMatrix,
          request.parameters?.constraints
        );
        
      case 'quantum_portfolio_optimization':
        return await engines.quantumPortfolioOptimizer.optimize(request.data);
        
      case 'quantum_annealing':
        return await engines.quantumOptimizer.optimizeWithQuantumAnnealing(
          request.data.expectedReturns,
          request.data.covarianceMatrix,
          request.parameters?.constraints
        );
        
      case 'quantum_comparison':
        return await engines.quantumOptimizer.compareQuantumClassical(
          request.data.expectedReturns,
          request.data.covarianceMatrix,
          request.parameters?.constraints
        );
        
      default:
        // Default to QAOA for unknown quantum requests
        return await engines.quantumOptimizer.optimizeWithQAOA(
          request.data.expectedReturns || [],
          request.data.covarianceMatrix || [],
          request.parameters?.constraints
        );
    }
  }

  /**
   * Validate result quality
   */
  async validateResultQuality(result, request) {
    const qualityFactors = {
      completeness: this.assessCompleteness(result, request),
      accuracy: this.assessAccuracy(result, request),
      relevance: this.assessRelevance(result, request),
      timeliness: this.assessTimeliness(result, request)
    };

    const qualityScore = Object.values(qualityFactors).reduce((sum, score) => sum + score, 0) / 4;

    return {
      score: qualityScore,
      factors: qualityFactors,
      acceptable: qualityScore >= this.config.qualityThreshold
    };
  }

  /**
   * Handle quality validation and potential retries
   */
  async handleQualityValidation(result, quality, request, capability, requestId) {
    if (quality.acceptable) {
      return result;
    }

    // If quality is insufficient, try higher capability
    const currentIndex = this.capabilityManager.hierarchy.indexOf(capability);
    const nextCapability = this.capabilityManager.hierarchy[currentIndex + 1];

    if (nextCapability && (result.retries || 0) < this.config.maxRetries) {
      logger.info(`🔄 Retrying request ${requestId} with ${nextCapability} capability`);
      
      const retryResult = await this.executeWithCapability(request, nextCapability, requestId);
      retryResult.retries = (result.retries || 0) + 1;
      
      const retryQuality = await this.validateResultQuality(retryResult, request);
      
      if (retryQuality.acceptable) {
        return retryResult;
      }
    }

    // Return original result with quality warning
    result.qualityWarning = {
      score: quality.score,
      threshold: this.config.qualityThreshold,
      factors: quality.factors
    };

    return result;
  }

  // Helper methods for complexity assessment
  assessDataComplexity(data) {
    if (!data) return 0.1;
    const size = JSON.stringify(data).length;
    if (size < 1000) return 0.1;
    if (size < 10000) return 0.3;
    if (size < 100000) return 0.5;
    return 0.7;
  }

  assessDomainComplexity(type) {
    const domainMap = {
      'fund_analysis': 0.2,
      'nav_prediction': 0.3,
      'portfolio_optimization': 0.7,
      'quantum_optimization': 0.9,
      'behavioral_analysis': 0.6,
      'autonomous_learning': 0.8
    };
    return domainMap[type] || 0.5;
  }

  assessComputationalNeeds(parameters) {
    if (!parameters) return 0.2;
    const complexity = parameters.complexity || 'medium';
    const complexityMap = { low: 0.1, medium: 0.3, high: 0.6, extreme: 0.9 };
    return complexityMap[complexity] || 0.3;
  }

  assessAccuracyNeeds(precision) {
    const precisionMap = { low: 0.1, standard: 0.3, high: 0.6, critical: 0.9 };
    return precisionMap[precision] || 0.3;
  }

  assessTimeConstraints(urgency) {
    const urgencyMap = { relaxed: 0.1, normal: 0.3, urgent: 0.6, critical: 0.9 };
    return urgencyMap[urgency] || 0.3;
  }

  getHistoricalComplexity(type) {
    // Return average complexity from historical data
    const history = this.requestHistory.filter(r => r.type === type);
    if (history.length === 0) return 0.5;
    
    const avgComplexity = history.reduce((sum, r) => sum + r.complexity.score, 0) / history.length;
    return avgComplexity;
  }

  getComplexityRecommendation(score) {
    if (score < 0.3) return 'basic';
    if (score < 0.6) return 'general';
    if (score < 0.8) return 'super';
    return 'quantum';
  }

  // Quality assessment helpers
  assessCompleteness(result, request) { return result && Object.keys(result).length > 0 ? 0.9 : 0.3; }
  assessAccuracy(result, request) { return 0.8; } // Placeholder
  assessRelevance(result, request) { return 0.8; } // Placeholder
  assessTimeliness(result, request) { return 0.9; } // Placeholder

  // Performance tracking
  async updatePerformanceMetrics(requestId, capability, result, startTime) {
    const processingTime = Date.now() - startTime;
    
    this.performanceMonitor.metrics.totalRequests++;
    this.performanceMonitor.metrics.successfulRequests++;
    
    // Update capability usage
    const currentUsage = this.performanceMonitor.metrics.capabilityUsage.get(capability) || 0;
    this.performanceMonitor.metrics.capabilityUsage.set(capability, currentUsage + 1);
    
    // Update average response time
    const totalTime = this.performanceMonitor.metrics.avgResponseTime * 
                     (this.performanceMonitor.metrics.totalRequests - 1) + processingTime;
    this.performanceMonitor.metrics.avgResponseTime = totalTime / this.performanceMonitor.metrics.totalRequests;
  }

  // Learning and adaptation
  async learnFromRequest(request, capability, result, quality) {
    if (!this.adaptationEngine.patternLearning) return;

    // Store request pattern
    this.requestHistory.push({
      type: request.type,
      capability,
      quality: quality.score,
      timestamp: Date.now(),
      complexity: await this.analyzeComplexity(request)
    });

    // Keep only recent history
    if (this.requestHistory.length > 1000) {
      this.requestHistory = this.requestHistory.slice(-1000);
    }

    // Update knowledge base
    this.knowledgeBase.set(request.type, {
      optimalCapability: capability,
      averageQuality: quality.score,
      lastUpdated: Date.now()
    });
  }

  // Utility methods
  generateRequestId() {
    return `asi_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get comprehensive metrics
   */
  getMetrics() {
    return {
      performance: this.performanceMonitor.metrics,
      capabilities: {
        basic: this.capabilities.basic ? 'initialized' : 'not_initialized',
        general: this.capabilities.general ? 'initialized' : 'not_initialized',
        super: this.capabilities.super ? 'initialized' : 'not_initialized',
        quantum: this.capabilities.quantum ? 'initialized' : 'not_initialized'
      },
      activeRequests: this.activeRequests.size,
      requestHistory: this.requestHistory.length,
      knowledgeBase: this.knowledgeBase.size,
      config: this.config,
      memoryUsage: process.memoryUsage(),
      tfMemory: tf.memory()
    };
  }

  /**
   * Health check
   */
  async getHealthStatus() {
    try {
      const capabilities = await Promise.all([
        this.capabilities.basic ? 'healthy' : 'unhealthy',
        this.capabilities.general ? 'healthy' : 'unhealthy',
        this.capabilities.super ? 'healthy' : 'unhealthy',
        this.capabilities.quantum ? 'healthy' : 'unhealthy'
      ]);

      const overallHealth = capabilities.every(status => status === 'healthy') ? 'healthy' : 'degraded';

      return {
        status: overallHealth,
        capabilities: {
          basic: capabilities[0],
          general: capabilities[1],
          super: capabilities[2],
          quantum: capabilities[3]
        },
        metrics: this.performanceMonitor.metrics,
        activeRequests: this.activeRequests.size
      };

    } catch (error) {
      return {
        status: 'unhealthy',
        error: error.message
      };
    }
  }

  /**
   * Mistake-driven learning: When a mistake is detected, research and update knowledge base.
   * @param {string} errorContext - Description of the mistake or failure
   */
  async learnFromMistake(errorContext) {
    const { summary, links } = await this.webResearchAgent.searchAndSummarize(errorContext);
    this.knowledgeBase.set(errorContext, { summary, links, source: 'web', timestamp: Date.now() });
    // Optionally, update curriculum or trigger further learning
  }

  /**
   * Autonomous curriculum building: Expand curriculum from web trends or gaps.
   * @param {string} topicSeed - Topic or gap to expand from
   */
  async buildCurriculumFromWeb(topicSeed) {
    const topics = await this.webResearchAgent.extractTopicsFromWeb(topicSeed);
    if (!this.curriculum) this.curriculum = { stages: [] };
    for (const topic of topics) {
      if (!this.curriculum.stages.find(s => s.name === topic)) {
        this.curriculum.stages.push({ name: topic, prerequisites: [], difficulty: 0.5 });
      }
    }
  }

  /**
   * Recursive, curiosity-driven learning loop: Identify gaps, research, and deepen understanding.
   */
  async autonomousLearningLoop(maxDepth = 3) {
    let depth = 0;
    while (depth < maxDepth) {
      // 1. Identify gaps (simple: missing curriculum, low knowledge)
      const gaps = this.identifyKnowledgeGaps();
      for (const gap of gaps) {
        await this.learnFromMistake(gap);
      }
      // 2. Expand curriculum from gaps
      for (const gap of gaps) {
        await this.buildCurriculumFromWeb(gap);
      }
      depth++;
    }
  }

  /**
   * Identify knowledge gaps (simple: curriculum items not in knowledgeBase)
   * @returns {string[]} List of gap topics
   */
  identifyKnowledgeGaps() {
    if (!this.curriculum || !Array.isArray(this.curriculum.stages)) return [];
    return this.curriculum.stages
      .map(s => s.name)
      .filter(name => !this.knowledgeBase.has(name));
  }

  /**
   * Handle chatbot requests - routes chatbot messages to the right intelligence models
   * @param {object} request - The request object with message, user context, etc.
   * @param {object} engines - The capability engines for this layer
   * @returns {object} - Chatbot response
   */
  async handleChatbot(request, engines) {
    const message = request.data?.message || '';
    const userProfile = request.data?.userProfile || {};
    const context = request.parameters?.context || '';
    // TODO: Add NLP intent detection and smart routing to models
    // Placeholder: Echo and log
    logger.info(`[Chatbot] Received message: '${message}' | Context: '${context}' | User: ${JSON.stringify(userProfile)}`);
    // Example: you could route to fund analysis, alpha, etc. based on intent
    return {
      success: true,
      reply: `🤖 ASI Chatbot: You said: '${message}' (context: ${context})`,
      context,
      actions: [],
      meta: {
        routedModel: 'placeholder',
        userProfile
      }
    };
  }
}

module.exports = { ASIMasterEngine };
