/**
 * 🛡️ ADVANCED RISK MANAGEMENT SYSTEM
 * 
 * Comprehensive risk assessment, monitoring, and mitigation for financial portfolios
 * Real-time risk analytics, stress testing, and automated risk controls
 * 
 * @author 35+ Years ASI Engineering Experience
 * @version 4.0.0 - Enterprise Risk Management
 */

const EventEmitter = require('events');
const logger = require('../utils/logger');

class RiskManagementSystem extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      confidenceLevel: options.confidenceLevel || 0.95,
      timeHorizon: options.timeHorizon || 252, // Trading days in a year
      maxPortfolioRisk: options.maxPortfolioRisk || 0.2, // 20% max portfolio risk
      stressTestScenarios: options.stressTestScenarios || 100,
      monitoringInterval: options.monitoringInterval || 300000, // 5 minutes
      alertThresholds: {
        var: options.alertThresholds?.var || 0.05, // 5% VaR threshold
        drawdown: options.alertThresholds?.drawdown || 0.15, // 15% drawdown threshold
        concentration: options.alertThresholds?.concentration || 0.25, // 25% concentration threshold
        leverage: options.alertThresholds?.leverage || 2.0, // 2x leverage threshold
        ...options.alertThresholds
      },
      ...options
    };
    
    // Risk models and calculators
    this.riskModels = {
      var: null,
      cvar: null,
      stressTesting: null,
      correlationMatrix: null,
      volatilityForecasting: null
    };
    
    // Risk metrics storage
    this.riskMetrics = new Map();
    this.historicalRisks = new Map();
    this.stressTestResults = new Map();
    
    // Risk limits and controls
    this.riskLimits = new Map();
    this.riskAlerts = [];
    
    // Market data for risk calculations
    this.marketData = new Map();
    this.correlationData = new Map();
    
    // Performance tracking
    this.metrics = {
      portfoliosMonitored: 0,
      riskAlertsGenerated: 0,
      stressTestsRun: 0,
      averageVaR: 0,
      riskAdjustedReturns: 0
    };
    
    this.isInitialized = false;
    this.monitoringActive = false;
  }

  async initialize() {
    try {
      logger.info('🛡️ Initializing Advanced Risk Management System...');
      
      // Initialize risk models
      await this.initializeRiskModels();
      
      // Initialize stress testing scenarios
      await this.initializeStressTestScenarios();
      
      // Initialize correlation models
      await this.initializeCorrelationModels();
      
      // Initialize risk limits
      await this.initializeRiskLimits();
      
      // Start risk monitoring
      this.startRiskMonitoring();
      
      this.isInitialized = true;
      logger.info('✅ Advanced Risk Management System initialized successfully');
      
    } catch (error) {
      logger.error('❌ Risk Management System initialization failed:', error);
      throw error;
    }
  }

  async initializeRiskModels() {
    // Value at Risk (VaR) model
    this.riskModels.var = {
      // Historical simulation VaR
      historicalVaR: (returns, confidenceLevel = 0.95) => {
        const sortedReturns = returns.sort((a, b) => a - b);
        const index = Math.floor((1 - confidenceLevel) * sortedReturns.length);
        return -sortedReturns[index];
      },
      
      // Parametric VaR (assuming normal distribution)
      parametricVaR: (returns, confidenceLevel = 0.95) => {
        const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
        const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1);
        const stdDev = Math.sqrt(variance);
        
        // Z-score for confidence level
        const zScore = this.getZScore(confidenceLevel);
        return -(mean - zScore * stdDev);
      },
      
      // Monte Carlo VaR
      monteCarloVaR: async (portfolio, scenarios = 10000, confidenceLevel = 0.95) => {
        const simulatedReturns = [];
        
        for (let i = 0; i < scenarios; i++) {
          const portfolioReturn = await this.simulatePortfolioReturn(portfolio);
          simulatedReturns.push(portfolioReturn);
        }
        
        return this.riskModels.var.historicalVaR(simulatedReturns, confidenceLevel);
      }
    };
    
    // Conditional Value at Risk (CVaR)
    this.riskModels.cvar = {
      calculate: (returns, confidenceLevel = 0.95) => {
        const var95 = this.riskModels.var.historicalVaR(returns, confidenceLevel);
        const tailReturns = returns.filter(r => r <= -var95);
        
        if (tailReturns.length === 0) return var95;
        
        return -tailReturns.reduce((sum, r) => sum + r, 0) / tailReturns.length;
      }
    };
    
    // Stress testing model
    this.riskModels.stressTesting = {
      scenarios: [
        { name: '2008 Financial Crisis', shocks: { equity: -0.4, bonds: -0.1, commodities: -0.3 } },
        { name: 'COVID-19 Pandemic', shocks: { equity: -0.35, bonds: 0.05, commodities: -0.25 } },
        { name: 'Interest Rate Shock', shocks: { equity: -0.15, bonds: -0.2, commodities: -0.1 } },
        { name: 'Inflation Spike', shocks: { equity: -0.2, bonds: -0.15, commodities: 0.1 } },
        { name: 'Geopolitical Crisis', shocks: { equity: -0.25, bonds: 0.02, commodities: 0.15 } }
      ],
      
      runStressTest: async (portfolio, scenario) => {
        let totalImpact = 0;
        
        for (const [assetClass, weight] of Object.entries(portfolio.allocation)) {
          const shock = scenario.shocks[assetClass] || 0;
          totalImpact += weight * shock;
        }
        
        return {
          scenario: scenario.name,
          portfolioImpact: totalImpact,
          newValue: portfolio.value * (1 + totalImpact),
          riskContribution: this.calculateRiskContribution(portfolio, scenario)
        };
      }
    };
    
    logger.info('📊 Risk models initialized');
  }

  async initializeStressTestScenarios() {
    // Custom stress test scenarios
    this.stressTestScenarios = [
      {
        name: 'Market Crash',
        probability: 0.05,
        shocks: { equity: -0.5, bonds: -0.1, real_estate: -0.3, commodities: -0.2 }
      },
      {
        name: 'Recession',
        probability: 0.15,
        shocks: { equity: -0.3, bonds: 0.1, real_estate: -0.2, commodities: -0.15 }
      },
      {
        name: 'Hyperinflation',
        probability: 0.02,
        shocks: { equity: -0.2, bonds: -0.4, real_estate: 0.1, commodities: 0.3 }
      },
      {
        name: 'Currency Crisis',
        probability: 0.08,
        shocks: { equity: -0.25, bonds: -0.1, real_estate: -0.15, commodities: 0.05 }
      },
      {
        name: 'Technology Bubble Burst',
        probability: 0.1,
        shocks: { equity: -0.4, bonds: 0.05, real_estate: -0.1, commodities: 0.0 }
      }
    ];
    
    logger.info(`🎭 ${this.stressTestScenarios.length} stress test scenarios initialized`);
  }

  async initializeCorrelationModels() {
    // Initialize correlation matrix calculator
    this.riskModels.correlationMatrix = {
      calculate: (returns) => {
        const assets = Object.keys(returns);
        const correlationMatrix = {};
        
        for (const asset1 of assets) {
          correlationMatrix[asset1] = {};
          for (const asset2 of assets) {
            correlationMatrix[asset1][asset2] = this.calculateCorrelation(
              returns[asset1],
              returns[asset2]
            );
          }
        }
        
        return correlationMatrix;
      },
      
      // Dynamic correlation with GARCH-like updating
      updateCorrelations: (currentCorrelations, newReturns, decayFactor = 0.94) => {
        // Exponentially weighted moving average for correlations
        const updatedCorrelations = {};
        
        for (const [asset1, correlations] of Object.entries(currentCorrelations)) {
          updatedCorrelations[asset1] = {};
          for (const [asset2, oldCorr] of Object.entries(correlations)) {
            const newCorr = this.calculateCorrelation([newReturns[asset1]], [newReturns[asset2]]);
            updatedCorrelations[asset1][asset2] = decayFactor * oldCorr + (1 - decayFactor) * newCorr;
          }
        }
        
        return updatedCorrelations;
      }
    };
    
    // Volatility forecasting model
    this.riskModels.volatilityForecasting = {
      // GARCH(1,1) model approximation
      garch: (returns, alpha = 0.1, beta = 0.85, omega = 0.00001) => {
        const forecasts = [];
        let variance = this.calculateVariance(returns.slice(0, 30)); // Initial variance
        
        for (let i = 30; i < returns.length; i++) {
          const prevReturn = returns[i - 1];
          variance = omega + alpha * Math.pow(prevReturn, 2) + beta * variance;
          forecasts.push(Math.sqrt(variance));
        }
        
        return forecasts;
      },
      
      // Exponentially weighted moving average
      ewma: (returns, lambda = 0.94) => {
        const forecasts = [];
        let variance = Math.pow(returns[0], 2);
        
        for (let i = 1; i < returns.length; i++) {
          variance = lambda * variance + (1 - lambda) * Math.pow(returns[i - 1], 2);
          forecasts.push(Math.sqrt(variance));
        }
        
        return forecasts;
      }
    };
    
    logger.info('📈 Correlation and volatility models initialized');
  }

  async initializeRiskLimits() {
    // Default risk limits
    this.riskLimits.set('portfolio_var', this.config.alertThresholds.var);
    this.riskLimits.set('max_drawdown', this.config.alertThresholds.drawdown);
    this.riskLimits.set('concentration_limit', this.config.alertThresholds.concentration);
    this.riskLimits.set('leverage_limit', this.config.alertThresholds.leverage);
    this.riskLimits.set('sector_concentration', 0.3); // 30% max in any sector
    this.riskLimits.set('single_position', 0.1); // 10% max in any single position
    
    logger.info('⚖️ Risk limits initialized');
  }

  async assessPortfolioRisk(portfolio) {
    try {
      const startTime = Date.now();
      
      // Calculate various risk metrics
      const riskAssessment = {
        portfolioId: portfolio.id,
        timestamp: new Date(),
        metrics: {}
      };
      
      // Value at Risk calculations
      if (portfolio.returns && portfolio.returns.length > 0) {
        riskAssessment.metrics.var_95 = this.riskModels.var.historicalVaR(portfolio.returns, 0.95);
        riskAssessment.metrics.var_99 = this.riskModels.var.historicalVaR(portfolio.returns, 0.99);
        riskAssessment.metrics.cvar_95 = this.riskModels.cvar.calculate(portfolio.returns, 0.95);
        
        // Parametric VaR for comparison
        riskAssessment.metrics.parametric_var = this.riskModels.var.parametricVaR(portfolio.returns, 0.95);
      }
      
      // Portfolio concentration risk
      riskAssessment.metrics.concentration = this.calculateConcentrationRisk(portfolio);
      
      // Maximum drawdown
      riskAssessment.metrics.max_drawdown = this.calculateMaxDrawdown(portfolio.returns || []);
      
      // Volatility metrics
      riskAssessment.metrics.volatility = this.calculateVolatility(portfolio.returns || []);
      riskAssessment.metrics.annualized_volatility = riskAssessment.metrics.volatility * Math.sqrt(252);
      
      // Sharpe ratio
      riskAssessment.metrics.sharpe_ratio = this.calculateSharpeRatio(portfolio.returns || [], 0.02); // 2% risk-free rate
      
      // Beta (if benchmark provided)
      if (portfolio.benchmark) {
        riskAssessment.metrics.beta = this.calculateBeta(portfolio.returns || [], portfolio.benchmark);
      }
      
      // Risk-adjusted returns
      riskAssessment.metrics.risk_adjusted_return = this.calculateRiskAdjustedReturn(portfolio);
      
      // Overall risk score (0-100)
      riskAssessment.riskScore = this.calculateOverallRiskScore(riskAssessment.metrics);
      
      // Risk level classification
      riskAssessment.riskLevel = this.classifyRiskLevel(riskAssessment.riskScore);
      
      // Processing time
      riskAssessment.processingTime = Date.now() - startTime;
      
      // Store risk metrics
      this.riskMetrics.set(portfolio.id, riskAssessment);
      
      // Check for risk limit breaches
      await this.checkRiskLimits(portfolio.id, riskAssessment);
      
      this.metrics.portfoliosMonitored++;
      
      return riskAssessment;
      
    } catch (error) {
      logger.error('❌ Portfolio risk assessment failed:', error);
      throw error;
    }
  }

  async runStressTest(portfolio, scenarios = null) {
    try {
      const testScenarios = scenarios || this.stressTestScenarios;
      const results = [];
      
      for (const scenario of testScenarios) {
        const result = await this.riskModels.stressTesting.runStressTest(portfolio, scenario);
        result.probability = scenario.probability;
        result.expectedLoss = result.portfolioImpact * scenario.probability;
        results.push(result);
      }
      
      // Calculate aggregate stress test metrics
      const aggregateResults = {
        portfolioId: portfolio.id,
        timestamp: new Date(),
        scenarios: results,
        summary: {
          worstCaseScenario: results.reduce((worst, current) => 
            current.portfolioImpact < worst.portfolioImpact ? current : worst
          ),
          expectedLoss: results.reduce((sum, r) => sum + r.expectedLoss, 0),
          probabilityWeightedLoss: results.reduce((sum, r) => sum + (r.portfolioImpact * r.probability), 0),
          scenariosAboveThreshold: results.filter(r => Math.abs(r.portfolioImpact) > 0.1).length
        }
      };
      
      // Store stress test results
      this.stressTestResults.set(portfolio.id, aggregateResults);
      this.metrics.stressTestsRun++;
      
      logger.info(`🎭 Stress test completed for portfolio ${portfolio.id}: ${results.length} scenarios`);
      
      return aggregateResults;
      
    } catch (error) {
      logger.error('❌ Stress test failed:', error);
      throw error;
    }
  }

  async monitorRealTimeRisk(portfolio) {
    try {
      // Real-time risk monitoring
      const currentRisk = await this.assessPortfolioRisk(portfolio);
      
      // Check for immediate risk alerts
      const alerts = await this.generateRiskAlerts(portfolio.id, currentRisk);
      
      if (alerts.length > 0) {
        for (const alert of alerts) {
          this.emit('riskAlert', alert);
          logger.warn(`🚨 Risk Alert: ${alert.message}`, alert);
        }
        
        this.riskAlerts.push(...alerts);
        this.metrics.riskAlertsGenerated += alerts.length;
      }
      
      // Update historical risk data
      this.updateHistoricalRisk(portfolio.id, currentRisk);
      
      return currentRisk;
      
    } catch (error) {
      logger.error('❌ Real-time risk monitoring failed:', error);
      throw error;
    }
  }

  async generateRiskAlerts(portfolioId, riskAssessment) {
    const alerts = [];
    
    // VaR threshold breach
    if (riskAssessment.metrics.var_95 > this.riskLimits.get('portfolio_var')) {
      alerts.push({
        portfolioId,
        type: 'VAR_BREACH',
        severity: 'HIGH',
        message: `Portfolio VaR (${(riskAssessment.metrics.var_95 * 100).toFixed(2)}%) exceeds limit (${(this.riskLimits.get('portfolio_var') * 100).toFixed(2)}%)`,
        value: riskAssessment.metrics.var_95,
        threshold: this.riskLimits.get('portfolio_var'),
        timestamp: new Date()
      });
    }
    
    // Concentration risk
    if (riskAssessment.metrics.concentration > this.riskLimits.get('concentration_limit')) {
      alerts.push({
        portfolioId,
        type: 'CONCENTRATION_RISK',
        severity: 'MEDIUM',
        message: `Portfolio concentration (${(riskAssessment.metrics.concentration * 100).toFixed(2)}%) exceeds limit`,
        value: riskAssessment.metrics.concentration,
        threshold: this.riskLimits.get('concentration_limit'),
        timestamp: new Date()
      });
    }
    
    // Maximum drawdown
    if (riskAssessment.metrics.max_drawdown > this.riskLimits.get('max_drawdown')) {
      alerts.push({
        portfolioId,
        type: 'DRAWDOWN_BREACH',
        severity: 'HIGH',
        message: `Maximum drawdown (${(riskAssessment.metrics.max_drawdown * 100).toFixed(2)}%) exceeds limit`,
        value: riskAssessment.metrics.max_drawdown,
        threshold: this.riskLimits.get('max_drawdown'),
        timestamp: new Date()
      });
    }
    
    // High overall risk score
    if (riskAssessment.riskScore > 80) {
      alerts.push({
        portfolioId,
        type: 'HIGH_RISK_SCORE',
        severity: 'MEDIUM',
        message: `Overall risk score (${riskAssessment.riskScore}) indicates high risk portfolio`,
        value: riskAssessment.riskScore,
        threshold: 80,
        timestamp: new Date()
      });
    }
    
    return alerts;
  }

  calculateConcentrationRisk(portfolio) {
    if (!portfolio.holdings || portfolio.holdings.length === 0) return 0;
    
    // Calculate Herfindahl-Hirschman Index (HHI)
    const totalValue = portfolio.holdings.reduce((sum, holding) => sum + holding.value, 0);
    const hhi = portfolio.holdings.reduce((sum, holding) => {
      const weight = holding.value / totalValue;
      return sum + Math.pow(weight, 2);
    }, 0);
    
    return hhi;
  }

  calculateMaxDrawdown(returns) {
    if (returns.length === 0) return 0;
    
    let peak = 0;
    let maxDrawdown = 0;
    let cumulativeReturn = 0;
    
    for (const returnValue of returns) {
      cumulativeReturn += returnValue;
      peak = Math.max(peak, cumulativeReturn);
      const drawdown = peak - cumulativeReturn;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }
    
    return maxDrawdown;
  }

  calculateVolatility(returns) {
    if (returns.length === 0) return 0;
    
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1);
    
    return Math.sqrt(variance);
  }

  calculateSharpeRatio(returns, riskFreeRate) {
    if (returns.length === 0) return 0;
    
    const meanReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const volatility = this.calculateVolatility(returns);
    
    return volatility > 0 ? (meanReturn - riskFreeRate) / volatility : 0;
  }

  calculateBeta(portfolioReturns, benchmarkReturns) {
    if (portfolioReturns.length !== benchmarkReturns.length || portfolioReturns.length === 0) {
      return 1; // Default beta
    }
    
    const covariance = this.calculateCovariance(portfolioReturns, benchmarkReturns);
    const benchmarkVariance = this.calculateVariance(benchmarkReturns);
    
    return benchmarkVariance > 0 ? covariance / benchmarkVariance : 1;
  }

  calculateCorrelation(returns1, returns2) {
    if (returns1.length !== returns2.length || returns1.length === 0) return 0;
    
    const covariance = this.calculateCovariance(returns1, returns2);
    const std1 = this.calculateVolatility(returns1);
    const std2 = this.calculateVolatility(returns2);
    
    return (std1 > 0 && std2 > 0) ? covariance / (std1 * std2) : 0;
  }

  calculateCovariance(returns1, returns2) {
    if (returns1.length !== returns2.length || returns1.length === 0) return 0;
    
    const mean1 = returns1.reduce((sum, r) => sum + r, 0) / returns1.length;
    const mean2 = returns2.reduce((sum, r) => sum + r, 0) / returns2.length;
    
    const covariance = returns1.reduce((sum, r1, i) => {
      return sum + (r1 - mean1) * (returns2[i] - mean2);
    }, 0) / (returns1.length - 1);
    
    return covariance;
  }

  calculateVariance(returns) {
    if (returns.length === 0) return 0;
    
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    return returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1);
  }

  calculateRiskAdjustedReturn(portfolio) {
    // Simplified risk-adjusted return calculation
    const returns = portfolio.returns || [];
    if (returns.length === 0) return 0;
    
    const meanReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const volatility = this.calculateVolatility(returns);
    
    return volatility > 0 ? meanReturn / volatility : 0;
  }

  calculateOverallRiskScore(metrics) {
    let score = 0;
    let weights = 0;
    
    // VaR contribution (30% weight)
    if (metrics.var_95 !== undefined) {
      score += (metrics.var_95 * 1000) * 0.3; // Scale VaR
      weights += 0.3;
    }
    
    // Volatility contribution (25% weight)
    if (metrics.annualized_volatility !== undefined) {
      score += (metrics.annualized_volatility * 100) * 0.25;
      weights += 0.25;
    }
    
    // Concentration contribution (20% weight)
    if (metrics.concentration !== undefined) {
      score += (metrics.concentration * 100) * 0.2;
      weights += 0.2;
    }
    
    // Drawdown contribution (25% weight)
    if (metrics.max_drawdown !== undefined) {
      score += (metrics.max_drawdown * 100) * 0.25;
      weights += 0.25;
    }
    
    return weights > 0 ? Math.min(100, score / weights) : 0;
  }

  classifyRiskLevel(riskScore) {
    if (riskScore >= 80) return 'HIGH';
    if (riskScore >= 60) return 'MEDIUM-HIGH';
    if (riskScore >= 40) return 'MEDIUM';
    if (riskScore >= 20) return 'LOW-MEDIUM';
    return 'LOW';
  }

  getZScore(confidenceLevel) {
    // Approximate Z-scores for common confidence levels
    const zScores = {
      0.90: 1.28,
      0.95: 1.65,
      0.99: 2.33,
      0.999: 3.09
    };
    
    return zScores[confidenceLevel] || 1.65;
  }

  async simulatePortfolioReturn(portfolio) {
    // Simplified Monte Carlo simulation for portfolio return
    // In a real implementation, this would use more sophisticated models
    
    let portfolioReturn = 0;
    
    for (const holding of portfolio.holdings || []) {
      const weight = holding.value / portfolio.totalValue;
      const expectedReturn = holding.expectedReturn || 0;
      const volatility = holding.volatility || 0.2;
      
      // Generate random return using normal distribution approximation
      const randomReturn = expectedReturn + volatility * this.generateNormalRandom();
      portfolioReturn += weight * randomReturn;
    }
    
    return portfolioReturn;
  }

  generateNormalRandom() {
    // Box-Muller transformation for normal distribution
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  calculateRiskContribution(portfolio, scenario) {
    const contributions = {};
    
    for (const [assetClass, weight] of Object.entries(portfolio.allocation || {})) {
      const shock = scenario.shocks[assetClass] || 0;
      contributions[assetClass] = weight * shock;
    }
    
    return contributions;
  }

  async checkRiskLimits(portfolioId, riskAssessment) {
    // This method is called by assessPortfolioRisk
    // Additional risk limit checks can be added here
    return true;
  }

  updateHistoricalRisk(portfolioId, riskAssessment) {
    if (!this.historicalRisks.has(portfolioId)) {
      this.historicalRisks.set(portfolioId, []);
    }
    
    const history = this.historicalRisks.get(portfolioId);
    history.push({
      timestamp: riskAssessment.timestamp,
      riskScore: riskAssessment.riskScore,
      var95: riskAssessment.metrics.var_95,
      volatility: riskAssessment.metrics.volatility
    });
    
    // Keep only last 1000 entries
    if (history.length > 1000) {
      history.splice(0, history.length - 1000);
    }
  }

  startRiskMonitoring() {
    if (this.monitoringActive) return;
    
    this.monitoringActive = true;
    
    setInterval(async () => {
      try {
        // Monitor all portfolios with stored risk metrics
        for (const [portfolioId, riskMetrics] of this.riskMetrics) {
          // In a real implementation, you would fetch current portfolio data
          // For now, we'll just emit a monitoring event
          this.emit('riskMonitoringCycle', { portfolioId, riskMetrics });
        }
      } catch (error) {
        logger.error('❌ Risk monitoring cycle failed:', error);
      }
    }, this.config.monitoringInterval);
    
    logger.info('📊 Risk monitoring started');
  }

  getRiskMetrics(portfolioId) {
    return this.riskMetrics.get(portfolioId);
  }

  getStressTestResults(portfolioId) {
    return this.stressTestResults.get(portfolioId);
  }

  getHistoricalRisk(portfolioId) {
    return this.historicalRisks.get(portfolioId) || [];
  }

  getAllRiskAlerts() {
    return this.riskAlerts;
  }

  getSystemMetrics() {
    return {
      ...this.metrics,
      portfoliosTracked: this.riskMetrics.size,
      activeAlerts: this.riskAlerts.filter(alert => 
        Date.now() - alert.timestamp.getTime() < 24 * 60 * 60 * 1000
      ).length,
      isInitialized: this.isInitialized,
      monitoringActive: this.monitoringActive
    };
  }

  async shutdown() {
    try {
      logger.info('🛑 Shutting down Risk Management System...');
      
      this.monitoringActive = false;
      
      logger.info('✅ Risk Management System shutdown completed');
      
    } catch (error) {
      logger.error('❌ Risk Management System shutdown failed:', error);
    }
  }
}

module.exports = { RiskManagementSystem };
