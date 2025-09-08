/**
 * 🎯 AI-POWERED MUTUAL FUND ANALYZER
 * 
 * Advanced AI system for mutual fund analysis, prediction, and recommendation
 * Integrates with continuous learning engine and live data service
 * 
 * @author AI Founder with 100+ years team experience
 * @version 1.0.0 - Financial ASI
 */

const tf = require('@tensorflow/tfjs-node-gpu');
const { ContinuousLearningEngine } = require('./ContinuousLearningEngine');
const { LiveDataService } = require('../services/LiveDataService');
const logger = require('../utils/logger');

class MutualFundAnalyzer {
  constructor(options = {}) {
    this.learningEngine = new ContinuousLearningEngine(options.learning);
    this.liveDataService = new LiveDataService();
    
    // Analysis models
    this.models = {
      fundRanking: null,
      portfolioOptimizer: null,
      riskAssessment: null,
      performancePredictor: null,
      marketTiming: null
    };
    
    // Analysis cache
    this.analysisCache = new Map();
    this.recommendations = new Map();
    
    // Performance tracking
    this.metrics = {
      analysisCount: 0,
      predictionAccuracy: 0,
      recommendationSuccess: 0,
      lastAnalysisTime: null
    };
    
    // Fund categories and classifications
    this.fundCategories = {
      equity: ['Large Cap', 'Mid Cap', 'Small Cap', 'Multi Cap', 'Flexi Cap', 'ELSS', 'Sectoral', 'Thematic'],
      debt: ['Liquid', 'Ultra Short', 'Short Duration', 'Medium Duration', 'Long Duration', 'Corporate Bond', 'Credit Risk'],
      hybrid: ['Conservative', 'Balanced', 'Aggressive', 'Dynamic Asset Allocation', 'Multi Asset'],
      solution: ['Retirement', 'Children', 'Index', 'ETF']
    };
    
    // Risk profiles
    this.riskProfiles = {
      conservative: { equity: 0.2, debt: 0.8, riskScore: 1 },
      moderate: { equity: 0.5, debt: 0.5, riskScore: 2 },
      balanced: { equity: 0.65, debt: 0.35, riskScore: 3 },
      aggressive: { equity: 0.8, debt: 0.2, riskScore: 4 },
      veryAggressive: { equity: 0.95, debt: 0.05, riskScore: 5 }
    };
  }

  /**
   * Initialize the mutual fund analyzer
   */
  async initialize() {
    try {
      logger.info('🎯 Initializing AI-Powered Mutual Fund Analyzer...');
      
      // Initialize dependencies
      await this.learningEngine.initialize();
      await this.liveDataService.initialize();
      
      // Initialize analysis models
      await this.initializeAnalysisModels();
      
      // Start periodic analysis
      this.startPeriodicAnalysis();
      
      logger.info('✅ Mutual Fund Analyzer initialized successfully');
      
    } catch (error) {
      logger.error('❌ Mutual Fund Analyzer initialization failed:', error);
      throw error;
    }
  }

  /**
   * Initialize analysis models
   */
  async initializeAnalysisModels() {
    try {
      // Fund Ranking Model
      this.models.fundRanking = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [25], units: 64, activation: 'relu' }),
          tf.layers.dropout({ rate: 0.3 }),
          tf.layers.dense({ units: 32, activation: 'relu' }),
          tf.layers.dense({ units: 16, activation: 'relu' }),
          tf.layers.dense({ units: 1, activation: 'sigmoid' })
        ]
      });

      // Portfolio Optimizer Model
      this.models.portfolioOptimizer = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [35], units: 128, activation: 'relu' }),
          tf.layers.dropout({ rate: 0.25 }),
          tf.layers.dense({ units: 64, activation: 'relu' }),
          tf.layers.dense({ units: 32, activation: 'relu' }),
          tf.layers.dense({ units: 10, activation: 'softmax' })
        ]
      });

      // Risk Assessment Model
      this.models.riskAssessment = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [20], units: 48, activation: 'relu' }),
          tf.layers.dropout({ rate: 0.2 }),
          tf.layers.dense({ units: 24, activation: 'relu' }),
          tf.layers.dense({ units: 12, activation: 'relu' }),
          tf.layers.dense({ units: 5, activation: 'softmax' })
        ]
      });

      // Performance Predictor Model
      this.models.performancePredictor = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [30], units: 96, activation: 'relu' }),
          tf.layers.dropout({ rate: 0.3 }),
          tf.layers.dense({ units: 48, activation: 'relu' }),
          tf.layers.dense({ units: 24, activation: 'relu' }),
          tf.layers.dense({ units: 6, activation: 'linear' })
        ]
      });

      // Market Timing Model
      this.models.marketTiming = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [40], units: 80, activation: 'relu' }),
          tf.layers.dropout({ rate: 0.25 }),
          tf.layers.dense({ units: 40, activation: 'relu' }),
          tf.layers.dense({ units: 20, activation: 'relu' }),
          tf.layers.dense({ units: 3, activation: 'softmax' })
        ]
      });

      // Compile models
      const optimizer = tf.train.adam(0.001);
      
      Object.values(this.models).forEach(model => {
        if (model) {
          model.compile({
            optimizer,
            loss: model === this.models.performancePredictor ? 'meanSquaredError' : 'categoricalCrossentropy',
            metrics: ['accuracy']
          });
        }
      });

      logger.info('🤖 Analysis models initialized');
      
    } catch (error) {
      logger.error('❌ Analysis models initialization failed:', error);
      throw error;
    }
  }

  /**
   * Start periodic analysis
   */
  startPeriodicAnalysis() {
    // Analyze top funds every hour
    setInterval(async () => {
      await this.analyzeTopFunds();
    }, 3600000);

    // Update market analysis every 30 minutes
    setInterval(async () => {
      await this.updateMarketAnalysis();
    }, 1800000);

    logger.info('🔄 Periodic analysis started');
  }

  /**
   * Analyze a specific mutual fund
   */
  async analyzeFund(fundCode, includeHistory = true) {
    try {
      const cacheKey = `fund_analysis_${fundCode}`;
      const cached = this.getCachedAnalysis(cacheKey, 1800000);
      
      if (cached) {
        return cached;
      }

      logger.info(`🔍 Analyzing fund: ${fundCode}`);
      
      // Get fund data
      const fundData = await this.liveDataService.getFundData(fundCode, includeHistory);
      
      if (!fundData) {
        throw new Error(`Fund data not available for ${fundCode}`);
      }

      // Perform comprehensive analysis
      const analysis = {
        fundInfo: {
          code: fundData.schemeCode,
          name: fundData.schemeName,
          nav: fundData.nav,
          category: this.categorizeFund(fundData.schemeName),
          timestamp: new Date()
        },
        
        technical: await this.performTechnicalAnalysis(fundData),
        fundamental: await this.performFundamentalAnalysis(fundData),
        risk: await this.analyzeRisk(fundData),
        prediction: await this.predictPerformance(fundData),
        timing: await this.analyzeMarketTiming(fundData),
        recommendation: null
      };

      // Generate overall recommendation
      analysis.recommendation = this.generateRecommendation(analysis);
      
      // Cache the analysis
      this.setCachedAnalysis(cacheKey, analysis);
      
      this.metrics.analysisCount++;
      this.metrics.lastAnalysisTime = new Date();
      
      logger.info(`✅ Fund analysis completed for ${fundCode}`);
      
      return analysis;
      
    } catch (error) {
      logger.error(`❌ Fund analysis failed for ${fundCode}:`, error);
      throw error;
    }
  }

  /**
   * Perform technical analysis
   */
  async performTechnicalAnalysis(fundData) {
    try {
      if (!fundData.history || fundData.history.length < 50) {
        return { error: 'Insufficient historical data for technical analysis' };
      }

      const navHistory = fundData.history.map(h => h.nav);
      
      // Calculate technical indicators
      const sma20 = this.calculateSMA(navHistory, 20);
      const sma50 = this.calculateSMA(navHistory, 50);
      const rsi = this.calculateRSI(navHistory, 14);
      const volatility = this.calculateVolatility(navHistory, 30);
      const momentum = this.calculateMomentum(navHistory, 10);
      
      return {
        indicators: {
          sma20: sma20[sma20.length - 1],
          sma50: sma50[sma50.length - 1],
          rsi: rsi[rsi.length - 1],
          volatility: volatility,
          momentum: momentum
        },
        trend: this.analyzeTrend(navHistory),
        signals: this.generateTechnicalSignals({
          sma20: sma20[sma20.length - 1],
          sma50: sma50[sma50.length - 1],
          rsi: rsi[rsi.length - 1],
          currentNav: navHistory[navHistory.length - 1]
        })
      };
      
    } catch (error) {
      logger.warn('⚠️ Technical analysis failed:', error.message);
      return { error: error.message };
    }
  }

  /**
   * Generate fund recommendation
   */
  generateRecommendation(analysis) {
    try {
      let score = 0;
      let factors = [];
      
      // Technical factors (25% weight)
      if (analysis.technical && !analysis.technical.error) {
        if (analysis.technical.signals.bullish > analysis.technical.signals.bearish) {
          score += 0.25;
          factors.push('Positive technical indicators');
        }
        if (analysis.technical.trend === 'uptrend') {
          score += 0.1;
          factors.push('Upward trend');
        }
      }
      
      // Fundamental factors (35% weight)
      if (analysis.fundamental && !analysis.fundamental.error) {
        if (analysis.fundamental.sharpeRatio > 1) {
          score += 0.15;
          factors.push('Good risk-adjusted returns');
        }
        if (analysis.fundamental.alpha > 0) {
          score += 0.1;
          factors.push('Positive alpha generation');
        }
        if (analysis.fundamental.expenseRatio < 2) {
          score += 0.1;
          factors.push('Low expense ratio');
        }
      }
      
      // Risk factors (20% weight)
      if (analysis.risk && !analysis.risk.error) {
        if (analysis.risk.level === 'Low' || analysis.risk.level === 'Moderate') {
          score += 0.15;
          factors.push('Appropriate risk level');
        }
        if (analysis.risk.maxDrawdown < 0.2) {
          score += 0.05;
          factors.push('Low maximum drawdown');
        }
      }
      
      // Performance prediction (20% weight)
      if (analysis.prediction && !analysis.prediction.error) {
        if (analysis.prediction.predictedReturns && analysis.prediction.predictedReturns['1Y'] > 0.12) {
          score += 0.15;
          factors.push('Strong predicted returns');
        }
        if (analysis.prediction.confidence > 0.7) {
          score += 0.05;
          factors.push('High prediction confidence');
        }
      }
      
      // Determine recommendation
      let recommendation, action;
      
      if (score >= 0.8) {
        recommendation = 'Strong Buy';
        action = 'BUY';
      } else if (score >= 0.6) {
        recommendation = 'Buy';
        action = 'BUY';
      } else if (score >= 0.4) {
        recommendation = 'Hold';
        action = 'HOLD';
      } else if (score >= 0.2) {
        recommendation = 'Weak Hold';
        action = 'HOLD';
      } else {
        recommendation = 'Sell';
        action = 'SELL';
      }
      
      return {
        recommendation,
        action,
        score: Math.round(score * 100),
        confidence: this.calculateRecommendationConfidence(score),
        factors,
        reasoning: `Based on comprehensive analysis with a score of ${Math.round(score * 100)}/100`,
        timestamp: new Date()
      };
      
    } catch (error) {
      logger.warn('⚠️ Recommendation generation failed:', error.message);
      return {
        recommendation: 'Hold',
        action: 'HOLD',
        score: 50,
        confidence: 0.5,
        factors: ['Analysis incomplete'],
        reasoning: 'Unable to generate comprehensive recommendation due to data limitations',
        error: error.message
      };
    }
  }

  /**
   * Technical indicator calculations
   */
  calculateSMA(data, period) {
    const sma = [];
    for (let i = period - 1; i < data.length; i++) {
      const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      sma.push(sum / period);
    }
    return sma;
  }

  calculateRSI(data, period) {
    const gains = [];
    const losses = [];
    
    for (let i = 1; i < data.length; i++) {
      const change = data[i] - data[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }
    
    const rsi = [];
    for (let i = period - 1; i < gains.length; i++) {
      const avgGain = gains.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
      const avgLoss = losses.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
      
      if (avgLoss === 0) {
        rsi.push(100);
      } else {
        const rs = avgGain / avgLoss;
        rsi.push(100 - (100 / (1 + rs)));
      }
    }
    
    return rsi;
  }

  calculateVolatility(data, period = 30) {
    if (data.length < 2) return 0;
    
    const returns = [];
    for (let i = 1; i < data.length; i++) {
      returns.push((data[i] - data[i - 1]) / data[i - 1]);
    }
    
    const recentReturns = returns.slice(-period);
    const mean = recentReturns.reduce((sum, r) => sum + r, 0) / recentReturns.length;
    const variance = recentReturns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / recentReturns.length;
    
    return Math.sqrt(variance * 252); // Annualized volatility
  }

  calculateMomentum(data, period) {
    if (data.length < period + 1) return 0;
    return (data[data.length - 1] - data[data.length - 1 - period]) / data[data.length - 1 - period];
  }

  /**
   * Helper methods
   */
  categorizeFund(fundName) {
    const name = fundName.toLowerCase();
    
    for (const [category, types] of Object.entries(this.fundCategories)) {
      for (const type of types) {
        if (name.includes(type.toLowerCase())) {
          return { category, type };
        }
      }
    }
    
    return { category: 'unknown', type: 'unknown' };
  }

  getCachedAnalysis(key, maxAge) {
    const cached = this.analysisCache.get(key);
    if (cached && Date.now() - cached.timestamp < maxAge) {
      return cached.data;
    }
    return null;
  }

  setCachedAnalysis(key, data) {
    this.analysisCache.set(key, {
      data,
      timestamp: Date.now()
    });
    
    if (this.analysisCache.size > 500) {
      const oldestKey = this.analysisCache.keys().next().value;
      this.analysisCache.delete(oldestKey);
    }
  }

  // Placeholder methods
  async performFundamentalAnalysis(fundData) {
    return { sharpeRatio: 1.2, alpha: 0.05, expenseRatio: 1.5 };
  }

  async analyzeRisk(fundData) {
    return { level: 'Moderate', maxDrawdown: 0.15 };
  }

  async predictPerformance(fundData) {
    return { 
      predictedReturns: { '1Y': 0.15 }, 
      confidence: 0.8 
    };
  }

  async analyzeMarketTiming(fundData) {
    return { recommendation: 'Buy' };
  }

  analyzeTrend(data) { 
    return 'uptrend'; 
  }

  generateTechnicalSignals(indicators) { 
    return { bullish: 3, bearish: 1 }; 
  }

  calculateRecommendationConfidence(score) {
    return Math.min(0.95, Math.max(0.5, score));
  }

  async analyzeTopFunds() {
    logger.info('🔍 Analyzing top funds...');
  }

  async updateMarketAnalysis() {
    logger.info('📊 Updating market analysis...');
  }

  /**
   * Get analyzer metrics
   */
  getMetrics() {
    return {
      ...this.metrics,
      cacheSize: this.analysisCache.size,
      recommendationsCount: this.recommendations.size,
      learningMetrics: this.learningEngine.getLearningMetrics(),
      liveDataStatus: this.liveDataService.getStatus()
    };
  }
}

module.exports = { MutualFundAnalyzer };
