const FreedomFinanceAI = require('../ai/freedomFinanceAI');
const axios = require('axios');
const logger = require('../utils/logger');

// Only initialize advanced AI client if not in test mode
let advancedAIClient = null;
if (process.env.NODE_ENV !== 'test') {
  advancedAIClient = require('../ai/geminiClient');
} else {
  advancedAIClient = {
    generate: async () => 'Simulated advanced AI response (test mode)'
  };
}

class AdvancedAIService {
  constructor() {
    this.ai = new FreedomFinanceAI();
    this.mlModels = new Map();
    this.predictionCache = new Map();
    this.optimizationHistory = new Map();
  }

  /**
   * Initialize the advanced AI service
   */
  async initialize() {
    try {
      await this.ai.initialize();
      await this.loadMLModels();
      logger.info('Advanced AI Service initialized successfully');
      return true;
    } catch (error) {
      logger.error('Failed to initialize Advanced AI Service:', error);
      return false;
    }
  }

  /**
   * Load machine learning models
   */
  async loadMLModels() {
    // Initialize different ML models for various predictions
    this.mlModels.set('fund_performance', this.createPerformanceModel());
    this.mlModels.set('market_trend', this.createTrendModel());
    this.mlModels.set('risk_assessment', this.createRiskModel());
    this.mlModels.set('portfolio_optimization', this.createOptimizationModel());
    
    logger.info('ML models loaded successfully');
  }

  /**
   * Create performance prediction model
   */
  createPerformanceModel() {
    return {
      name: 'fund_performance_predictor',
      version: '1.0',
      features: ['nav_history', 'market_conditions', 'fund_category', 'expense_ratio'],
      predict: async (data) => {
        // Advanced ML-based performance prediction
        const prediction = await this.predictFundPerformance(data);
        return prediction;
      }
    };
  }

  /**
   * Create market trend prediction model
   */
  createTrendModel() {
    return {
      name: 'market_trend_predictor',
      version: '1.0',
      features: ['technical_indicators', 'sentiment_data', 'macro_economic_data'],
      predict: async (data) => {
        const prediction = await this.predictMarketTrend(data);
        return prediction;
      }
    };
  }

  /**
   * Create risk assessment model
   */
  createRiskModel() {
    return {
      name: 'risk_assessment_model',
      version: '1.0',
      features: ['volatility', 'drawdown', 'correlation', 'sector_exposure'],
      assess: async (data) => {
        const riskScore = await this.assessPortfolioRisk(data);
        return riskScore;
      }
    };
  }

  /**
   * Create portfolio optimization model
   */
  createOptimizationModel() {
    return {
      name: 'portfolio_optimizer',
      version: '1.0',
      features: ['current_allocation', 'risk_tolerance', 'investment_goals', 'market_conditions'],
      optimize: async (data) => {
        const optimization = await this.optimizePortfolio(data);
        return optimization;
      }
    };
  }

  /**
   * Predict fund performance using ML
   */
  async predictFundPerformance(fundData) {
    try {
      const { navHistory, marketConditions, fundCategory, expenseRatio } = fundData;
      
      // Calculate technical indicators
      const indicators = this.calculateTechnicalIndicators(navHistory);
      
      // Analyze market sentiment
      const sentiment = await this.analyzeMarketSentiment(marketConditions);
      
      // Predict future performance
      const prediction = {
        shortTerm: this.predictShortTerm(indicators, sentiment),
        mediumTerm: this.predictMediumTerm(indicators, sentiment, fundCategory),
        longTerm: this.predictLongTerm(indicators, sentiment, expenseRatio),
        confidence: this.calculateConfidence(indicators, sentiment),
        factors: {
          technical: indicators,
          sentiment: sentiment,
          fundamental: { category: fundCategory, expenseRatio }
        }
      };

      return prediction;
    } catch (error) {
      logger.error('Error predicting fund performance:', error);
      return null;
    }
  }

  /**
   * Predict market trend
   */
  async predictMarketTrend(marketData) {
    try {
      const { technicalIndicators, sentimentData, macroEconomicData } = marketData;
      
      // Analyze multiple timeframes
      const trends = {
        immediate: this.analyzeImmediateTrend(technicalIndicators),
        shortTerm: this.analyzeShortTermTrend(technicalIndicators, sentimentData),
        mediumTerm: this.analyzeMediumTermTrend(technicalIndicators, macroEconomicData),
        longTerm: this.analyzeLongTermTrend(macroEconomicData)
      };

      // Combine predictions
      const overallTrend = this.combineTrendPredictions(trends);
      
      return {
        trends,
        overallTrend,
        confidence: this.calculateTrendConfidence(trends),
        keyFactors: this.identifyKeyFactors(trends)
      };
    } catch (error) {
      logger.error('Error predicting market trend:', error);
      return null;
    }
  }

  /**
   * Assess portfolio risk
   */
  async assessPortfolioRisk(portfolioData) {
    try {
      const { volatility, drawdown, correlation, sectorExposure } = portfolioData;
      
      // Calculate risk metrics
      const riskMetrics = {
        volatilityRisk: this.calculateVolatilityRisk(volatility),
        drawdownRisk: this.calculateDrawdownRisk(drawdown),
        correlationRisk: this.calculateCorrelationRisk(correlation),
        concentrationRisk: this.calculateConcentrationRisk(sectorExposure),
        liquidityRisk: this.calculateLiquidityRisk(portfolioData)
      };

      // Overall risk score
      const overallRisk = this.calculateOverallRisk(riskMetrics);
      
      return {
        riskMetrics,
        overallRisk,
        riskLevel: this.getRiskLevel(overallRisk),
        recommendations: this.generateRiskRecommendations(riskMetrics)
      };
    } catch (error) {
      logger.error('Error assessing portfolio risk:', error);
      return null;
    }
  }

  /**
   * Optimize portfolio allocation
   */
  async optimizePortfolio(portfolioData) {
    try {
      const { currentAllocation, riskTolerance, investmentGoals, marketConditions } = portfolioData;
      
      // Run optimization algorithms
      const optimization = {
        efficientFrontier: this.calculateEfficientFrontier(currentAllocation, riskTolerance),
        optimalAllocation: this.findOptimalAllocation(currentAllocation, riskTolerance, investmentGoals),
        rebalancingRecommendations: this.generateRebalancingRecommendations(currentAllocation, marketConditions),
        taxOptimization: this.optimizeForTaxes(currentAllocation, investmentGoals)
      };

      return optimization;
    } catch (error) {
      logger.error('Error optimizing portfolio:', error);
      return null;
    }
  }

  /**
   * Calculate technical indicators
   */
  calculateTechnicalIndicators(navHistory) {
    const prices = navHistory.map(item => parseFloat(item.nav));
    
    return {
      sma20: this.calculateSMA(prices, 20),
      sma50: this.calculateSMA(prices, 50),
      rsi: this.calculateRSI(prices, 14),
      macd: this.calculateMACD(prices),
      bollingerBands: this.calculateBollingerBands(prices, 20),
      volatility: this.calculateVolatility(prices)
    };
  }

  /**
   * Calculate Simple Moving Average
   */
  calculateSMA(prices, period) {
    if (prices.length < period) return null;
    const sum = prices.slice(-period).reduce((a, b) => a + b, 0);
    return sum / period;
  }

  /**
   * Calculate RSI
   */
  calculateRSI(prices, period = 14) {
    if (prices.length < period + 1) return null;
    
    let gains = 0;
    let losses = 0;
    
    for (let i = 1; i <= period; i++) {
      const change = prices[prices.length - i] - prices[prices.length - i - 1];
      if (change > 0) gains += change;
      else losses -= change;
    }
    
    const avgGain = gains / period;
    const avgLoss = losses / period;
    const rs = avgGain / avgLoss;
    
    return 100 - (100 / (1 + rs));
  }

  /**
   * Calculate MACD
   */
  calculateMACD(prices) {
    const ema12 = this.calculateEMA(prices, 12);
    const ema26 = this.calculateEMA(prices, 26);
    const macd = ema12 - ema26;
    const signal = this.calculateEMA([...Array(prices.length - 26).fill(0), macd], 9);
    
    return {
      macd,
      signal,
      histogram: macd - signal
    };
  }

  /**
   * Calculate Exponential Moving Average
   */
  calculateEMA(prices, period) {
    if (prices.length < period) return null;
    
    const multiplier = 2 / (period + 1);
    let ema = prices[0];
    
    for (let i = 1; i < prices.length; i++) {
      ema = (prices[i] * multiplier) + (ema * (1 - multiplier));
    }
    
    return ema;
  }

  /**
   * Calculate Bollinger Bands
   */
  calculateBollingerBands(prices, period = 20) {
    if (prices.length < period) return null;
    
    const sma = this.calculateSMA(prices, period);
    const variance = prices.slice(-period).reduce((sum, price) => {
      return sum + Math.pow(price - sma, 2);
    }, 0) / period;
    
    const standardDeviation = Math.sqrt(variance);
    
    return {
      upper: sma + (2 * standardDeviation),
      middle: sma,
      lower: sma - (2 * standardDeviation)
    };
  }

  /**
   * Calculate volatility
   */
  calculateVolatility(prices) {
    if (prices.length < 2) return 0;
    
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }
    
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => {
      return sum + Math.pow(ret - mean, 2);
    }, 0) / returns.length;
    
    return Math.sqrt(variance);
  }

  /**
   * Analyze market sentiment
   */
  async analyzeMarketSentiment(marketConditions) {
    try {
      // Analyze multiple sentiment sources
      const sentiment = {
        news: await this.analyzeNewsSentiment(),
        social: await this.analyzeSocialSentiment(),
        technical: this.analyzeTechnicalSentiment(marketConditions),
        institutional: await this.analyzeInstitutionalSentiment()
      };

      // Combine sentiment scores
      const overallSentiment = this.combineSentimentScores(sentiment);
      
      return {
        components: sentiment,
        overall: overallSentiment,
        confidence: this.calculateSentimentConfidence(sentiment)
      };
    } catch (error) {
      logger.error('Error analyzing market sentiment:', error);
      return { overall: 'NEUTRAL', confidence: 0.5 };
    }
  }

  /**
   * Analyze news sentiment
   */
  async analyzeNewsSentiment() {
    // Mock implementation - in production, integrate with news APIs
    return {
      score: Math.random() * 2 - 1, // -1 to 1
      sources: ['financial_news', 'economic_reports'],
      confidence: 0.7
    };
  }

  /**
   * Analyze social sentiment
   */
  async analyzeSocialSentiment() {
    // Mock implementation - in production, integrate with social media APIs
    return {
      score: Math.random() * 2 - 1,
      sources: ['twitter', 'reddit', 'forums'],
      confidence: 0.6
    };
  }

  /**
   * Analyze technical sentiment
   */
  analyzeTechnicalSentiment(marketConditions) {
    // Analyze technical indicators for sentiment
    const indicators = marketConditions.technicalIndicators || {};
    
    let bullishSignals = 0;
    let bearishSignals = 0;
    
    if (indicators.rsi < 30) bullishSignals++;
    if (indicators.rsi > 70) bearishSignals++;
    
    return {
      score: (bullishSignals - bearishSignals) / (bullishSignals + bearishSignals + 1),
      signals: { bullish: bullishSignals, bearish: bearishSignals },
      confidence: 0.8
    };
  }

  /**
   * Analyze institutional sentiment
   */
  async analyzeInstitutionalSentiment() {
    // Mock implementation - in production, analyze institutional flows
    return {
      score: Math.random() * 2 - 1,
      sources: ['fii_flows', 'dii_flows', 'mutual_fund_flows'],
      confidence: 0.9
    };
  }

  /**
   * Combine sentiment scores
   */
  combineSentimentScores(sentiment) {
    const weights = {
      news: 0.3,
      social: 0.2,
      technical: 0.3,
      institutional: 0.2
    };

    const weightedScore = Object.keys(sentiment).reduce((total, key) => {
      return total + (sentiment[key].score * weights[key]);
    }, 0);

    if (weightedScore > 0.3) return 'BULLISH';
    if (weightedScore < -0.3) return 'BEARISH';
    return 'NEUTRAL';
  }

  /**
   * Predict short-term performance
   */
  predictShortTerm(indicators, sentiment) {
    const score = this.calculatePredictionScore(indicators, sentiment, 'short');
    return {
      direction: score > 0 ? 'UP' : 'DOWN',
      magnitude: Math.abs(score),
      confidence: this.calculatePredictionConfidence(indicators, sentiment)
    };
  }

  /**
   * Predict medium-term performance
   */
  predictMediumTerm(indicators, sentiment, fundCategory) {
    const score = this.calculatePredictionScore(indicators, sentiment, 'medium');
    return {
      direction: score > 0 ? 'UP' : 'DOWN',
      magnitude: Math.abs(score),
      confidence: this.calculatePredictionConfidence(indicators, sentiment),
      categoryFactor: this.getCategoryFactor(fundCategory)
    };
  }

  /**
   * Predict long-term performance
   */
  predictLongTerm(indicators, sentiment, expenseRatio) {
    const score = this.calculatePredictionScore(indicators, sentiment, 'long');
    return {
      direction: score > 0 ? 'UP' : 'DOWN',
      magnitude: Math.abs(score),
      confidence: this.calculatePredictionConfidence(indicators, sentiment),
      expenseFactor: this.getExpenseFactor(expenseRatio)
    };
  }

  /**
   * Calculate prediction score
   */
  calculatePredictionScore(indicators, sentiment, timeframe) {
    let score = 0;
    
    // Technical indicators
    if (indicators.rsi < 30) score += 0.3;
    if (indicators.rsi > 70) score -= 0.3;
    
    // Sentiment
    if (sentiment.overall === 'BULLISH') score += 0.2;
    if (sentiment.overall === 'BEARISH') score -= 0.2;
    
    // Timeframe adjustment
    if (timeframe === 'short') score *= 1.2;
    if (timeframe === 'long') score *= 0.8;
    
    return Math.max(-1, Math.min(1, score));
  }

  /**
   * Calculate prediction confidence
   */
  calculatePredictionConfidence(indicators, sentiment) {
    const technicalConfidence = indicators.volatility ? Math.max(0.3, 1 - indicators.volatility) : 0.5;
    const sentimentConfidence = sentiment.confidence || 0.5;
    
    return (technicalConfidence + sentimentConfidence) / 2;
  }

  /**
   * Get category factor
   */
  getCategoryFactor(fundCategory) {
    const factors = {
      'Large Cap': 1.0,
      'Mid Cap': 1.2,
      'Small Cap': 1.5,
      'Multi Cap': 1.1,
      'Sectoral': 1.3
    };
    
    return factors[fundCategory] || 1.0;
  }

  /**
   * Get expense factor
   */
  getExpenseFactor(expenseRatio) {
    if (expenseRatio < 0.5) return 1.1;
    if (expenseRatio < 1.0) return 1.0;
    if (expenseRatio < 1.5) return 0.9;
    return 0.8;
  }

  /**
   * Calculate confidence
   */
  calculateConfidence(indicators, sentiment) {
    const technicalConfidence = this.calculateTechnicalConfidence(indicators);
    const sentimentConfidence = sentiment.confidence || 0.5;
    
    return (technicalConfidence + sentimentConfidence) / 2;
  }

  /**
   * Calculate technical confidence
   */
  calculateTechnicalConfidence(indicators) {
    let confidence = 0.5;
    
    if (indicators.rsi && indicators.rsi > 0 && indicators.rsi < 100) confidence += 0.2;
    if (indicators.volatility && indicators.volatility < 0.5) confidence += 0.2;
    if (indicators.sma20 && indicators.sma50) confidence += 0.1;
    
    return Math.min(1, confidence);
  }

  /**
   * Generate comprehensive AI insights
   */
  async generateInsights(portfolioData, marketData) {
    try {
      const insights = {
        performance: await this.ai.analyzePortfolio(portfolioData, marketData),
        predictions: await this.generatePredictions(portfolioData, marketData),
        optimization: await this.generateOptimizationRecommendations(portfolioData),
        risk: await this.assessPortfolioRisk(portfolioData),
        market: await this.ai.generateMarketInsights(marketData)
      };

      return insights;
    } catch (error) {
      logger.error('Error generating insights:', error);
      return null;
    }
  }

  /**
   * Generate predictions
   */
  async generatePredictions(portfolioData, marketData) {
    const predictions = {};
    
    for (const fund of portfolioData.funds || []) {
      predictions[fund.schemeCode] = await this.predictFundPerformance({
        navHistory: fund.navHistory,
        marketConditions: marketData,
        fundCategory: fund.category,
        expenseRatio: fund.expenseRatio
      });
    }
    
    return predictions;
  }

  /**
   * Generate optimization recommendations
   */
  async generateOptimizationRecommendations(portfolioData) {
    return await this.optimizePortfolio({
      currentAllocation: portfolioData.allocation,
      riskTolerance: portfolioData.riskTolerance,
      investmentGoals: portfolioData.goals,
      marketConditions: portfolioData.marketConditions
    });
  }
}

module.exports = {
  AdvancedAIService,
  advancedAIClient: advancedAIClient || { generate: async () => 'Simulated advanced AI response (test mode)' }
}; 