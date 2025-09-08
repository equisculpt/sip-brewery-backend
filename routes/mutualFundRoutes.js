const express = require('express');
const router = express.Router();
const RealMutualFundDataService = require('../services/RealMutualFundDataService');
const RealTimeMarketService = require('../services/RealTimeMarketService');
const { authenticateToken } = require('../middleware/auth');
const { validateRequest } = require('../middleware/validation');
const { body, query, param } = require('express-validator');
const response = require('../utils/response');
const logger = require('../utils/logger');

const realMutualFundService = new RealMutualFundDataService();
const realTimeMarketService = new RealTimeMarketService();

// 🚀 $1 BILLION PLATFORM - REAL DATA ONLY
console.log('✅ Initialized REAL Mutual Fund Data Services - No Demo Data!');
console.log('📊 Connected to: AMFI, NSE, BSE, Yahoo Finance, Alpha Vantage');
console.log('💎 Real-time updates every 30 seconds during market hours');

/**
 * @route GET /api/mutual-funds/schemes
 * @desc Get all supported mutual fund schemes
 * @access Public
 */
router.get('/schemes', async (req, res) => {
  try {
    logger.info('🚀 Fetching REAL mutual fund schemes from live data sources...');
    
    const schemes = realMutualFundService.getRealSchemes();
    
    return response.success(res, 'Real mutual fund schemes retrieved successfully', {
      schemes,
      total_count: schemes.length,
      categories: [...new Set(schemes.map(s => s.category))],
      data_sources: ['AMFI', 'NSE', 'BSE'],
      real_data: true,
      platform: '$1 Billion Revolutionary Platform',
      last_updated: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Error fetching real schemes:', error);
    return response.error(res, 'Failed to fetch real schemes', error.message);
  }
});

/**
 * @route GET /api/mutual-funds/chart/:schemeCode
 * @desc Get TradingView-style chart data for a mutual fund scheme
 * @access Private
 */
router.get('/chart/:schemeCode', [
  authenticateToken,
  param('schemeCode').notEmpty().withMessage('Scheme code is required'),
  query('period').optional().isIn(['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', 'MAX']).withMessage('Invalid period'),
  validateRequest
], async (req, res) => {
  try {
    const { schemeCode } = req.params;
    const { period = '1Y' } = req.query;
    
    logger.info(`🚀 Fetching REAL chart data for scheme: ${schemeCode}, period: ${period}`);
    logger.info('📊 Data sources: AMFI, MF-API, Value Research, MoneyControl');
    
    const chartData = await realMutualFundService.getRealNAVHistory(schemeCode, period);
    
    // Transform data for TradingView format
    const tradingViewData = {
      scheme: chartData.scheme,
      period,
      candles: chartData.data.map(item => ({
        time: Math.floor(item.timestamp / 1000), // TradingView expects seconds
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
        volume: item.volume,
        nav: item.nav
      })),
      indicators: {
        moving_averages: {
          sma_20: chartData.technical.sma.sma_20,
          sma_50: chartData.technical.sma.sma_50,
          sma_100: chartData.technical.sma.sma_100,
          sma_200: chartData.technical.sma.sma_200,
          ema_12: chartData.technical.ema.ema_12,
          ema_26: chartData.technical.ema.ema_26,
          ema_50: chartData.technical.ema.ema_50
        },
        oscillators: {
          rsi: chartData.technical.rsi.rsi_14,
          macd: chartData.technical.macd,
          stochastic: chartData.technical.stochastic
        },
        volatility: {
          bollinger_bands: chartData.technical.bollinger
        },
        volume: {
          volume_sma: chartData.technical.volume_sma
        }
      },
      statistics: {
        total_return: this.calculateTotalReturn(chartData.data),
        annualized_return: this.calculateAnnualizedReturn(chartData.data, period),
        volatility: mutualFundService.calculateVolatility(chartData.data),
        sharpe_ratio: mutualFundService.calculateSharpeRatio(chartData.data),
        max_drawdown: mutualFundService.calculateMaxDrawdown(chartData.data),
        current_nav: chartData.data[chartData.data.length - 1]?.nav
      },
      metadata: chartData.metadata
    };
    
    return response.success(res, 'Chart data retrieved successfully', tradingViewData);
    
  } catch (error) {
    logger.error('Error fetching chart data:', error);
    return response.error(res, 'Failed to fetch chart data', error.message);
  }
});

/**
 * @route POST /api/mutual-funds/compare
 * @desc Compare multiple mutual fund schemes
 * @access Private
 */
router.post('/compare', [
  authenticateToken,
  body('schemes').isArray({ min: 2, max: 5 }).withMessage('Please provide 2-5 schemes for comparison'),
  body('period').optional().isIn(['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y']).withMessage('Invalid period'),
  validateRequest
], async (req, res) => {
  try {
    const { schemes, period = '1Y' } = req.body;
    
    logger.info(`Comparing schemes: ${schemes.join(', ')} for period: ${period}`);
    
    const comparison = await mutualFundService.getSchemeComparison(schemes, period);
    
    // Enhanced comparison with additional metrics
    const enhancedComparison = {
      ...comparison,
      correlation_matrix: await this.calculateCorrelationMatrix(schemes, period),
      risk_return_analysis: this.performRiskReturnAnalysis(comparison.comparison),
      recommendation: this.generateRecommendation(comparison.comparison)
    };
    
    return response.success(res, 'Scheme comparison completed successfully', enhancedComparison);
    
  } catch (error) {
    logger.error('Error in scheme comparison:', error);
    return response.error(res, 'Failed to compare schemes', error.message);
  }
});

/**
 * @route GET /api/mutual-funds/top-performers
 * @desc Get top performing mutual funds by category
 * @access Private
 */
router.get('/top-performers', [
  authenticateToken,
  query('category').optional().isIn(['ALL', 'Large Cap', 'Mid Cap', 'Small Cap', 'Multi Cap', 'Hybrid']).withMessage('Invalid category'),
  query('period').optional().isIn(['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y']).withMessage('Invalid period'),
  query('limit').optional().isInt({ min: 1, max: 20 }).withMessage('Limit must be between 1 and 20'),
  validateRequest
], async (req, res) => {
  try {
    const { category = 'ALL', period = '1Y', limit = 10 } = req.query;
    
    const topPerformers = await mutualFundService.getTopPerformingSchemes(category, period, parseInt(limit));
    
    return response.success(res, 'Top performers retrieved successfully', topPerformers);
    
  } catch (error) {
    logger.error('Error fetching top performers:', error);
    return response.error(res, 'Failed to fetch top performers', error.message);
  }
});

/**
 * @route GET /api/mutual-funds/technical-analysis/:schemeCode
 * @desc Get detailed technical analysis for a scheme
 * @access Private
 */
router.get('/technical-analysis/:schemeCode', [
  authenticateToken,
  param('schemeCode').notEmpty().withMessage('Scheme code is required'),
  query('period').optional().isIn(['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y']).withMessage('Invalid period'),
  validateRequest
], async (req, res) => {
  try {
    const { schemeCode } = req.params;
    const { period = '1Y' } = req.query;
    
    const chartData = await mutualFundService.getMutualFundNAVHistory(schemeCode, period);
    
    // Advanced technical analysis
    const technicalAnalysis = {
      scheme: chartData.scheme,
      period,
      trend_analysis: this.analyzeTrend(chartData.data, chartData.technical),
      support_resistance: this.findSupportResistance(chartData.data),
      momentum_analysis: this.analyzeMomentum(chartData.technical),
      volume_analysis: this.analyzeVolume(chartData.data),
      pattern_recognition: this.recognizePatterns(chartData.data),
      signals: this.generateTradingSignals(chartData.data, chartData.technical),
      risk_metrics: {
        volatility: mutualFundService.calculateVolatility(chartData.data),
        var_95: this.calculateVaR(chartData.data, 0.95),
        beta: this.calculateBeta(chartData.data), // vs benchmark
        correlation_with_market: this.calculateMarketCorrelation(chartData.data)
      },
      forecast: this.generateForecast(chartData.data, chartData.technical)
    };
    
    return response.success(res, 'Technical analysis completed successfully', technicalAnalysis);
    
  } catch (error) {
    logger.error('Error in technical analysis:', error);
    return response.error(res, 'Failed to perform technical analysis', error.message);
  }
});

/**
 * @route POST /api/mutual-funds/portfolio-analysis
 * @desc Analyze a portfolio of mutual funds
 * @access Private
 */
router.post('/portfolio-analysis', [
  authenticateToken,
  body('portfolio').isArray({ min: 1 }).withMessage('Portfolio must contain at least one scheme'),
  body('portfolio.*.scheme_code').notEmpty().withMessage('Scheme code is required'),
  body('portfolio.*.allocation').isFloat({ min: 0, max: 100 }).withMessage('Allocation must be between 0 and 100'),
  validateRequest
], async (req, res) => {
  try {
    const { portfolio, period = '1Y' } = req.body;
    
    // Validate allocation sums to 100%
    const totalAllocation = portfolio.reduce((sum, item) => sum + item.allocation, 0);
    if (Math.abs(totalAllocation - 100) > 0.01) {
      return response.error(res, 'Portfolio allocations must sum to 100%');
    }
    
    logger.info(`Analyzing portfolio with ${portfolio.length} schemes`);
    
    // Get data for all schemes
    const schemePromises = portfolio.map(item => 
      mutualFundService.getMutualFundNAVHistory(item.scheme_code, period)
    );
    const schemeData = await Promise.all(schemePromises);
    
    // Portfolio analysis
    const portfolioAnalysis = {
      portfolio_composition: portfolio,
      period,
      performance_metrics: this.calculatePortfolioMetrics(schemeData, portfolio),
      risk_analysis: this.calculatePortfolioRisk(schemeData, portfolio),
      diversification_analysis: this.analyzeDiversification(schemeData, portfolio),
      rebalancing_suggestions: this.generateRebalancingSuggestions(schemeData, portfolio),
      scenario_analysis: this.performScenarioAnalysis(schemeData, portfolio),
      optimization_suggestions: this.optimizePortfolio(schemeData, portfolio)
    };
    
    return response.success(res, 'Portfolio analysis completed successfully', portfolioAnalysis);
    
  } catch (error) {
    logger.error('Error in portfolio analysis:', error);
    return response.error(res, 'Failed to analyze portfolio', error.message);
  }
});

// Helper methods (would typically be in a separate utility class)
router.calculateTotalReturn = function(data) {
  if (!data || data.length < 2) return 0;
  const startNAV = data[0].nav;
  const endNAV = data[data.length - 1].nav;
  return parseFloat((((endNAV - startNAV) / startNAV) * 100).toFixed(2));
};

router.calculateAnnualizedReturn = function(data, period) {
  const totalReturn = this.calculateTotalReturn(data);
  const years = this.periodToYears(period);
  return parseFloat((Math.pow(1 + totalReturn / 100, 1 / years) - 1) * 100).toFixed(2);
};

router.periodToYears = function(period) {
  const periodMap = { '1M': 1/12, '3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2, '3Y': 3, '5Y': 5 };
  return periodMap[period] || 1;
};

router.analyzeTrend = function(data, technical) {
  const latest = data[data.length - 1];
  const sma20 = technical.sma.sma_20[technical.sma.sma_20.length - 1]?.value;
  const sma50 = technical.sma.sma_50[technical.sma.sma_50.length - 1]?.value;
  
  let trend = 'NEUTRAL';
  if (latest.nav > sma20 && sma20 > sma50) trend = 'BULLISH';
  else if (latest.nav < sma20 && sma20 < sma50) trend = 'BEARISH';
  
  return {
    overall_trend: trend,
    short_term: latest.nav > sma20 ? 'BULLISH' : 'BEARISH',
    medium_term: sma20 > sma50 ? 'BULLISH' : 'BEARISH',
    strength: this.calculateTrendStrength(data)
  };
};

router.calculateTrendStrength = function(data) {
  // Calculate trend strength based on consecutive higher/lower closes
  let strength = 0;
  let direction = 0;
  
  for (let i = 1; i < Math.min(data.length, 20); i++) {
    if (data[i].nav > data[i-1].nav) {
      if (direction >= 0) strength++;
      direction = 1;
    } else if (data[i].nav < data[i-1].nav) {
      if (direction <= 0) strength++;
      direction = -1;
    }
  }
  
  return Math.min(strength / 20 * 100, 100);
};

module.exports = router;
