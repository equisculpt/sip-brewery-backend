const moment = require('moment');
const logger = require('../utils/logger');

class NAVHistoryService {
  constructor() {
    this.periodMap = {
      '1w': 7,
      '1m': 30,
      '3m': 90,
      '6m': 180,
      '1y': 365,
      '3y': 1095,
      '5y': 1825
    };
  }

  /**
   * Get NAV history with calculations
   */
  async getNAVHistory({ fundCode, period = '1y', includeCalculations = true }) {
    try {
      logger.info(`Getting NAV history for fund: ${fundCode}, period: ${period}`);

      const navData = await this.fetchNAVData(fundCode, period);
      
      const result = {
        fundCode,
        period,
        data: navData,
        summary: {},
        calculations: {}
      };

      if (includeCalculations && navData.length > 0) {
        result.calculations = this.calculateNAVMetrics(navData);
        result.summary = this.generateNAVSummary(navData, result.calculations);
      }

      return result;
    } catch (error) {
      logger.error('Error getting NAV history:', error);
      throw error;
    }
  }

  /**
   * Calculate NAV-based performance metrics
   */
  async calculateNAVPerformance({ fundCode, period = '1y', benchmark = 'NIFTY50' }) {
    try {
      logger.info(`Calculating NAV performance for fund: ${fundCode}, period: ${period}`);

      const navData = await this.getNAVHistory({ fundCode, period });
      const benchmarkData = await this.getBenchmarkData(benchmark, period);

      const performance = {
        fund: navData.calculations,
        benchmark: benchmarkData.calculations,
        comparison: {},
        analysis: {}
      };

      // Calculate comparison metrics
      performance.comparison = this.calculateComparisonMetrics(
        navData.calculations,
        benchmarkData.calculations
      );

      // Generate performance analysis
      performance.analysis = this.generatePerformanceAnalysis(performance);

      return performance;
    } catch (error) {
      logger.error('Error calculating NAV performance:', error);
      throw error;
    }
  }

  /**
   * Get NAV data for multiple funds
   */
  async getMultipleFundsNAV({ fundCodes, period = '1y' }) {
    try {
      logger.info(`Getting NAV data for ${fundCodes.length} funds, period: ${period}`);

      const results = {};
      
      for (const fundCode of fundCodes) {
        try {
          results[fundCode] = await this.getNAVHistory({ fundCode, period });
        } catch (error) {
          logger.error(`Error getting NAV for fund ${fundCode}:`, error);
          results[fundCode] = { error: error.message };
        }
      }

      return results;
    } catch (error) {
      logger.error('Error getting multiple funds NAV:', error);
      throw error;
    }
  }

  /**
   * Calculate rolling returns
   */
  async calculateRollingReturns({ fundCode, period = '1y', rollingPeriod = '1m' }) {
    try {
      logger.info(`Calculating rolling returns for fund: ${fundCode}`);

      const navData = await this.getNAVHistory({ fundCode, period });
      const rollingReturns = this.computeRollingReturns(navData.data, rollingPeriod);

      return {
        fundCode,
        period,
        rollingPeriod,
        returns: rollingReturns,
        summary: this.calculateRollingReturnsSummary(rollingReturns)
      };
    } catch (error) {
      logger.error('Error calculating rolling returns:', error);
      throw error;
    }
  }

  /**
   * Fetch NAV data from source
   */
  async fetchNAVData(fundCode, period) {
    try {
      // This would typically fetch from database or external API
      // For now, generating mock data
      const days = this.periodMap[period] || 365;
      const data = [];
      const startDate = moment().subtract(days, 'days');
      
      let baseNAV = 45.67; // Starting NAV
      
      for (let i = 0; i < days; i++) {
        const date = moment(startDate).add(i, 'days');
        
        // Skip weekends
        if (date.day() === 0 || date.day() === 6) {
          continue;
        }

        // Simulate NAV movement
        const dailyReturn = (Math.random() - 0.5) * 0.02; // ±1% daily return
        baseNAV = baseNAV * (1 + dailyReturn);
        
        data.push({
          date: date.toDate(),
          nav: parseFloat(baseNAV.toFixed(4)),
          change: parseFloat((dailyReturn * 100).toFixed(2)),
          volume: Math.floor(Math.random() * 1000000) + 100000
        });
      }

      return data;
    } catch (error) {
      logger.error('Error fetching NAV data:', error);
      throw error;
    }
  }

  /**
   * Calculate NAV metrics
   */
  calculateNAVMetrics(navData) {
    if (navData.length < 2) {
      return {};
    }

    const calculations = {
      returns: {},
      volatility: {},
      riskMetrics: {},
      technicalIndicators: {}
    };

    // Calculate returns
    calculations.returns = this.calculateReturns(navData);

    // Calculate volatility
    calculations.volatility = this.calculateVolatility(navData);

    // Calculate risk metrics
    calculations.riskMetrics = this.calculateRiskMetrics(navData);

    // Calculate technical indicators
    calculations.technicalIndicators = this.calculateTechnicalIndicators(navData);

    return calculations;
  }

  /**
   * Calculate returns
   */
  calculateReturns(navData) {
    const returns = {};
    const sortedData = navData.sort((a, b) => new Date(a.date) - new Date(b.date));
    
    if (sortedData.length < 2) {
      return returns;
    }

    const firstNAV = sortedData[0].nav;
    const lastNAV = sortedData[sortedData.length - 1].nav;

    // Absolute return
    returns.absoluteReturn = ((lastNAV - firstNAV) / firstNAV) * 100;

    // Annualized return
    const days = moment(sortedData[sortedData.length - 1].date).diff(moment(sortedData[0].date), 'days');
    const years = days / 365;
    
    if (years > 0) {
      returns.annualizedReturn = (Math.pow(lastNAV / firstNAV, 1 / years) - 1) * 100;
    }

    // Calculate returns for different periods
    const periods = [7, 30, 90, 180, 365];
    returns.periodReturns = {};

    periods.forEach(period => {
      const periodData = sortedData.filter(item => 
        moment(sortedData[sortedData.length - 1].date).diff(moment(item.date), 'days') <= period
      );
      
      if (periodData.length > 1) {
        const periodFirstNAV = periodData[0].nav;
        const periodLastNAV = periodData[periodData.length - 1].nav;
        returns.periodReturns[`${period}d`] = ((periodLastNAV - periodFirstNAV) / periodFirstNAV) * 100;
      }
    });

    return returns;
  }

  /**
   * Calculate volatility
   */
  calculateVolatility(navData) {
    const volatility = {};
    const sortedData = navData.sort((a, b) => new Date(a.date) - new Date(b.date));
    
    if (sortedData.length < 2) {
      return volatility;
    }

    // Calculate daily returns
    const dailyReturns = [];
    for (let i = 1; i < sortedData.length; i++) {
      const dailyReturn = (sortedData[i].nav - sortedData[i-1].nav) / sortedData[i-1].nav;
      dailyReturns.push(dailyReturn);
    }

    // Calculate standard deviation
    const mean = dailyReturns.reduce((sum, ret) => sum + ret, 0) / dailyReturns.length;
    const variance = dailyReturns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / dailyReturns.length;
    const stdDev = Math.sqrt(variance);

    // Annualized volatility
    volatility.annualized = stdDev * Math.sqrt(252) * 100; // 252 trading days

    // Rolling volatility (30-day)
    volatility.rolling30d = this.calculateRollingVolatility(dailyReturns, 30);

    return volatility;
  }

  /**
   * Calculate risk metrics
   */
  calculateRiskMetrics(navData) {
    const riskMetrics = {};
    const sortedData = navData.sort((a, b) => new Date(a.date) - new Date(b.date));
    
    if (sortedData.length < 2) {
      return riskMetrics;
    }

    // Calculate daily returns
    const dailyReturns = [];
    for (let i = 1; i < sortedData.length; i++) {
      const dailyReturn = (sortedData[i].nav - sortedData[i-1].nav) / sortedData[i-1].nav;
      dailyReturns.push(dailyReturn);
    }

    // Calculate Sharpe ratio (assuming risk-free rate of 6%)
    const meanReturn = dailyReturns.reduce((sum, ret) => sum + ret, 0) / dailyReturns.length;
    const stdDev = Math.sqrt(dailyReturns.reduce((sum, ret) => sum + Math.pow(ret - meanReturn, 2), 0) / dailyReturns.length);
    const riskFreeRate = 0.06 / 252; // Daily risk-free rate
    
    riskMetrics.sharpeRatio = stdDev > 0 ? (meanReturn - riskFreeRate) / stdDev : 0;

    // Calculate maximum drawdown
    riskMetrics.maxDrawdown = this.calculateMaxDrawdown(sortedData);

    // Calculate VaR (Value at Risk) - 95% confidence
    const sortedReturns = dailyReturns.sort((a, b) => a - b);
    const varIndex = Math.floor(dailyReturns.length * 0.05);
    riskMetrics.var95 = sortedReturns[varIndex] * 100;

    // Calculate CVaR (Conditional Value at Risk)
    const varReturns = sortedReturns.slice(0, varIndex + 1);
    riskMetrics.cvar95 = varReturns.reduce((sum, ret) => sum + ret, 0) / varReturns.length * 100;

    return riskMetrics;
  }

  /**
   * Calculate technical indicators
   */
  calculateTechnicalIndicators(navData) {
    const indicators = {};
    const sortedData = navData.sort((a, b) => new Date(a.date) - new Date(b.date));
    
    if (sortedData.length < 20) {
      return indicators;
    }

    // Calculate moving averages
    indicators.sma20 = this.calculateSMA(sortedData, 20);
    indicators.sma50 = this.calculateSMA(sortedData, 50);
    indicators.ema20 = this.calculateEMA(sortedData, 20);

    // Calculate RSI
    indicators.rsi = this.calculateRSI(sortedData, 14);

    // Calculate Bollinger Bands
    indicators.bollingerBands = this.calculateBollingerBands(sortedData, 20, 2);

    return indicators;
  }

  /**
   * Calculate Simple Moving Average
   */
  calculateSMA(data, period) {
    if (data.length < period) {
      return null;
    }

    const navs = data.map(item => item.nav);
    const sma = [];
    
    for (let i = period - 1; i < navs.length; i++) {
      const sum = navs.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      sma.push(sum / period);
    }

    return sma;
  }

  /**
   * Calculate Exponential Moving Average
   */
  calculateEMA(data, period) {
    if (data.length < period) {
      return null;
    }

    const navs = data.map(item => item.nav);
    const ema = [];
    const multiplier = 2 / (period + 1);

    // First EMA is SMA
    let currentEMA = navs.slice(0, period).reduce((a, b) => a + b, 0) / period;
    ema.push(currentEMA);

    // Calculate subsequent EMAs
    for (let i = period; i < navs.length; i++) {
      currentEMA = (navs[i] * multiplier) + (currentEMA * (1 - multiplier));
      ema.push(currentEMA);
    }

    return ema;
  }

  /**
   * Calculate RSI
   */
  calculateRSI(data, period) {
    if (data.length < period + 1) {
      return null;
    }

    const gains = [];
    const losses = [];

    for (let i = 1; i < data.length; i++) {
      const change = data[i].nav - data[i - 1].nav;
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }

    const avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
    const avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;

    const rs = avgGain / avgLoss;
    const rsi = 100 - (100 / (1 + rs));

    return rsi;
  }

  /**
   * Calculate Bollinger Bands
   */
  calculateBollingerBands(data, period, stdDevMultiplier) {
    if (data.length < period) {
      return null;
    }

    const sma = this.calculateSMA(data, period);
    if (!sma) {
      return null;
    }

    const navs = data.map(item => item.nav);
    const bands = {
      upper: [],
      middle: sma,
      lower: []
    };

    for (let i = period - 1; i < navs.length; i++) {
      const slice = navs.slice(i - period + 1, i + 1);
      const mean = slice.reduce((a, b) => a + b, 0) / period;
      const variance = slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / period;
      const stdDev = Math.sqrt(variance);

      bands.upper.push(mean + (stdDev * stdDevMultiplier));
      bands.lower.push(mean - (stdDev * stdDevMultiplier));
    }

    return bands;
  }

  /**
   * Calculate maximum drawdown
   */
  calculateMaxDrawdown(data) {
    let maxDrawdown = 0;
    let peak = data[0].nav;

    for (let i = 1; i < data.length; i++) {
      if (data[i].nav > peak) {
        peak = data[i].nav;
      } else {
        const drawdown = (peak - data[i].nav) / peak;
        if (drawdown > maxDrawdown) {
          maxDrawdown = drawdown;
        }
      }
    }

    return maxDrawdown * 100; // Return as percentage
  }

  /**
   * Calculate rolling volatility
   */
  calculateRollingVolatility(returns, window) {
    if (returns.length < window) {
      return null;
    }

    const rollingVol = [];
    
    for (let i = window - 1; i < returns.length; i++) {
      const windowReturns = returns.slice(i - window + 1, i + 1);
      const mean = windowReturns.reduce((sum, ret) => sum + ret, 0) / window;
      const variance = windowReturns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / window;
      const stdDev = Math.sqrt(variance);
      
      rollingVol.push(stdDev * Math.sqrt(252) * 100); // Annualized
    }

    return rollingVol;
  }

  /**
   * Compute rolling returns
   */
  computeRollingReturns(navData, rollingPeriod) {
    const days = this.periodMap[rollingPeriod] || 30;
    const sortedData = navData.sort((a, b) => new Date(a.date) - new Date(b.date));
    const rollingReturns = [];

    for (let i = days; i < sortedData.length; i++) {
      const startNAV = sortedData[i - days].nav;
      const endNAV = sortedData[i].nav;
      const returnPercent = ((endNAV - startNAV) / startNAV) * 100;
      
      rollingReturns.push({
        date: sortedData[i].date,
        return: returnPercent,
        startNAV,
        endNAV
      });
    }

    return rollingReturns;
  }

  /**
   * Calculate rolling returns summary
   */
  calculateRollingReturnsSummary(rollingReturns) {
    if (rollingReturns.length === 0) {
      return {};
    }

    const returns = rollingReturns.map(r => r.return);
    const sortedReturns = returns.sort((a, b) => a - b);

    return {
      count: returns.length,
      average: returns.reduce((sum, ret) => sum + ret, 0) / returns.length,
      median: sortedReturns[Math.floor(returns.length / 2)],
      min: Math.min(...returns),
      max: Math.max(...returns),
      positiveReturns: returns.filter(r => r > 0).length,
      negativeReturns: returns.filter(r => r < 0).length,
      positivePercentage: (returns.filter(r => r > 0).length / returns.length) * 100
    };
  }

  /**
   * Get benchmark data
   */
  async getBenchmarkData(benchmark, period) {
    try {
      // This would typically fetch from market data service
      // For now, generating mock benchmark data
      const days = this.periodMap[period] || 365;
      const data = [];
      const startDate = moment().subtract(days, 'days');
      
      let baseValue = 18000; // Starting value for NIFTY50
      
      for (let i = 0; i < days; i++) {
        const date = moment(startDate).add(i, 'days');
        
        if (date.day() === 0 || date.day() === 6) {
          continue;
        }

        const dailyReturn = (Math.random() - 0.5) * 0.015; // ±0.75% daily return
        baseValue = baseValue * (1 + dailyReturn);
        
        data.push({
          date: date.toDate(),
          value: parseFloat(baseValue.toFixed(2)),
          change: parseFloat((dailyReturn * 100).toFixed(2))
        });
      }

      return {
        benchmark,
        period,
        data,
        calculations: this.calculateNAVMetrics(data)
      };
    } catch (error) {
      logger.error('Error getting benchmark data:', error);
      throw error;
    }
  }

  /**
   * Calculate comparison metrics
   */
  calculateComparisonMetrics(fundCalculations, benchmarkCalculations) {
    const comparison = {};

    // Return comparison
    if (fundCalculations.returns && benchmarkCalculations.returns) {
      comparison.returnDifference = fundCalculations.returns.annualizedReturn - benchmarkCalculations.returns.annualizedReturn;
      comparison.returnRatio = fundCalculations.returns.annualizedReturn / benchmarkCalculations.returns.annualizedReturn;
    }

    // Volatility comparison
    if (fundCalculations.volatility && benchmarkCalculations.volatility) {
      comparison.volatilityDifference = fundCalculations.volatility.annualized - benchmarkCalculations.volatility.annualized;
      comparison.volatilityRatio = fundCalculations.volatility.annualized / benchmarkCalculations.volatility.annualized;
    }

    // Risk-adjusted return comparison
    if (fundCalculations.riskMetrics && benchmarkCalculations.riskMetrics) {
      comparison.sharpeRatioDifference = fundCalculations.riskMetrics.sharpeRatio - benchmarkCalculations.riskMetrics.sharpeRatio;
    }

    return comparison;
  }

  /**
   * Generate performance analysis
   */
  generatePerformanceAnalysis(performance) {
    const analysis = {
      outperformance: false,
      riskAdjusted: false,
      consistency: false,
      recommendations: []
    };

    // Check outperformance
    if (performance.comparison.returnDifference > 0) {
      analysis.outperformance = true;
      analysis.recommendations.push({
        type: 'positive',
        message: 'Fund is outperforming benchmark',
        action: 'Consider maintaining position'
      });
    } else {
      analysis.recommendations.push({
        type: 'warning',
        message: 'Fund is underperforming benchmark',
        action: 'Review fund selection'
      });
    }

    // Check risk-adjusted performance
    if (performance.comparison.sharpeRatioDifference > 0) {
      analysis.riskAdjusted = true;
      analysis.recommendations.push({
        type: 'positive',
        message: 'Better risk-adjusted returns than benchmark',
        action: 'Good risk management'
      });
    }

    // Check consistency
    if (performance.fund.volatility && performance.fund.volatility.annualized < 20) {
      analysis.consistency = true;
      analysis.recommendations.push({
        type: 'info',
        message: 'Low volatility indicates consistent performance',
        action: 'Suitable for conservative investors'
      });
    }

    return analysis;
  }

  /**
   * Generate NAV summary
   */
  generateNAVSummary(navData, calculations) {
    if (navData.length === 0) {
      return {};
    }

    const sortedData = navData.sort((a, b) => new Date(a.date) - new Date(b.date));
    const firstNAV = sortedData[0].nav;
    const lastNAV = sortedData[sortedData.length - 1].nav;
    const highestNAV = Math.max(...sortedData.map(item => item.nav));
    const lowestNAV = Math.min(...sortedData.map(item => item.nav));

    return {
      period: {
        start: sortedData[0].date,
        end: sortedData[sortedData.length - 1].date,
        days: sortedData.length
      },
      nav: {
        start: firstNAV,
        end: lastNAV,
        highest: highestNAV,
        lowest: lowestNAV,
        change: lastNAV - firstNAV,
        changePercent: ((lastNAV - firstNAV) / firstNAV) * 100
      },
      performance: calculations.returns || {},
      risk: calculations.riskMetrics || {},
      volatility: calculations.volatility || {}
    };
  }
}

module.exports = new NAVHistoryService(); 