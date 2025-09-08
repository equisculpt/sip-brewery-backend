/**
 * 📊 PERFORMANCE ANALYZER
 * 
 * Advanced performance metrics and analysis for backtesting results
 * 
 * @author 35+ Years ASI Engineering Experience
 * @version 4.0.0 - Performance Analytics Engine
 */

const logger = require('../../utils/logger');

class PerformanceAnalyzer {
  constructor(options = {}) {
    this.config = {
      riskFreeRate: options.riskFreeRate || 0.02,
      benchmarkReturn: options.benchmarkReturn || 0.08,
      confidenceLevel: options.confidenceLevel || 0.95,
      ...options
    };
  }

  calculateMetrics(portfolioHistory, benchmarkReturns = []) {
    if (portfolioHistory.length === 0) return {};
    
    const returns = portfolioHistory.map(snapshot => snapshot.dailyReturn).slice(1);
    const values = portfolioHistory.map(snapshot => snapshot.totalValue);
    
    // Basic performance metrics
    const totalReturn = (values[values.length - 1] / values[0]) - 1;
    const annualizedReturn = Math.pow(1 + totalReturn, 252 / returns.length) - 1;
    
    // Risk metrics
    const volatility = this.calculateVolatility(returns);
    const annualizedVolatility = volatility * Math.sqrt(252);
    const maxDrawdown = this.calculateMaxDrawdown(values);
    
    // Risk-adjusted metrics
    const sharpeRatio = (annualizedReturn - this.config.riskFreeRate) / annualizedVolatility;
    const sortinoRatio = this.calculateSortinoRatio(returns, this.config.riskFreeRate);
    const calmarRatio = annualizedReturn / Math.abs(maxDrawdown);
    
    // Advanced metrics
    const var95 = this.calculateVaR(returns, 0.95);
    const cvar95 = this.calculateCVaR(returns, 0.95);
    const skewness = this.calculateSkewness(returns);
    const kurtosis = this.calculateKurtosis(returns);
    
    // Benchmark comparison
    let alpha = 0, beta = 1, informationRatio = 0, trackingError = 0;
    if (benchmarkReturns.length === returns.length) {
      beta = this.calculateBeta(returns, benchmarkReturns);
      alpha = annualizedReturn - (this.config.riskFreeRate + beta * (this.config.benchmarkReturn - this.config.riskFreeRate));
      trackingError = this.calculateTrackingError(returns, benchmarkReturns);
      informationRatio = alpha / trackingError;
    }
    
    // Win/Loss statistics
    const winRate = returns.filter(r => r > 0).length / returns.length;
    const avgWin = returns.filter(r => r > 0).reduce((sum, r) => sum + r, 0) / returns.filter(r => r > 0).length || 0;
    const avgLoss = returns.filter(r => r < 0).reduce((sum, r) => sum + r, 0) / returns.filter(r => r < 0).length || 0;
    const profitFactor = Math.abs(avgWin * winRate / (avgLoss * (1 - winRate)));
    
    return {
      totalReturn,
      annualizedReturn,
      volatility: annualizedVolatility,
      maxDrawdown,
      sharpeRatio,
      sortinoRatio,
      calmarRatio,
      var95,
      cvar95,
      skewness,
      kurtosis,
      alpha,
      beta,
      informationRatio,
      trackingError,
      winRate,
      avgWin,
      avgLoss,
      profitFactor,
      tradingDays: returns.length
    };
  }

  attributePerformance(portfolioHistory, sectorData = {}) {
    const attribution = {
      assetAllocation: {},
      sectorAllocation: {},
      securitySelection: {},
      interaction: {}
    };
    
    for (const snapshot of portfolioHistory) {
      for (const [symbol, position] of snapshot.positions) {
        const weight = position.marketValue / snapshot.totalValue;
        const sector = sectorData[symbol] || 'Unknown';
        
        if (!attribution.assetAllocation[symbol]) {
          attribution.assetAllocation[symbol] = { weight: 0, return: 0 };
        }
        
        attribution.assetAllocation[symbol].weight += weight;
      }
    }
    
    return attribution;
  }

  calculateVolatility(returns) {
    if (returns.length === 0) return 0;
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1);
    return Math.sqrt(variance);
  }

  calculateMaxDrawdown(values) {
    let maxDrawdown = 0;
    let peak = values[0];
    
    for (const value of values) {
      if (value > peak) {
        peak = value;
      }
      const drawdown = (peak - value) / peak;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }
    
    return maxDrawdown;
  }

  calculateAverageDrawdown(values) {
    const drawdowns = [];
    let peak = values[0];
    let inDrawdown = false;
    let currentDrawdown = 0;
    
    for (const value of values) {
      if (value > peak) {
        if (inDrawdown) {
          drawdowns.push(currentDrawdown);
          inDrawdown = false;
          currentDrawdown = 0;
        }
        peak = value;
      } else {
        const drawdown = (peak - value) / peak;
        currentDrawdown = Math.max(currentDrawdown, drawdown);
        inDrawdown = true;
      }
    }
    
    if (inDrawdown) {
      drawdowns.push(currentDrawdown);
    }
    
    return drawdowns.length > 0 ? drawdowns.reduce((sum, d) => sum + d, 0) / drawdowns.length : 0;
  }

  calculateDrawdownDuration(values) {
    let maxDuration = 0;
    let currentDuration = 0;
    let peak = values[0];
    
    for (let i = 1; i < values.length; i++) {
      if (values[i] > peak) {
        peak = values[i];
        maxDuration = Math.max(maxDuration, currentDuration);
        currentDuration = 0;
      } else {
        currentDuration++;
      }
    }
    
    return Math.max(maxDuration, currentDuration);
  }

  calculateSortinoRatio(returns, riskFreeRate) {
    const excessReturns = returns.map(r => r - riskFreeRate / 252);
    const downside = excessReturns.filter(r => r < 0);
    
    if (downside.length === 0) return Infinity;
    
    const downsideDeviation = Math.sqrt(
      downside.reduce((sum, r) => sum + r * r, 0) / downside.length
    );
    
    const avgExcessReturn = excessReturns.reduce((sum, r) => sum + r, 0) / excessReturns.length;
    
    return (avgExcessReturn * 252) / (downsideDeviation * Math.sqrt(252));
  }

  calculateVaR(returns, confidenceLevel) {
    const sortedReturns = returns.slice().sort((a, b) => a - b);
    const index = Math.floor((1 - confidenceLevel) * sortedReturns.length);
    return -sortedReturns[index];
  }

  calculateCVaR(returns, confidenceLevel) {
    const var95 = this.calculateVaR(returns, confidenceLevel);
    const tailReturns = returns.filter(r => r <= -var95);
    
    if (tailReturns.length === 0) return var95;
    
    return -tailReturns.reduce((sum, r) => sum + r, 0) / tailReturns.length;
  }

  calculateExpectedShortfall(returns, alpha) {
    const sortedReturns = returns.slice().sort((a, b) => a - b);
    const cutoffIndex = Math.floor(alpha * sortedReturns.length);
    const tailReturns = sortedReturns.slice(0, cutoffIndex);
    
    return tailReturns.length > 0 ? 
      -tailReturns.reduce((sum, r) => sum + r, 0) / tailReturns.length : 0;
  }

  calculateSkewness(returns) {
    const n = returns.length;
    const mean = returns.reduce((sum, r) => sum + r, 0) / n;
    const std = this.calculateVolatility(returns);
    
    const skewness = returns.reduce((sum, r) => {
      return sum + Math.pow((r - mean) / std, 3);
    }, 0) / n;
    
    return skewness;
  }

  calculateKurtosis(returns) {
    const n = returns.length;
    const mean = returns.reduce((sum, r) => sum + r, 0) / n;
    const std = this.calculateVolatility(returns);
    
    const kurtosis = returns.reduce((sum, r) => {
      return sum + Math.pow((r - mean) / std, 4);
    }, 0) / n;
    
    return kurtosis - 3; // Excess kurtosis
  }

  calculateBeta(portfolioReturns, benchmarkReturns) {
    if (portfolioReturns.length !== benchmarkReturns.length || portfolioReturns.length === 0) {
      return 1;
    }
    
    const covariance = this.calculateCovariance(portfolioReturns, benchmarkReturns);
    const benchmarkVariance = this.calculateVariance(benchmarkReturns);
    
    return benchmarkVariance > 0 ? covariance / benchmarkVariance : 1;
  }

  calculateTrackingError(portfolioReturns, benchmarkReturns) {
    if (portfolioReturns.length !== benchmarkReturns.length) return 0;
    
    const excessReturns = portfolioReturns.map((r, i) => r - benchmarkReturns[i]);
    return this.calculateVolatility(excessReturns) * Math.sqrt(252);
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

  calculateRollingVolatility(returns, window) {
    const rollingVols = [];
    
    for (let i = window; i <= returns.length; i++) {
      const windowReturns = returns.slice(i - window, i);
      rollingVols.push(this.calculateVolatility(windowReturns));
    }
    
    return rollingVols;
  }

  calculateRollingMetrics(portfolioHistory, window = 30) {
    const rollingMetrics = [];
    
    for (let i = window; i <= portfolioHistory.length; i++) {
      const windowHistory = portfolioHistory.slice(i - window, i);
      const metrics = this.calculateMetrics(windowHistory);
      
      rollingMetrics.push({
        endDate: windowHistory[windowHistory.length - 1].timestamp,
        ...metrics
      });
    }
    
    return rollingMetrics;
  }

  generatePerformanceReport(portfolioHistory, benchmarkReturns = [], options = {}) {
    const metrics = this.calculateMetrics(portfolioHistory, benchmarkReturns);
    const attribution = this.attributePerformance(portfolioHistory, options.sectorData);
    const rollingMetrics = this.calculateRollingMetrics(portfolioHistory, options.rollingWindow || 30);
    
    return {
      summary: {
        totalReturn: metrics.totalReturn,
        annualizedReturn: metrics.annualizedReturn,
        volatility: metrics.volatility,
        sharpeRatio: metrics.sharpeRatio,
        maxDrawdown: metrics.maxDrawdown,
        winRate: metrics.winRate
      },
      detailedMetrics: metrics,
      attribution,
      rollingMetrics: rollingMetrics.slice(-12), // Last 12 periods
      riskAnalysis: {
        var95: metrics.var95,
        cvar95: metrics.cvar95,
        skewness: metrics.skewness,
        kurtosis: metrics.kurtosis,
        maxDrawdown: metrics.maxDrawdown,
        averageDrawdown: this.calculateAverageDrawdown(portfolioHistory.map(s => s.totalValue))
      },
      benchmarkComparison: benchmarkReturns.length > 0 ? {
        alpha: metrics.alpha,
        beta: metrics.beta,
        informationRatio: metrics.informationRatio,
        trackingError: metrics.trackingError
      } : null
    };
  }
}

module.exports = { PerformanceAnalyzer };
