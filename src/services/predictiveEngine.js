const logger = require('../utils/logger');
const { Fund, FundPerformance, MacroData } = require('../models');
const aiPortfolioOptimizer = require('./aiPortfolioOptimizer');

class PredictiveEngine {
  constructor() {
    this.predictionModels = {
      FUND_PERFORMANCE: 'fund_performance',
      MARKET_TREND: 'market_trend',
      RISK_ASSESSMENT: 'risk_assessment',
      SECTOR_ANALYSIS: 'sector_analysis'
    };

    this.earlyWarningThresholds = {
      NAV_DECLINE: 0.05, // 5% decline
      AUM_DECLINE: 0.1, // 10% decline
      SECTOR_STRESS: 0.15, // 15% sector decline
      MANAGER_CHANGE: 1.0 // Any manager change
    };
  }

  /**
   * Predict mutual fund performance using AI
   */
  async predictFundPerformance(fundData, marketConditions, timeHorizon = '3M') {
    try {
      logger.info('Starting fund performance prediction', { 
        schemeCode: fundData.schemeCode, 
        timeHorizon 
      });

      // Gather historical data
      const historicalData = await this.gatherHistoricalData(fundData.schemeCode);
      const macroData = await this.getMacroData();
      const sectorData = await this.getSectorData(fundData.category);

      // Calculate prediction features
      const features = await this.calculatePredictionFeatures(
        fundData, 
        historicalData, 
        macroData, 
        sectorData
      );

      // Generate predictions for different time horizons
      const predictions = {
        prediction1M: await this.predictForHorizon(features, '1M'),
        prediction3M: await this.predictForHorizon(features, '3M'),
        prediction6M: await this.predictForHorizon(features, '6M'),
        prediction1Y: await this.predictForHorizon(features, '1Y')
      };

      // Calculate risk assessment
      const riskAssessment = await this.assessRisk(features, marketConditions);

      // Generate confidence levels
      const confidenceLevels = this.calculateConfidenceLevels(features, historicalData);

      // Identify key factors
      const keyFactors = this.identifyKeyFactors(features, predictions);

      // Check for early warnings
      const earlyWarnings = await this.checkEarlyWarnings(fundData, historicalData);

      const result = {
        success: true,
        data: {
          schemeCode: fundData.schemeCode,
          fundName: fundData.fundName,
          predictions,
          riskAssessment,
          confidenceLevels,
          keyFactors,
          earlyWarnings,
          predictionScore: this.calculatePredictionScore(predictions, confidenceLevels),
          trend: this.determineTrend(predictions),
          next3MonthSignal: this.generateSignal(predictions.prediction3M, riskAssessment)
        }
      };

      logger.info('Fund performance prediction completed', { 
        schemeCode: fundData.schemeCode,
        predictionScore: result.data.predictionScore 
      });

      return result;
    } catch (error) {
      logger.error('Fund performance prediction failed', { error: error.message });
      return {
        success: false,
        message: 'Failed to predict fund performance',
        error: error.message
      };
    }
  }

  /**
   * Generate early warning alerts for SIP decisions
   */
  async generateEarlyWarnings(userId, portfolio) {
    try {
      logger.info('Generating early warnings for user', { userId });

      const warnings = [];

      // Check each holding in the portfolio
      for (const holding of portfolio.holdings) {
        const fundWarnings = await this.checkFundWarnings(holding);
        warnings.push(...fundWarnings);
      }

      // Check portfolio-level warnings
      const portfolioWarnings = await this.checkPortfolioWarnings(portfolio);
      warnings.push(...portfolioWarnings);

      // Check market-level warnings
      const marketWarnings = await this.checkMarketWarnings();
      warnings.push(...marketWarnings);

      // Generate actionable recommendations
      const recommendations = this.generateRecommendations(warnings, portfolio);

      return {
        success: true,
        data: {
          warnings,
          recommendations,
          alertLevel: this.calculateAlertLevel(warnings),
          shouldPauseSIP: this.shouldPauseSIP(warnings),
          shouldReduceSIP: this.shouldReduceSIP(warnings)
        }
      };
    } catch (error) {
      logger.error('Early warning generation failed', { error: error.message });
      return {
        success: false,
        message: 'Failed to generate early warnings',
        error: error.message
      };
    }
  }

  /**
   * Predict market trends for Indian markets
   */
  async predictMarketTrends(timeHorizon = '3M') {
    try {
      logger.info('Starting market trend prediction', { timeHorizon });

      // Gather market data
      const niftyData = await this.getNiftyData();
      const sensexData = await this.getSensexData();
      const sectorData = await this.getSectorPerformance();
      const macroData = await this.getMacroData();

      // Analyze market indicators
      const indicators = this.analyzeMarketIndicators(niftyData, sensexData, sectorData, macroData);

      // Generate trend predictions
      const trends = {
        nifty: await this.predictIndexTrend(niftyData, indicators, timeHorizon),
        sensex: await this.predictIndexTrend(sensexData, indicators, timeHorizon),
        sectors: await this.predictSectorTrends(sectorData, indicators, timeHorizon)
      };

      // Calculate market sentiment
      const sentiment = this.calculateMarketSentiment(indicators);

      // Generate investment signals
      const signals = this.generateMarketSignals(trends, sentiment);

      return {
        success: true,
        data: {
          trends,
          sentiment,
          signals,
          indicators,
          confidence: this.calculateMarketConfidence(indicators)
        }
      };
    } catch (error) {
      logger.error('Market trend prediction failed', { error: error.message });
      return {
        success: false,
        message: 'Failed to predict market trends',
        error: error.message
      };
    }
  }

  // Helper methods for fund performance prediction
  async gatherHistoricalData(schemeCode) {
    try {
      // In real implementation, fetch from fund database
      return {
        navHistory: [
          { date: '2024-01-01', nav: 100 },
          { date: '2024-02-01', nav: 105 },
          { date: '2024-03-01', nav: 110 },
          { date: '2024-04-01', nav: 108 },
          { date: '2024-05-01', nav: 115 }
        ],
        aumHistory: [
          { date: '2024-01-01', aum: 1000000000 },
          { date: '2024-02-01', aum: 1050000000 },
          { date: '2024-03-01', aum: 1100000000 },
          { date: '2024-04-01', aum: 1080000000 },
          { date: '2024-05-01', aum: 1150000000 }
        ],
        returns: {
          '1M': 0.05,
          '3M': 0.15,
          '6M': 0.25,
          '1Y': 0.35,
          '3Y': 0.45
        }
      };
    } catch (error) {
      logger.error('Error gathering historical data', { error: error.message });
      return null;
    }
  }

  async getMacroData() {
    try {
      // In real implementation, fetch from RBI/MOSPI APIs
      return {
        gdp: 7.2,
        inflation: 5.5,
        repoRate: 6.5,
        fiscalDeficit: 5.8,
        currentAccountDeficit: 1.2
      };
    } catch (error) {
      logger.error('Error fetching macro data', { error: error.message });
      return {};
    }
  }

  async getSectorData(category) {
    try {
      // In real implementation, fetch sector-specific data
      return {
        sectorPerformance: 0.12,
        sectorValuation: 18.5,
        sectorMomentum: 0.08,
        sectorRisk: 0.15
      };
    } catch (error) {
      logger.error('Error fetching sector data', { error: error.message });
      return {};
    }
  }

  async calculatePredictionFeatures(fundData, historicalData, macroData, sectorData) {
    try {
      const features = {};

      // NAV trend features
      if (historicalData && historicalData.navHistory) {
        const navTrend = this.calculateTrend(historicalData.navHistory.map(h => h.nav));
        features.navTrend = navTrend;
        features.navVolatility = this.calculateVolatility(historicalData.navHistory.map(h => h.nav));
        features.navMomentum = this.calculateMomentum(historicalData.navHistory.map(h => h.nav));
      }

      // AUM features
      if (historicalData && historicalData.aumHistory) {
        const aumTrend = this.calculateTrend(historicalData.aumHistory.map(h => h.aum));
        features.aumTrend = aumTrend;
        features.aumGrowth = this.calculateGrowth(historicalData.aumHistory.map(h => h.aum));
      }

      // Return features
      if (historicalData && historicalData.returns) {
        features.returns1M = historicalData.returns['1M'];
        features.returns3M = historicalData.returns['3M'];
        features.returns6M = historicalData.returns['6M'];
        features.returns1Y = historicalData.returns['1Y'];
        features.returns3Y = historicalData.returns['3Y'];
      }

      // Macro features
      features.gdp = macroData.gdp || 0;
      features.inflation = macroData.inflation || 0;
      features.repoRate = macroData.repoRate || 0;

      // Sector features
      features.sectorPerformance = sectorData.sectorPerformance || 0;
      features.sectorValuation = sectorData.sectorValuation || 0;
      features.sectorMomentum = sectorData.sectorMomentum || 0;

      // Fund-specific features
      features.expenseRatio = fundData.expenseRatio || 0;
      features.aum = fundData.aum || 0;
      features.fundAge = fundData.fundAge || 0;

      return features;
    } catch (error) {
      logger.error('Error calculating prediction features', { error: error.message });
      return {};
    }
  }

  async predictForHorizon(features, horizon) {
    try {
      // Simple linear regression model (in real implementation, use ML models)
      const baseReturn = features.returns1Y || 0.12;
      const navMomentum = features.navMomentum || 0;
      const sectorMomentum = features.sectorMomentum || 0;
      const macroFactor = (features.gdp - 6) * 0.01; // GDP impact

      let prediction = baseReturn;

      // Adjust for momentum
      prediction += navMomentum * 0.3;
      prediction += sectorMomentum * 0.2;

      // Adjust for macro factors
      prediction += macroFactor;

      // Adjust for time horizon
      const horizonMultiplier = {
        '1M': 0.25,
        '3M': 0.75,
        '6M': 1.5,
        '1Y': 3.0
      };

      prediction *= horizonMultiplier[horizon] || 1;

      return Math.max(-0.5, Math.min(1.0, prediction)); // Cap between -50% and 100%
    } catch (error) {
      logger.error('Error predicting for horizon', { error: error.message });
      return 0;
    }
  }

  async assessRisk(features, marketConditions) {
    try {
      const riskFactors = [];

      // NAV volatility risk
      if (features.navVolatility > 0.2) {
        riskFactors.push({
          factor: 'High NAV Volatility',
          score: features.navVolatility * 100,
          impact: 'HIGH'
        });
      }

      // AUM decline risk
      if (features.aumTrend < -0.1) {
        riskFactors.push({
          factor: 'AUM Decline',
          score: Math.abs(features.aumTrend) * 100,
          impact: 'MEDIUM'
        });
      }

      // Sector risk
      if (features.sectorRisk > 0.2) {
        riskFactors.push({
          factor: 'Sector Risk',
          score: features.sectorRisk * 100,
          impact: 'HIGH'
        });
      }

      // Macro risk
      if (features.inflation > 6) {
        riskFactors.push({
          factor: 'High Inflation',
          score: (features.inflation - 6) * 20,
          impact: 'MEDIUM'
        });
      }

      const overallRisk = riskFactors.reduce((sum, factor) => sum + factor.score, 0) / riskFactors.length;

      return {
        overallRisk: Math.min(100, overallRisk),
        riskLevel: this.getRiskLevel(overallRisk),
        riskFactors,
        volatility: features.navVolatility || 0.15,
        downsideRisk: this.calculateDownsideRisk(features),
        stressTestResults: this.runStressTests(features, marketConditions)
      };
    } catch (error) {
      logger.error('Error assessing risk', { error: error.message });
      return {
        overallRisk: 50,
        riskLevel: 'MODERATE',
        riskFactors: [],
        volatility: 0.15,
        downsideRisk: 0.10,
        stressTestResults: {}
      };
    }
  }

  calculateConfidenceLevels(features, historicalData) {
    try {
      let confidence = 0.7; // Base confidence

      // Increase confidence with more historical data
      if (historicalData && historicalData.navHistory) {
        confidence += Math.min(0.2, historicalData.navHistory.length * 0.01);
      }

      // Decrease confidence with high volatility
      if (features.navVolatility > 0.2) {
        confidence -= 0.1;
      }

      // Decrease confidence with low AUM
      if (features.aum < 100000000) { // Less than 10 crore
        confidence -= 0.1;
      }

      return Math.max(0.3, Math.min(0.95, confidence));
    } catch (error) {
      logger.error('Error calculating confidence levels', { error: error.message });
      return 0.7;
    }
  }

  identifyKeyFactors(features, predictions) {
    const factors = [];

    // NAV momentum
    if (features.navMomentum > 0.05) {
      factors.push('Strong NAV momentum');
    } else if (features.navMomentum < -0.05) {
      factors.push('Weak NAV momentum');
    }

    // Sector performance
    if (features.sectorPerformance > 0.1) {
      factors.push('Strong sector performance');
    } else if (features.sectorPerformance < -0.1) {
      factors.push('Weak sector performance');
    }

    // Macro conditions
    if (features.gdp > 7) {
      factors.push('Strong GDP growth');
    } else if (features.gdp < 6) {
      factors.push('Weak GDP growth');
    }

    // Fund-specific factors
    if (features.expenseRatio < 0.015) {
      factors.push('Low expense ratio');
    }

    if (features.aum > 1000000000) { // More than 100 crore
      factors.push('Large fund size');
    }

    return factors;
  }

  async checkEarlyWarnings(fundData, historicalData) {
    const warnings = [];

    try {
      // Check NAV decline
      if (historicalData && historicalData.navHistory) {
        const recentNav = historicalData.navHistory[historicalData.navHistory.length - 1].nav;
        const previousNav = historicalData.navHistory[historicalData.navHistory.length - 2].nav;
        const navDecline = (previousNav - recentNav) / previousNav;

        if (navDecline > this.earlyWarningThresholds.NAV_DECLINE) {
          warnings.push({
            type: 'NAV_DECLINE',
            severity: 'HIGH',
            message: `NAV declined by ${(navDecline * 100).toFixed(1)}% in the last period`,
            action: 'Consider reducing SIP or switching to better performing funds'
          });
        }
      }

      // Check AUM decline
      if (historicalData && historicalData.aumHistory) {
        const recentAum = historicalData.aumHistory[historicalData.aumHistory.length - 1].aum;
        const previousAum = historicalData.aumHistory[historicalData.aumHistory.length - 2].aum;
        const aumDecline = (previousAum - recentAum) / previousAum;

        if (aumDecline > this.earlyWarningThresholds.AUM_DECLINE) {
          warnings.push({
            type: 'AUM_DECLINE',
            severity: 'MEDIUM',
            message: `AUM declined by ${(aumDecline * 100).toFixed(1)}% in the last period`,
            action: 'Monitor fund performance closely'
          });
        }
      }

      return warnings;
    } catch (error) {
      logger.error('Error checking early warnings', { error: error.message });
      return warnings;
    }
  }

  calculatePredictionScore(predictions, confidenceLevels) {
    try {
      const avgPrediction = (predictions.prediction3M + predictions.prediction6M) / 2;
      const score = (avgPrediction + 0.5) * confidenceLevels * 100; // Convert to 0-100 scale
      return Math.max(1, Math.min(100, score));
    } catch (error) {
      logger.error('Error calculating prediction score', { error: error.message });
      return 50;
    }
  }

  determineTrend(predictions) {
    try {
      const trend = (predictions.prediction3M + predictions.prediction6M) / 2;
      if (trend > 0.05) return 'BULLISH';
      if (trend < -0.05) return 'BEARISH';
      return 'NEUTRAL';
    } catch (error) {
      logger.error('Error determining trend', { error: error.message });
      return 'NEUTRAL';
    }
  }

  generateSignal(prediction3M, riskAssessment) {
    try {
      if (prediction3M > 0.1 && riskAssessment.overallRisk < 60) {
        return 'BUY';
      } else if (prediction3M < -0.1 || riskAssessment.overallRisk > 80) {
        return 'SELL';
      } else {
        return 'HOLD';
      }
    } catch (error) {
      logger.error('Error generating signal', { error: error.message });
      return 'HOLD';
    }
  }

  // Helper methods for early warnings
  async checkFundWarnings(holding) {
    const warnings = [];

    try {
      // Check for underperformance
      if (holding.returns1Y < -0.1) {
        warnings.push({
          type: 'FUND_UNDERPERFORMANCE',
          severity: 'HIGH',
          fundName: holding.fundName,
          message: `${holding.fundName} has underperformed by ${Math.abs(holding.returns1Y * 100).toFixed(1)}% in the last year`,
          action: 'Consider switching to better performing funds'
        });
      }

      // Check for high expense ratio
      if (holding.expenseRatio > 0.025) {
        warnings.push({
          type: 'HIGH_EXPENSE_RATIO',
          severity: 'MEDIUM',
          fundName: holding.fundName,
          message: `${holding.fundName} has a high expense ratio of ${(holding.expenseRatio * 100).toFixed(2)}%`,
          action: 'Consider funds with lower expense ratios'
        });
      }

      return warnings;
    } catch (error) {
      logger.error('Error checking fund warnings', { error: error.message });
      return warnings;
    }
  }

  async checkPortfolioWarnings(portfolio) {
    const warnings = [];

    try {
      // Check for over-concentration
      const totalValue = portfolio.holdings.reduce((sum, h) => sum + h.currentValue, 0);
      const maxAllocation = Math.max(...portfolio.holdings.map(h => h.currentValue / totalValue));

      if (maxAllocation > 0.3) {
        warnings.push({
          type: 'OVER_CONCENTRATION',
          severity: 'MEDIUM',
          message: 'Portfolio is over-concentrated in a single fund',
          action: 'Consider diversifying across more funds'
        });
      }

      // Check for low diversification
      const uniqueCategories = new Set(portfolio.holdings.map(h => h.fundCategory)).size;
      if (uniqueCategories < 3) {
        warnings.push({
          type: 'LOW_DIVERSIFICATION',
          severity: 'MEDIUM',
          message: 'Portfolio lacks diversification across fund categories',
          action: 'Consider adding funds from different categories'
        });
      }

      return warnings;
    } catch (error) {
      logger.error('Error checking portfolio warnings', { error: error.message });
      return warnings;
    }
  }

  async checkMarketWarnings() {
    const warnings = [];

    try {
      // Check for high market volatility
      const marketData = await this.getMarketData();
      if (marketData.volatility > 0.25) {
        warnings.push({
          type: 'HIGH_MARKET_VOLATILITY',
          severity: 'HIGH',
          message: 'Market is experiencing high volatility',
          action: 'Consider reducing equity exposure or increasing SIP frequency'
        });
      }

      return warnings;
    } catch (error) {
      logger.error('Error checking market warnings', { error: error.message });
      return warnings;
    }
  }

  generateRecommendations(warnings, portfolio) {
    const recommendations = [];

    try {
      // Group warnings by type
      const warningTypes = warnings.reduce((acc, warning) => {
        acc[warning.type] = (acc[warning.type] || 0) + 1;
        return acc;
      }, {});

      // Generate recommendations based on warning patterns
      if (warningTypes.FUND_UNDERPERFORMANCE > 0) {
        recommendations.push({
          type: 'SWITCH_FUNDS',
          priority: 'HIGH',
          description: 'Switch underperforming funds to better alternatives',
          action: 'Review and replace funds with consistent underperformance'
        });
      }

      if (warningTypes.OVER_CONCENTRATION > 0) {
        recommendations.push({
          type: 'DIVERSIFY',
          priority: 'MEDIUM',
          description: 'Diversify portfolio to reduce concentration risk',
          action: 'Add funds from different categories and fund houses'
        });
      }

      if (warningTypes.HIGH_MARKET_VOLATILITY > 0) {
        recommendations.push({
          type: 'REDUCE_RISK',
          priority: 'HIGH',
          description: 'Reduce portfolio risk during high market volatility',
          action: 'Increase debt fund allocation or reduce SIP amounts'
        });
      }

      return recommendations;
    } catch (error) {
      logger.error('Error generating recommendations', { error: error.message });
      return [];
    }
  }

  calculateAlertLevel(warnings) {
    try {
      const highSeverityCount = warnings.filter(w => w.severity === 'HIGH').length;
      const mediumSeverityCount = warnings.filter(w => w.severity === 'MEDIUM').length;

      if (highSeverityCount > 2) return 'CRITICAL';
      if (highSeverityCount > 0 || mediumSeverityCount > 3) return 'HIGH';
      if (mediumSeverityCount > 0) return 'MEDIUM';
      return 'LOW';
    } catch (error) {
      logger.error('Error calculating alert level', { error: error.message });
      return 'LOW';
    }
  }

  shouldPauseSIP(warnings) {
    try {
      const criticalWarnings = warnings.filter(w => 
        w.severity === 'HIGH' && 
        (w.type === 'FUND_UNDERPERFORMANCE' || w.type === 'HIGH_MARKET_VOLATILITY')
      );
      return criticalWarnings.length > 1;
    } catch (error) {
      logger.error('Error determining SIP pause', { error: error.message });
      return false;
    }
  }

  shouldReduceSIP(warnings) {
    try {
      const highSeverityWarnings = warnings.filter(w => w.severity === 'HIGH');
      return highSeverityWarnings.length > 0;
    } catch (error) {
      logger.error('Error determining SIP reduction', { error: error.message });
      return false;
    }
  }

  // Utility methods
  calculateTrend(values) {
    if (values.length < 2) return 0;
    const n = values.length;
    const sumX = (n * (n - 1)) / 2;
    const sumY = values.reduce((sum, val, i) => sum + val * i, 0);
    const sumXY = values.reduce((sum, val, i) => sum + val * i, 0);
    const sumX2 = values.reduce((sum, val, i) => sum + i * i, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    return slope;
  }

  calculateVolatility(values) {
    if (values.length < 2) return 0;
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (values.length - 1);
    return Math.sqrt(variance);
  }

  calculateMomentum(values) {
    if (values.length < 2) return 0;
    const recent = values.slice(-3); // Last 3 values
    const older = values.slice(-6, -3); // Previous 3 values
    const recentAvg = recent.reduce((sum, val) => sum + val, 0) / recent.length;
    const olderAvg = older.reduce((sum, val) => sum + val, 0) / older.length;
    return (recentAvg - olderAvg) / olderAvg;
  }

  calculateGrowth(values) {
    if (values.length < 2) return 0;
    const first = values[0];
    const last = values[values.length - 1];
    return (last - first) / first;
  }

  getRiskLevel(riskScore) {
    if (riskScore < 30) return 'LOW';
    if (riskScore < 60) return 'MODERATE';
    if (riskScore < 80) return 'HIGH';
    return 'VERY_HIGH';
  }

  calculateDownsideRisk(features) {
    return features.navVolatility * 0.7; // Simplified calculation
  }

  runStressTests(features, marketConditions) {
    return {
      marketCrash: -0.2,
      recession: -0.15,
      inflation: 0.02
    };
  }

  async getMarketData() {
    return {
      volatility: 0.15,
      trend: 'BULLISH'
    };
  }

  async getNiftyData() {
    return {
      current: 22000,
      change: 0.5,
      volume: 1000000000
    };
  }

  async getSensexData() {
    return {
      current: 72000,
      change: 0.4,
      volume: 2000000000
    };
  }

  async getSectorPerformance() {
    return {
      technology: 0.15,
      healthcare: 0.12,
      finance: 0.08,
      consumer: 0.10
    };
  }

  analyzeMarketIndicators(niftyData, sensexData, sectorData, macroData) {
    return {
      trend: 'BULLISH',
      momentum: 0.08,
      volatility: 0.15,
      sentiment: 0.7
    };
  }

  async predictIndexTrend(indexData, indicators, timeHorizon) {
    return 0.12; // Mock prediction
  }

  async predictSectorTrends(sectorData, indicators, timeHorizon) {
    return {
      technology: 0.15,
      healthcare: 0.10,
      finance: 0.08
    };
  }

  calculateMarketSentiment(indicators) {
    return {
      score: 0.7,
      level: 'POSITIVE',
      factors: ['Strong GDP growth', 'Stable inflation']
    };
  }

  generateMarketSignals(trends, sentiment) {
    return {
      nifty: 'BUY',
      sensex: 'BUY',
      overall: 'BULLISH'
    };
  }

  calculateMarketConfidence(indicators) {
    return 0.75;
  }
}

module.exports = new PredictiveEngine(); 