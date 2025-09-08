const logger = require('../utils/logger');
const marketAnalyticsEngine = require('../services/marketAnalyticsEngine');
const { authenticateUser } = require('../middleware/auth');

class MarketAnalyticsController {
  /**
   * Scrape NSE/BSE daily data
   */
  async scrapeMarketData(req, res) {
    try {
      logger.info('Market data scraping request received');

      const { date } = req.body;
      const scrapingDate = date ? new Date(date) : new Date();

      const result = await marketAnalyticsEngine.scrapeMarketData(scrapingDate);

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'Market data scraped successfully',
          data: result.data
        });
      } else {
        res.status(500).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Market data scraping controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to scrape market data',
        error: error.message
      });
    }
  }

  /**
   * Analyze market sentiment
   */
  async analyzeMarketSentiment(req, res) {
    try {
      logger.info('Market sentiment analysis request received');

      const { period } = req.body;
      const analysisPeriod = period || '1d';

      const result = await marketAnalyticsEngine.analyzeMarketSentiment(analysisPeriod);

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'Market sentiment analyzed successfully',
          data: result.data
        });
      } else {
        res.status(500).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Market sentiment analysis controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to analyze market sentiment',
        error: error.message
      });
    }
  }

  /**
   * Fetch macroeconomic data
   */
  async fetchMacroData(req, res) {
    try {
      logger.info('Macro data fetching request received');

      const result = await marketAnalyticsEngine.fetchMacroData();

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'Macro data fetched successfully',
          data: result.data
        });
      } else {
        res.status(500).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Macro data fetching controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to fetch macro data',
        error: error.message
      });
    }
  }

  /**
   * Analyze sector correlations
   */
  async analyzeSectorCorrelations(req, res) {
    try {
      logger.info('Sector correlation analysis request received');

      const result = await marketAnalyticsEngine.analyzeSectorCorrelations();

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'Sector correlations analyzed successfully',
          data: result.data
        });
      } else {
        res.status(500).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Sector correlation analysis controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to analyze sector correlations',
        error: error.message
      });
    }
  }

  /**
   * Predict high-risk funds
   */
  async predictHighRiskFunds(req, res) {
    try {
      logger.info('High-risk fund prediction request received');

      const result = await marketAnalyticsEngine.predictHighRiskFunds();

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'High-risk funds predicted successfully',
          data: result.data
        });
      } else {
        res.status(500).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('High-risk fund prediction controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to predict high-risk funds',
        error: error.message
      });
    }
  }

  /**
   * Perform comprehensive market analysis
   */
  async performComprehensiveAnalysis(req, res) {
    try {
      logger.info('Comprehensive market analysis request received');

      const result = await marketAnalyticsEngine.performComprehensiveAnalysis();

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'Comprehensive analysis completed successfully',
          data: result.data
        });
      } else {
        res.status(500).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Comprehensive analysis controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to perform comprehensive analysis',
        error: error.message
      });
    }
  }

  /**
   * Get market analytics dashboard data
   */
  async getMarketAnalyticsDashboard(req, res) {
    try {
      logger.info('Market analytics dashboard request received');

      const { userId } = req.user;

      // Get all market analytics data
      const [
        marketData,
        sentimentData,
        macroData,
        sectorData,
        riskData
      ] = await Promise.all([
        marketAnalyticsEngine.scrapeMarketData(),
        marketAnalyticsEngine.analyzeMarketSentiment(),
        marketAnalyticsEngine.fetchMacroData(),
        marketAnalyticsEngine.analyzeSectorCorrelations(),
        marketAnalyticsEngine.predictHighRiskFunds()
      ]);

      const dashboardData = {
        marketData: marketData.success ? marketData.data : null,
        sentimentData: sentimentData.success ? sentimentData.data : null,
        macroData: macroData.success ? macroData.data : null,
        sectorData: sectorData.success ? sectorData.data : null,
        riskData: riskData.success ? riskData.data : null,
        lastUpdated: new Date().toISOString()
      };

      res.status(200).json({
        success: true,
        message: 'Market analytics dashboard data retrieved successfully',
        data: dashboardData
      });
    } catch (error) {
      logger.error('Market analytics dashboard controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get market analytics dashboard data',
        error: error.message
      });
    }
  }

  /**
   * Get market insights for user portfolio
   */
  async getMarketInsightsForPortfolio(req, res) {
    try {
      logger.info('Market insights for portfolio request received');

      const { userId } = req.user;
      const { portfolioId } = req.params;

      // Get comprehensive analysis
      const analysis = await marketAnalyticsEngine.performComprehensiveAnalysis();

      if (!analysis.success) {
        return res.status(500).json({
          success: false,
          message: 'Failed to get market analysis',
          error: analysis.error
        });
      }

      // Generate portfolio-specific insights
      const insights = {
        marketOutlook: analysis.data.outlook,
        sectorOpportunities: analysis.data.sectorData?.sectorInsights || [],
        riskAlerts: analysis.data.riskData?.riskAlerts || [],
        macroImpact: analysis.data.macroData?.marketImpact || {},
        sentimentTrend: analysis.data.sentimentData?.sentiment || {},
        recommendations: this.generatePortfolioRecommendations(analysis.data),
        timestamp: new Date().toISOString()
      };

      res.status(200).json({
        success: true,
        message: 'Market insights for portfolio retrieved successfully',
        data: insights
      });
    } catch (error) {
      logger.error('Market insights for portfolio controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get market insights for portfolio',
        error: error.message
      });
    }
  }

  /**
   * Get sector performance trends
   */
  async getSectorPerformanceTrends(req, res) {
    try {
      logger.info('Sector performance trends request received');

      const { period } = req.query;
      const analysisPeriod = period || '1m';

      const sectorData = await marketAnalyticsEngine.analyzeSectorCorrelations();

      if (!sectorData.success) {
        return res.status(500).json({
          success: false,
          message: 'Failed to get sector data',
          error: sectorData.error
        });
      }

      const trends = {
        sectorTrends: sectorData.data.sectorTrends,
        correlations: sectorData.data.correlations,
        insights: sectorData.data.sectorInsights,
        period: analysisPeriod,
        timestamp: new Date().toISOString()
      };

      res.status(200).json({
        success: true,
        message: 'Sector performance trends retrieved successfully',
        data: trends
      });
    } catch (error) {
      logger.error('Sector performance trends controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get sector performance trends',
        error: error.message
      });
    }
  }

  /**
   * Get market sentiment trends
   */
  async getMarketSentimentTrends(req, res) {
    try {
      logger.info('Market sentiment trends request received');

      const { period } = req.query;
      const analysisPeriod = period || '1w';

      const sentimentData = await marketAnalyticsEngine.analyzeMarketSentiment(analysisPeriod);

      if (!sentimentData.success) {
        return res.status(500).json({
          success: false,
          message: 'Failed to get sentiment data',
          error: sentimentData.error
        });
      }

      const trends = {
        overallSentiment: sentimentData.data.sentiment.overall,
        categorySentiment: sentimentData.data.sentiment.byCategory,
        timeSentiment: sentimentData.data.sentiment.byTime,
        correlation: sentimentData.data.correlation,
        newsCount: sentimentData.data.newsCount,
        period: analysisPeriod,
        timestamp: new Date().toISOString()
      };

      res.status(200).json({
        success: true,
        message: 'Market sentiment trends retrieved successfully',
        data: trends
      });
    } catch (error) {
      logger.error('Market sentiment trends controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get market sentiment trends',
        error: error.message
      });
    }
  }

  /**
   * Get macroeconomic indicators
   */
  async getMacroeconomicIndicators(req, res) {
    try {
      logger.info('Macroeconomic indicators request received');

      const macroData = await marketAnalyticsEngine.fetchMacroData();

      if (!macroData.success) {
        return res.status(500).json({
          success: false,
          message: 'Failed to get macro data',
          error: macroData.error
        });
      }

      const indicators = {
        gdp: macroData.data.macroData.gdp,
        inflation: macroData.data.macroData.inflation,
        repoRate: macroData.data.macroData.repoRate,
        fiscalDeficit: macroData.data.macroData.fiscalDeficit,
        currentAccountDeficit: macroData.data.macroData.currentAccountDeficit,
        forexReserves: macroData.data.macroData.forexReserves,
        industrialProduction: macroData.data.macroData.industrialProduction,
        tradeBalance: macroData.data.macroData.tradeBalance,
        marketImpact: macroData.data.marketImpact,
        lastUpdated: macroData.data.lastUpdated
      };

      res.status(200).json({
        success: true,
        message: 'Macroeconomic indicators retrieved successfully',
        data: indicators
      });
    } catch (error) {
      logger.error('Macroeconomic indicators controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get macroeconomic indicators',
        error: error.message
      });
    }
  }

  /**
   * Get risk assessment summary
   */
  async getRiskAssessmentSummary(req, res) {
    try {
      logger.info('Risk assessment summary request received');

      const riskData = await marketAnalyticsEngine.predictHighRiskFunds();

      if (!riskData.success) {
        return res.status(500).json({
          success: false,
          message: 'Failed to get risk data',
          error: riskData.error
        });
      }

      const summary = {
        highRiskFunds: riskData.data.highRiskFunds,
        riskAlerts: riskData.data.riskAlerts,
        sectorStress: riskData.data.sectorStress,
        riskThreshold: riskData.data.riskThreshold,
        totalHighRiskFunds: riskData.data.highRiskFunds.length,
        analysisDate: riskData.data.analysisDate
      };

      res.status(200).json({
        success: true,
        message: 'Risk assessment summary retrieved successfully',
        data: summary
      });
    } catch (error) {
      logger.error('Risk assessment summary controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get risk assessment summary',
        error: error.message
      });
    }
  }

  // Helper methods
  generatePortfolioRecommendations(analysisData) {
    try {
      const recommendations = [];

      // Market outlook recommendations
      if (analysisData.outlook?.equity === 'POSITIVE') {
        recommendations.push({
          type: 'MARKET_OPPORTUNITY',
          priority: 'MEDIUM',
          message: 'Market conditions are favorable for equity investments',
          action: 'Consider increasing equity allocation'
        });
      }

      // Sector opportunity recommendations
      if (analysisData.sectorData?.sectorInsights) {
        analysisData.sectorData.sectorInsights.forEach(insight => {
          recommendations.push({
            type: 'SECTOR_OPPORTUNITY',
            priority: 'LOW',
            message: insight.insight,
            action: insight.recommendation
          });
        });
      }

      // Risk alert recommendations
      if (analysisData.riskData?.riskAlerts) {
        analysisData.riskData.riskAlerts.forEach(alert => {
          recommendations.push({
            type: 'RISK_ALERT',
            priority: 'HIGH',
            message: alert.alert,
            action: alert.recommendation
          });
        });
      }

      return recommendations;
    } catch (error) {
      logger.error('Portfolio recommendations generation failed', { error: error.message });
      return [];
    }
  }
}

module.exports = new MarketAnalyticsController(); 