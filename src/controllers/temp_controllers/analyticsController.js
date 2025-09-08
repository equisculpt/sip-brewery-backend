const portfolioAnalyticsService = require('../services/portfolioAnalyticsService');
const chartDataService = require('../services/chartDataService');
const investmentCalculatorService = require('../services/investmentCalculatorService');
const riskProfilingService = require('../services/riskProfilingService');
const navHistoryService = require('../services/navHistoryService');
const taxCalculationService = require('../services/taxCalculationService');
const logger = require('../utils/logger');

class AnalyticsController {
  /**
   * Get comprehensive performance analytics with chart data
   */
  async getPerformanceAnalytics(req, res) {
    try {
      const { userId } = req.user;
      const { period = '1y', fundCode, includeChartData = true } = req.query;

      logger.info(`Getting performance analytics for user: ${userId}, period: ${period}`);

      const analytics = await portfolioAnalyticsService.calculatePortfolioAnalytics({
        userId,
        period,
        fundCode,
        includeChartData
      });

      res.json({
        success: true,
        data: analytics
      });
    } catch (error) {
      logger.error('Error getting performance analytics:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to get performance analytics',
        error: error.message
      });
    }
  }

  /**
   * Get chart data for various chart types
   */
  async getChartData(req, res) {
    try {
      const { userId } = req.user;
      const { chartType, period = '1y', fundCode, options = {} } = req.body;

      logger.info(`Getting chart data for user: ${userId}, chartType: ${chartType}`);

      const chartData = await chartDataService.generateChartData({
        userId,
        chartType,
        period,
        fundCode,
        options
      });

      res.json({
        success: true,
        data: chartData
      });
    } catch (error) {
      logger.error('Error getting chart data:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to get chart data',
        error: error.message
      });
    }
  }

  /**
   * Calculate SIP future value and projections
   */
  async calculateSIPProjections(req, res) {
    try {
      const { 
        monthlyAmount, 
        duration, 
        expectedReturn, 
        fundCode,
        startDate = new Date(),
        includeInflation = true,
        includeTaxes = true
      } = req.body;

      logger.info(`Calculating SIP projections for amount: ${monthlyAmount}, duration: ${duration}`);

      const projections = await investmentCalculatorService.calculateSIPProjections({
        monthlyAmount,
        duration,
        expectedReturn,
        fundCode,
        startDate,
        includeInflation,
        includeTaxes
      });

      res.json({
        success: true,
        data: projections
      });
    } catch (error) {
      logger.error('Error calculating SIP projections:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to calculate SIP projections',
        error: error.message
      });
    }
  }

  /**
   * Calculate goal-based investment requirements
   */
  async calculateGoalBasedInvestment(req, res) {
    try {
      const { 
        goalAmount, 
        targetDate, 
        currentSavings = 0,
        riskProfile = 'moderate',
        includeInflation = true
      } = req.body;

      logger.info(`Calculating goal-based investment for amount: ${goalAmount}, target: ${targetDate}`);

      const investmentPlan = await investmentCalculatorService.calculateGoalBasedInvestment({
        goalAmount,
        targetDate,
        currentSavings,
        riskProfile,
        includeInflation
      });

      res.json({
        success: true,
        data: investmentPlan
      });
    } catch (error) {
      logger.error('Error calculating goal-based investment:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to calculate goal-based investment',
        error: error.message
      });
    }
  }

  /**
   * Get risk profiling and assessment
   */
  async getRiskProfiling(req, res) {
    try {
      const { userId } = req.user;
      const { includePortfolioRisk = true, includeMarketRisk = true } = req.query;

      logger.info(`Getting risk profiling for user: ${userId}`);

      const riskProfile = await riskProfilingService.getComprehensiveRiskProfile({
        userId,
        includePortfolioRisk,
        includeMarketRisk
      });

      res.json({
        success: true,
        data: riskProfile
      });
    } catch (error) {
      logger.error('Error getting risk profiling:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to get risk profiling',
        error: error.message
      });
    }
  }

  /**
   * Get NAV history with calculations
   */
  async getNAVHistory(req, res) {
    try {
      const { fundCode } = req.params;
      const { period = '1y', includeCalculations = true } = req.query;

      logger.info(`Getting NAV history for fund: ${fundCode}, period: ${period}`);

      const navHistory = await navHistoryService.getNAVHistory({
        fundCode,
        period,
        includeCalculations
      });

      res.json({
        success: true,
        data: navHistory
      });
    } catch (error) {
      logger.error('Error getting NAV history:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to get NAV history',
        error: error.message
      });
    }
  }

  /**
   * Calculate tax implications and optimization
   */
  async getTaxCalculations(req, res) {
    try {
      const { userId } = req.user;
      const { financialYear, includeOptimization = true } = req.query;

      logger.info(`Getting tax calculations for user: ${userId}, FY: ${financialYear}`);

      const taxCalculations = await taxCalculationService.getTaxCalculations({
        userId,
        financialYear,
        includeOptimization
      });

      res.json({
        success: true,
        data: taxCalculations
      });
    } catch (error) {
      logger.error('Error getting tax calculations:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to get tax calculations',
        error: error.message
      });
    }
  }

  /**
   * Get XIRR analytics
   */
  async getXIRRAnalytics(req, res) {
    try {
      const { userId } = req.user;
      const { timeframe = '1y', fundCode } = req.query;

      logger.info(`Getting XIRR analytics for user: ${userId}, timeframe: ${timeframe}`);

      const xirrAnalytics = await portfolioAnalyticsService.calculateXIRR(userId, timeframe);

      res.json({
        success: true,
        data: xirrAnalytics
      });
    } catch (error) {
      logger.error('Error getting XIRR analytics:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to get XIRR analytics',
        error: error.message
      });
    }
  }

  /**
   * Get portfolio comparison analytics
   */
  async getPortfolioComparison(req, res) {
    try {
      const { userId } = req.user;
      const { benchmark = 'NIFTY50', period = '1y' } = req.query;

      logger.info(`Getting portfolio comparison for user: ${userId}, benchmark: ${benchmark}`);

      const comparison = await portfolioAnalyticsService.getPortfolioComparison(userId);

      res.json({
        success: true,
        data: comparison
      });
    } catch (error) {
      logger.error('Error getting portfolio comparison:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to get portfolio comparison',
        error: error.message
      });
    }
  }

  /**
   * Get comprehensive dashboard analytics
   */
  async getDashboardAnalytics(req, res) {
    try {
      const { userId } = req.user;
      const { includeCharts = true, includeRecommendations = true } = req.query;

      logger.info(`Getting dashboard analytics for user: ${userId}`);

      const dashboardData = await portfolioAnalyticsService.getPortfolioSummary(userId);

      // Add chart data if requested
      if (includeCharts) {
        dashboardData.charts = await chartDataService.generateDashboardCharts(userId);
      }

      // Add recommendations if requested
      if (includeRecommendations) {
        dashboardData.recommendations = await portfolioAnalyticsService.generateRecommendations({
          userId,
          type: 'dashboard'
        });
      }

      res.json({
        success: true,
        data: dashboardData
      });
    } catch (error) {
      logger.error('Error getting dashboard analytics:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to get dashboard analytics',
        error: error.message
      });
    }
  }

  /**
   * Get platform analytics (admin)
   */
  async getPlatformAnalytics(req, res) {
    try {
      const { period = '30d', type = 'overview' } = req.query;

      logger.info(`Getting platform analytics, period: ${period}, type: ${type}`);

      const platformAnalytics = await portfolioAnalyticsService.getPlatformAnalytics({
        period,
        type
      });

      res.json({
        success: true,
        data: platformAnalytics
      });
    } catch (error) {
      logger.error('Error getting platform analytics:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to get platform analytics',
        error: error.message
      });
    }
  }

  /**
   * Get regional analytics
   */
  async getRegionalAnalytics(req, res) {
    try {
      const { region, period = '30d' } = req.query;

      logger.info(`Getting regional analytics for region: ${region}, period: ${period}`);

      const regionalAnalytics = await portfolioAnalyticsService.getRegionalAnalytics({
        region,
        period
      });

      res.json({
        success: true,
        data: regionalAnalytics
      });
    } catch (error) {
      logger.error('Error getting regional analytics:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to get regional analytics',
        error: error.message
      });
    }
  }

  /**
   * Get agent analytics
   */
  async getAgentAnalytics(req, res) {
    try {
      const { agentId, period = '30d' } = req.query;

      logger.info(`Getting agent analytics for agent: ${agentId}, period: ${period}`);

      const agentAnalytics = await portfolioAnalyticsService.getAgentAnalytics({
        agentId,
        period
      });

      res.json({
        success: true,
        data: agentAnalytics
      });
    } catch (error) {
      logger.error('Error getting agent analytics:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to get agent analytics',
        error: error.message
      });
    }
  }

  /**
   * Comprehensive fund comparison with detailed analysis and ratings
   */
  async compareFunds(req, res) {
    try {
      const { 
        fundCodes, 
        category = null, 
        period = '1y',
        investmentAmount = 100000,
        includeRatings = true,
        includeRecommendations = true
      } = req.body;

      // Validate input
      if (!fundCodes || !Array.isArray(fundCodes) || fundCodes.length < 2 || fundCodes.length > 5) {
        return res.status(400).json({
          success: false,
          message: 'Please provide 2-5 fund codes for comparison'
        });
      }

      logger.info(`Comparing funds: ${fundCodes.join(', ')}, category: ${category}, period: ${period}`);

      const comparison = await portfolioAnalyticsService.compareFunds({
        fundCodes,
        category,
        period,
        investmentAmount,
        includeRatings,
        includeRecommendations
      });

      res.json({
        success: true,
        data: comparison
      });
    } catch (error) {
      logger.error('Error comparing funds:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to compare funds',
        error: error.message
      });
    }
  }
}

module.exports = new AnalyticsController(); 