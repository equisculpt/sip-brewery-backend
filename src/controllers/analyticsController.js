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
}

module.exports = new AnalyticsController();
