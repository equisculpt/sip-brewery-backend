const logger = require('../utils/logger');
const featureExtractor = require('../utils/featureExtractor');

class AGIAnalyticsController {
  /**
   * Upload and analyze user portfolio
   */
  async portfolioAnalytics(req, res, next) {
    try {
      const { portfolio, mfAnalytics = {}, stockAnalytics = {} } = req.body;
      if (!portfolio || !Array.isArray(portfolio)) {
        return res.status(400).json({ success: false, message: 'portfolio (array) is required' });
      }
      const result = featureExtractor.extractPortfolioFeatures(portfolio, mfAnalytics, stockAnalytics);
      res.json({ success: true, data: result });
    } catch (error) {
      logger.error('Error in portfolioAnalytics', { error: error.message });
      next(error);
    }
  }

  /**
   * Get mutual fund analytics
   */
  async mutualFundAnalytics(req, res, next) {
    try {
      const { schemeDetail, navHistory, returns } = req.body;
      if (!schemeDetail || !navHistory || !returns) {
        return res.status(400).json({ success: false, message: 'schemeDetail, navHistory, and returns are required' });
      }
      const result = featureExtractor.extractMfFeatures({ schemeDetail, navHistory, returns });
      res.json({ success: true, data: result });
    } catch (error) {
      logger.error('Error in mutualFundAnalytics', { error: error.message });
      next(error);
    }
  }

  /**
   * Compare stocks vs mutual funds
   */
  async compareStockVsFund(req, res, next) {
    try {
      const { stockAnalytics = {}, mfAnalytics = {} } = req.body;
      const result = featureExtractor.compareStockVsMf(stockAnalytics, mfAnalytics);
      res.json({ success: true, data: result });
    } catch (error) {
      logger.error('Error in compareStockVsFund', { error: error.message });
      next(error);
    }
  }
}

module.exports = new AGIAnalyticsController();
