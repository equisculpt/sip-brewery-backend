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
}

module.exports = new MarketAnalyticsController();
