const benchmarkService = require('../services/benchmarkService');
const aiService = require('../services/aiService');
const response = require('../utils/response');
const logger = require('../utils/logger');

class BenchmarkController {
  /**
   * Get benchmark data for a specific date range, with optional limit for latest N records
   */
  async getBenchmarkData(req, res) {
    try {
      const { indexId } = req.params;
      const { from, to, limit } = req.query;

      logger.info('Fetching benchmark data', { indexId, from, to, limit });

      const data = await benchmarkService.getBenchmarkData(indexId, from, to);

      // If limit is specified, return only the latest N records
      let resultData = data.data;
      if (limit) {
        resultData = resultData.slice(-parseInt(limit));
      }

      return response.success(res, {
        ...data,
        data: resultData
      }, 'Benchmark data retrieved successfully');

    } catch (error) {
      logger.error('Error fetching benchmark data', { error: error.message });
      return response.error(res, 'Failed to fetch benchmark data', 500, error);
    }
  }

  /**
   * Get real-time market status and indices
   */
  async getMarketStatus(req, res) {
    try {
      logger.info('Fetching market status');

      const marketStatus = await benchmarkService.getMarketStatus();

      return response.success(res, marketStatus, 'Market status retrieved successfully');

    } catch (error) {
      logger.error('Error fetching market status', { error: error.message });
      return response.error(res, 'Failed to fetch market status', 500, error);
    }
  }

  /**
   * Get gainers and losers for a specific index
   */
  async getGainersAndLosers(req, res) {
    try {
      const { indexName = 'NIFTY 50' } = req.query;

      logger.info('Fetching gainers and losers', { indexName });
      // ...rest of implementation
    } catch (error) {
      logger.error('Error fetching gainers and losers', { error: error.message });
      return response.error(res, 'Failed to fetch gainers and losers', 500, error);
    }
  }
}

module.exports = new BenchmarkController();
