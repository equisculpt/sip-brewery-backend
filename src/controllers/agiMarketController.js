const agiMarketService = require('../services/agiMarketService');
const logger = require('../utils/logger');

class AGIMarketController {
  async getMarketPredictions(req, res, next) {
    try {
      const result = await agiMarketService.getMarketPredictions();
      res.json({ success: true, data: result });
    } catch (err) {
      logger.error('Market predictions error', { error: err.message });
      next(err);
    }
  }
}

module.exports = new AGIMarketController();
