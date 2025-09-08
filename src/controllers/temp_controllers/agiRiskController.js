const agiRiskService = require('../services/agiRiskService');
const logger = require('../utils/logger');

class AGIRiskController {
  async intelligentRiskManagement(req, res, next) {
    try {
      const { portfolio } = req.body;
      if (!portfolio) {
        return res.status(400).json({ success: false, message: 'portfolio is required' });
      }
      const result = await agiRiskService.intelligentRiskManagement(portfolio);
      res.json({ success: true, data: result });
    } catch (err) {
      logger.error('Risk management error', { error: err.message });
      next(err);
    }
  }
}

module.exports = new AGIRiskController();
