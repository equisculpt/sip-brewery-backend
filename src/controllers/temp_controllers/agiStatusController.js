const logger = require('../utils/logger');
const agiEngine = require('../services/agiEngine');

class AGIStatusController {
  /**
   * Get AGI capabilities
   */
  async getAGICapabilities(req, res, next) {
    try {
      const result = await agiEngine.getCapabilities();
      res.json({ success: true, data: result });
    } catch (error) {
      logger.error('Error in getAGICapabilities', { error: error.message });
      next(error);
    }
  }

  /**
   * Get AGI system status
   */
  async getAGIStatus(req, res, next) {
    try {
      const result = await agiEngine.getStatus();
      res.json({ success: true, data: result });
    } catch (error) {
      logger.error('Error in getAGIStatus', { error: error.message });
      next(error);
    }
  }
}

module.exports = new AGIStatusController();
