const logger = require('../utils/logger');
const agiEngine = require('../services/agiEngine');
const { validateObjectId } = require('../middleware/validation');

class AGIInsightsController {
  /**
   * Get weekly AGI insights for user
   */
  async getWeeklyInsights(req, res, next) {
    try {
      const { userId } = req.params;
      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }
      const result = await agiEngine.generateWeeklyInsights(userId);
      if (!result.success) {
        return res.status(404).json(result);
      }
      res.json(result);
    } catch (error) {
      logger.error('Error in getWeeklyInsights', { error: error.message });
      next(error);
    }
  }
}

module.exports = new AGIInsightsController();
