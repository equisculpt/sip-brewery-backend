const logger = require('../utils/logger');
const agiEngine = require('../services/agiEngine');
const { validateObjectId } = require('../middleware/validation');

class AGIRecommendationsController {
  /**
   * Get personalized investment recommendations
   */
  async getPersonalizedRecommendations(req, res, next) {
    try {
      const { userId } = req.params;
      const { type } = req.query;
      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }
      const result = await agiEngine.generatePersonalizedRecommendations(userId, type);
      if (!result.success) {
        return res.status(404).json(result);
      }
      res.json(result);
    } catch (error) {
      logger.error('Error in getPersonalizedRecommendations', { error: error.message });
      next(error);
    }
  }
}

module.exports = new AGIRecommendationsController();
