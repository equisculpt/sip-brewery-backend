const logger = require('../utils/logger');
const agiEngine = require('../services/agiEngine');

class AGIFeedbackController {
  /**
   * Submit feedback for AGI improvement
   */
  async submitFeedback(req, res, next) {
    try {
      const { userId, feedback } = req.body;
      if (!userId || !feedback) {
        return res.status(400).json({ success: false, message: 'userId and feedback are required' });
      }
      const result = await agiEngine.submitFeedback(userId, feedback);
      if (!result.success) {
        return res.status(500).json(result);
      }
      res.json(result);
    } catch (error) {
      logger.error('Error in submitFeedback', { error: error.message });
      next(error);
    }
  }
}

module.exports = new AGIFeedbackController();
