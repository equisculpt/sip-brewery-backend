const agiBehavioralService = require('../services/agiBehavioralService');
const logger = require('../utils/logger');

class AGIBehavioralController {
  async generateBehavioralNudges(req, res, next) {
    try {
      const { userId, context } = req.body;
      if (!userId || !context) {
        return res.status(400).json({ success: false, message: 'userId and context are required' });
      }
      const result = await agiBehavioralService.generateBehavioralNudges(userId, context);
      res.json({ success: true, data: result });
    } catch (err) {
      logger.error('Behavioral nudges error', { error: err.message });
      next(err);
    }
  }
}

module.exports = new AGIBehavioralController();
