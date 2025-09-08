const agiBehaviorService = require('../services/agiBehaviorService');
const logger = require('../utils/logger');

class AGIBehaviorController {
  async trackUserBehavior(req, res, next) {
    try {
      const { userId, action, context } = req.body;
      if (!userId || !action) {
        return res.status(400).json({ success: false, message: 'userId and action are required' });
      }
      const result = await agiBehaviorService.trackBehavior(userId, action, context);
      res.json({ success: true, data: result });
    } catch (err) {
      logger.error('Behavior tracking error', { userId: req.body.userId, error: err.message });
      next(err);
    }
  }
  async learnFromMarketEvents(req, res, next) {
    try {
      const { userId, event, reaction } = req.body;
      if (!userId || !event) {
        return res.status(400).json({ success: false, message: 'userId and event are required' });
      }
      const result = await agiBehaviorService.learnFromMarketEvents(userId, event, reaction);
      res.json({ success: true, data: result });
    } catch (err) {
      logger.error('Learning from market events error', { userId: req.body.userId, error: err.message });
      next(err);
    }
  }
}

module.exports = new AGIBehaviorController();
