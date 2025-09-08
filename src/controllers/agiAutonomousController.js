const agiAutonomousService = require('../services/agiAutonomousService');
const logger = require('../utils/logger');

class AGIAutonomousController {
  async initializeAGI(req, res, next) {
    try {
      const { userId } = req.body;
      if (!userId) {
        return res.status(400).json({ success: false, message: 'userId is required' });
      }
      const result = await agiAutonomousService.initializeAGI(userId);
      res.json({ success: true, data: result });
    } catch (err) {
      logger.error('Initialize AGI error', { error: err.message });
      next(err);
    }
  }
  async autonomousPortfolioManagement(req, res, next) {
    try {
      const { userId, portfolio } = req.body;
      if (!userId || !portfolio) {
        return res.status(400).json({ success: false, message: 'userId and portfolio are required' });
      }
      const result = await agiAutonomousService.autonomousPortfolioManagement(userId, portfolio);
      res.json({ success: true, data: result });
    } catch (err) {
      logger.error('Autonomous management error', { error: err.message });
      next(err);
    }
  }
}

module.exports = new AGIAutonomousController();
