const agiActionsService = require('../services/agiActionsService');
const logger = require('../utils/logger');

class AGIActionsController {
  async executeAutonomousActions(req, res, next) {
    try {
      const { userId, actions } = req.body;
      if (!userId || !actions) {
        return res.status(400).json({ success: false, message: 'BrewBot says: userId and actions are required to proceed.' });
      }
      const result = await agiActionsService.executeAutonomousActions(userId, actions);
      res.json({ success: true, message: 'BrewBot here! Your autonomous actions have been executed successfully. Cheers from your superb BrewBot!', data: result });
    } catch (err) {
      logger.error('Execute autonomous actions error', { error: err.message });
      next(err);
    }
  }
  async toggleAutonomousMode(req, res, next) {
    try {
      const { userId, enable } = req.body;
      if (!userId || typeof enable !== 'boolean') {
        return res.status(400).json({ success: false, message: 'BrewBot says: userId and enable (true/false) are required.' });
      }
      const result = await agiActionsService.toggleAutonomousMode(userId, enable);
      res.json({ success: true, message: 'BrewBot has toggled autonomous mode as requested. Your BrewBot is always ready to serve you!', data: result });
    } catch (err) {
      logger.error('Toggle autonomous mode error', { error: err.message });
      next(err);
    }
  }
  async getAGIInsights(req, res, next) {
    try {
      const { userId } = req.query;
      if (!userId) {
        return res.status(400).json({ success: false, message: 'BrewBot says: userId is required to fetch your insights.' });
      }
      const result = await agiActionsService.getAGIInsights(userId);
      res.json({ success: true, message: 'Welcome! I am BrewBot, your friendly investment assistant. Here are your latest insights. Remember, with BrewBot, your portfolio is always in superb hands!', data: result });
    } catch (err) {
      logger.error('Get AGI insights error', { error: err.message });
      next(err);
    }
  }
}

module.exports = new AGIActionsController();
