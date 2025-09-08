/**
 * AGIActionsController handles endpoints related to AGI-driven autonomous actions and insights.
 * Security: userId is now taken from req.user (populated by authentication middleware) for all endpoints.
 * @module controllers/agiActionsController
 */
const agiActionsService = require('../services/agiActionsService');
const logger = require('../utils/logger');

/**
 * Helper to validate action objects (customize as per schema)
 * @param {any} action
 * @returns {boolean}
 */
function isValidAction(action) {
  // Example: check action is an object with a type property (customize as needed)
  return action && typeof action === 'object' && typeof action.type === 'string';
}

class AGIActionsController {
  /**
   * Execute a list of autonomous actions for the authenticated user.
   * @route POST /agi/execute-autonomous-actions
   * @param {Array} req.body.actions - Array of actions to execute.
   * @returns {Object} 200 - Success message and result data
   * @returns {Object} 400 - Validation error
   * @returns {Object} 500 - Internal server error
   * Security: userId is taken from req.user (populated by authentication middleware)
   */
  async executeAutonomousActions(req, res, next) {
    try {
      const userId = req.user && req.user.supabaseId;
      const { actions } = req.body;
      if (!userId) {
        return res.status(401).json({ success: false, message: 'Authentication required.' });
      }
      if (!actions || !Array.isArray(actions) || actions.length === 0) {
        return res.status(400).json({ success: false, message: 'BrewBot says: actions (non-empty array) are required to proceed.' });
      }
      // Validate each action (customize as needed)
      if (!actions.every(isValidAction)) {
        return res.status(400).json({ success: false, message: 'BrewBot says: Each action must have a valid structure.' });
      }
      const result = await agiActionsService.executeAutonomousActions(userId, actions);
      logger.info('Autonomous actions executed', { userId, actionsCount: actions.length });
      res.json({ success: true, message: 'BrewBot here! Your autonomous actions have been executed successfully. Cheers from your superb BrewBot!', data: result });
    } catch (err) {
      logger.error('Execute autonomous actions error', { error: err.message });
      next(err);
    }
  }

  /**
   * Toggle the autonomous mode for the authenticated user.
   * @route POST /agi/toggle-autonomous-mode
   * @param {boolean} req.body.enable - Whether to enable or disable autonomous mode.
   * @returns {Object} 200 - Success message and result data
   * @returns {Object} 400 - Validation error
   * @returns {Object} 500 - Internal server error
   * Security: userId is taken from req.user (populated by authentication middleware)
   */
  async toggleAutonomousMode(req, res, next) {
    try {
      const userId = req.user && req.user.supabaseId;
      const { enable } = req.body;
      if (!userId) {
        return res.status(401).json({ success: false, message: 'Authentication required.' });
      }
      if (typeof enable !== 'boolean') {
        return res.status(400).json({ success: false, message: 'BrewBot says: enable (true/false) is required.' });
      }
      const result = await agiActionsService.toggleAutonomousMode(userId, enable);
      logger.info('Autonomous mode toggled', { userId, enable });
      res.json({ success: true, message: 'BrewBot has toggled autonomous mode as requested. Your BrewBot is always ready to serve you!', data: result });
    } catch (err) {
      logger.error('Toggle autonomous mode error', { error: err.message });
      next(err);
    }
  }

  /**
   * Get AGI-generated insights for the authenticated user.
   * @route GET /agi/insights
   * @returns {Object} 200 - Success message and insights data
   * @returns {Object} 400 - Validation error
   * @returns {Object} 500 - Internal server error
   * Security: userId is taken from req.user (populated by authentication middleware)
   */
  async getAGIInsights(req, res, next) {
    try {
      const userId = req.user && req.user.supabaseId;
      if (!userId) {
        return res.status(401).json({ success: false, message: 'Authentication required.' });
      }
      const result = await agiActionsService.getAGIInsights(userId);
      logger.info('AGI insights retrieved', { userId });
      res.json({ success: true, message: 'Welcome! I am BrewBot, your friendly investment assistant. Here are your latest insights. Remember, with BrewBot, your portfolio is always in superb hands!', data: result });
    } catch (err) {
      logger.error('Get AGI insights error', { error: err.message });
      next(err);
    }
  }
}

module.exports = new AGIActionsController();
