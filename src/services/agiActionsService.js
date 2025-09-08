/**
 * AGIActionsService - Production-ready implementation for AGI actions and insights.
 * Handles autonomous actions, toggling autonomous mode, and retrieving AGI insights securely.
 */
const { User, UserBehavior, AGIInsight } = require('../models');
const logger = require('../utils/logger');

class AGIActionsService {
  /**
   * Execute autonomous actions for a user.
   * @param {string} userId - Authenticated user's supabaseId
   * @param {Array<Object>} actions - Array of action objects (validated by controller)
   * @returns {Promise<Object>} Result summary
   * @throws {Error} On user not found or DB error
   */
  async executeAutonomousActions(userId, actions) {
    try {
      // Find user and UserBehavior doc
      const user = await User.findOne({ supabaseId: userId });
      if (!user) {
        logger.warn('User not found for autonomous actions', { userId });
        throw new Error('User not found');
      }
      let behavior = await UserBehavior.findOne({ userId: user._id });
      if (!behavior) {
        behavior = new UserBehavior({ userId: user._id, actions: [] });
      }
      // Add actions to UserBehavior
      for (const actionObj of actions) {
        behavior.actions.push({
          action: actionObj.action,
          timestamp: actionObj.timestamp ? new Date(actionObj.timestamp) : new Date(),
          context: actionObj.context || {},
          metadata: actionObj.metadata || {},
        });
      }
      await behavior.save();
      logger.info('Autonomous actions executed and saved', { userId, actionsCount: actions.length });
      return { executed: true, actionsCount: actions.length };
    } catch (err) {
      logger.error('Failed to execute autonomous actions', { userId, error: err.message });
      throw new Error('Failed to execute autonomous actions. Please try again.');
    }
  }

  /**
   * Toggle autonomous mode for a user.
   * @param {string} userId - Authenticated user's supabaseId
   * @param {boolean} enable - Enable or disable autonomous mode
   * @returns {Promise<Object>} Result summary
   * @throws {Error} On user not found or DB error
   */
  async toggleAutonomousMode(userId, enable) {
    try {
      // Find user and UserBehavior doc
      const user = await User.findOne({ supabaseId: userId });
      if (!user) {
        logger.warn('User not found for toggle autonomous mode', { userId });
        throw new Error('User not found');
      }
      let behavior = await UserBehavior.findOne({ userId: user._id });
      if (!behavior) {
        behavior = new UserBehavior({ userId: user._id });
      }
      // Update autonomousMode
      behavior.aiInteraction = behavior.aiInteraction || {};
      behavior.aiInteraction.autonomousMode = behavior.aiInteraction.autonomousMode || {};
      behavior.aiInteraction.autonomousMode.enabled = !!enable;
      behavior.aiInteraction.autonomousMode.lastToggle = new Date();
      await behavior.save();
      logger.info('Autonomous mode toggled', { userId, enabled: !!enable });
      return { toggled: true, enabled: !!enable };
    } catch (err) {
      logger.error('Failed to toggle autonomous mode', { userId, error: err.message });
      throw new Error('Failed to toggle autonomous mode. Please try again.');
    }
  }

  /**
   * Retrieve active AGI insights for a user.
   * @param {string} userId - Authenticated user's supabaseId
   * @returns {Promise<Object>} Insights array
   * @throws {Error} On user not found or DB error
   */
  async getAGIInsights(userId) {
    try {
      const user = await User.findOne({ supabaseId: userId });
      if (!user) {
        logger.warn('User not found for getAGIInsights', { userId });
        throw new Error('User not found');
      }
      // Use static method if available, else filter manually
      let insights;
      if (typeof AGIInsight.findActiveInsights === 'function') {
        insights = await AGIInsight.findActiveInsights(user._id);
      } else {
        insights = await AGIInsight.find({
          userId: user._id,
          status: 'active',
          expiresAt: { $gt: new Date() },
        }).sort({ priority: -1, createdAt: -1 });
      }
      logger.info('Retrieved AGI insights', { userId, insightsCount: insights.length });
      return { insights };
    } catch (err) {
      logger.error('Failed to retrieve AGI insights', { userId, error: err.message });
      throw new Error('Failed to retrieve AGI insights. Please try again.');
    }
  }
}

module.exports = new AGIActionsService();
