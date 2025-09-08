const smartSipService = require('../services/smartSipService');
const { successResponse, errorResponse } = require('../utils/response');
const logger = require('../utils/logger');
const marketScoreService = require('../services/marketScoreService');

class SmartSipController {
  /**
   * Start a new SIP (static or smart)
   */
  async startSIP(req, res) {
    try {
      if (!req.userId) {
        return errorResponse(res, 'Authentication required', null, 401);
      }

      const { sipType, averageSip, fundSelection, sipDay, preferences } = req.body;

      // Validate required fields
      if (!sipType || !averageSip || !fundSelection) {
        return errorResponse(res, 'Missing required fields: sipType, averageSip, fundSelection', null, 400);
      }

      // Validate SIP type
      if (!['STATIC', 'SMART'].includes(sipType)) {
        return errorResponse(res, 'Invalid SIP type. Must be STATIC or SMART', null, 400);
      }

      // Validate average SIP amount
      if (averageSip < 1000 || averageSip > 100000) {
        return errorResponse(res, 'Average SIP amount must be between ₹1,000 and ₹1,00,000', null, 400);
      }

      logger.info('Starting SIP', { userId: req.userId, sipType, averageSip });

      const result = await smartSipService.startSIP(req.userId, {
        sipType,
        averageSip,
        fundSelection,
        sipDay,
        preferences
      });

      return successResponse(res, result.message, result);

    } catch (error) {
      logger.error('Error starting SIP:', error);
      return errorResponse(res, 'Failed to start SIP', { error: error.message }, 500);
    }
  }

  /**
   * Get current SIP recommendation
   */
  async getSIPRecommendation(req, res) {
    try {
      if (!req.userId) {
        return errorResponse(res, 'Authentication required', null, 401);
      }

      logger.info('Getting SIP recommendation', { userId: req.userId });

      const recommendation = await smartSipService.getSIPRecommendation(req.userId);

      return successResponse(res, 'SIP recommendation retrieved successfully', recommendation);

    } catch (error) {
      logger.error('Error getting SIP recommendation:', error);
      return errorResponse(res, 'Failed to get SIP recommendation', { error: error.message }, 500);
    }
  }

  /**
   * Get user's SIP details
   */
  async getSIPDetails(req, res) {
    try {
      if (!req.userId) {
        return errorResponse(res, 'Authentication required', null, 401);
      }

      logger.info('Getting SIP details', { userId: req.userId });

      const details = await smartSipService.getSIPDetails(req.userId);

      if (!details) {
        return errorResponse(res, 'No active SIP found', null, 404);
      }

      return successResponse(res, 'SIP details retrieved successfully', details);

    } catch (error) {
      logger.error('Error getting SIP details:', error);
      return errorResponse(res, 'Failed to get SIP details', { error: error.message }, 500);
    }
  }

  /**
   * Update SIP preferences
   */
  async updateSIPPreferences(req, res) {
    try {
      if (!req.userId) {
        return errorResponse(res, 'Authentication required', null, 401);
      }

      const { riskTolerance, marketTiming, aiEnabled, notifications } = req.body;

      logger.info('Updating SIP preferences', { userId: req.userId });

      const result = await smartSipService.updateSIPPreferences(req.userId, {
        riskTolerance,
        marketTiming,
        aiEnabled,
        notifications
      });

      return successResponse(res, result.message, result);

    } catch (error) {
      logger.error('Error updating SIP preferences:', error);
      return errorResponse(res, 'Failed to update SIP preferences', { error: error.message }, 500);
    }
  }

  /**
   * Update SIP status (pause/resume)
   */
  async updateSIPStatus(req, res) {
    try {
      if (!req.userId) {
        return errorResponse(res, 'Authentication required', null, 401);
      }

      const { status } = req.body;

      if (!['ACTIVE', 'PAUSED', 'STOPPED'].includes(status)) {
        return errorResponse(res, 'Invalid status. Must be ACTIVE, PAUSED, or STOPPED', null, 400);
      }

      logger.info('Updating SIP status', { userId: req.userId, status });

      const result = await smartSipService.updateSIPStatus(req.userId, status);

      return successResponse(res, result.message, result);

    } catch (error) {
      logger.error('Error updating SIP status:', error);
      return errorResponse(res, 'Failed to update SIP status', { error: error.message }, 500);
    }
  }

  /**
   * Execute SIP manually
   */
  async executeSIP(req, res) {
    try {
      if (!req.userId) {
        return errorResponse(res, 'Authentication required', null, 401);
      }

      logger.info('Executing SIP manually', { userId: req.userId });

      const result = await smartSipService.executeSIP(req.userId);

      if (!result.success) {
        // Always include data field
        return errorResponse(res, result.message, { nextSIPDate: result.nextSIPDate }, 400);
      }

      return successResponse(res, result.message, result);

    } catch (error) {
      logger.error('Error executing SIP:', error);
      // Always include data field
      return errorResponse(res, 'Failed to execute SIP', { error: error.message }, 500);
    }
  }

  /**
   * Get SIP analytics
   */
  async getSIPAnalytics(req, res) {
    try {
      if (!req.userId) {
        return errorResponse(res, 'Authentication required', null, 401);
      }

      logger.info('Getting SIP analytics', { userId: req.userId });

      const analytics = await smartSipService.getSIPAnalytics(req.userId);

      if (!analytics) {
        return errorResponse(res, 'No SIP data found', null, 404);
      }

      return successResponse(res, 'SIP analytics retrieved successfully', analytics);

    } catch (error) {
      logger.error('Error getting SIP analytics:', error);
      return errorResponse(res, 'Failed to get SIP analytics', { error: error.message }, 500);
    }
  }

  /**
   * Get SIP history
   */
  async getSIPHistory(req, res) {
    try {
      if (!req.userId) {
        return errorResponse(res, 'Authentication required', null, 401);
      }

      const { limit = 10 } = req.query;

      logger.info('Getting SIP history', { userId: req.userId, limit });

      const history = await smartSipService.getSIPHistory(req.userId, parseInt(limit));

      return successResponse(res, 'SIP history retrieved successfully', history);

    } catch (error) {
      logger.error('Error getting SIP history:', error);
      return errorResponse(res, 'Failed to get SIP history', { error: error.message }, 500);
    }
  }

  /**
   * Get market analysis (for frontend display)
   */
  async getMarketAnalysis(req, res) {
    try {
      if (!req.userId) {
        return errorResponse(res, 'Authentication required', null, 401);
      }

      logger.info('Getting market analysis', { userId: req.userId });

      let analysis;
      try {
        analysis = await marketScoreService.calculateMarketScore();
      } catch (err) {
        logger.error('Error in marketScoreService:', err);
        return errorResponse(res, 'Failed to get market analysis', { error: err.message }, 500);
      }

      return successResponse(res, 'Market analysis retrieved successfully', analysis);

    } catch (error) {
      logger.error('Error getting market analysis:', error);
      return errorResponse(res, 'Failed to get market analysis', { error: error.message }, 500);
    }
  }

  /**
   * Get all user SIPs (admin endpoint)
   */
  async getAllUserSIPs(req, res) {
    try {
      if (!req.userId) {
        return errorResponse(res, 'Authentication required', null, 401);
      }

      logger.info('Getting all user SIPs', { userId: req.userId });

      const sips = await smartSipService.getAllActiveSIPs();

      return successResponse(res, 'All SIPs retrieved successfully', sips);

    } catch (error) {
      logger.error('Error getting all user SIPs:', error);
      return errorResponse(res, 'Failed to get all user SIPs', { error: error.message }, 500);
    }
  }
}

module.exports = new SmartSipController(); 