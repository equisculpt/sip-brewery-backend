const leaderboardService = require('../services/leaderboardService');
const response = require('../utils/response');
const logger = require('../utils/logger');

class LeaderboardController {
  /**
   * Get leaderboard for a specific duration
   */
  async getLeaderboard(req, res) {
    try {
      const { duration } = req.params;
      
      // Validate duration
      if (!['1M', '3M', '6M', '1Y', '3Y'].includes(duration)) {
        return response.error(res, 'Invalid duration. Must be 1M, 3M, 6M, 1Y, or 3Y', 400);
      }

      logger.info('Getting leaderboard', { duration });

      const leaderboard = await leaderboardService.getLeaderboard(duration);

      if (!leaderboard) {
        return response.error(res, 'Leaderboard not found for this duration', 404);
      }

      return response.success(res, leaderboard, 'Leaderboard retrieved successfully');

    } catch (error) {
      logger.error('Error getting leaderboard:', error);
      return response.error(res, 'Failed to get leaderboard', 500, error);
    }
  }

  /**
   * Get all leaderboards
   */
  async getAllLeaderboards(req, res) {
    try {
      logger.info('Getting all leaderboards');

      const durations = ['1M', '3M', '6M', '1Y', '3Y'];
      const leaderboards = {};

      for (const duration of durations) {
        try {
          const leaderboard = await leaderboardService.getLeaderboard(duration);
          if (leaderboard) {
            leaderboards[duration] = leaderboard;
          }
        } catch (error) {
          logger.error(`Error getting leaderboard for ${duration}:`, error);
        }
      }

      return response.success(res, leaderboards, 'All leaderboards retrieved successfully');

    } catch (error) {
      logger.error('Error getting all leaderboards:', error);
      return response.error(res, 'Failed to get leaderboards', 500, error);
    }
  }
}

module.exports = new LeaderboardController();
