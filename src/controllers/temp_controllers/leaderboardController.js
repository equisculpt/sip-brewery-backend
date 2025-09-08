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

  /**
   * Copy portfolio
   */
  async copyPortfolio(req, res) {
    try {
      if (!req.userId) {
        return response.error(res, 'Authentication required', 401);
      }

      const { sourceSecretCode, investmentType, averageSip } = req.body;

      // Validate required fields
      if (!sourceSecretCode || !investmentType) {
        return response.error(res, 'Missing required fields: sourceSecretCode, investmentType', 400);
      }

      // Validate investment type
      if (!['SIP', 'LUMPSUM'].includes(investmentType)) {
        return response.error(res, 'Invalid investment type. Must be SIP or LUMPSUM', 400);
      }

      // Validate average SIP amount for SIP investment
      if (investmentType === 'SIP') {
        if (!averageSip || averageSip < 1000 || averageSip > 100000) {
          return response.error(res, 'Average SIP amount must be between ₹1,000 and ₹1,00,000', 400);
        }
      }

      logger.info('Copying portfolio', { 
        userId: req.userId, 
        sourceSecretCode, 
        investmentType, 
        averageSip 
      });

      const result = await leaderboardService.copyPortfolio(
        req.userId,
        sourceSecretCode,
        investmentType,
        averageSip
      );

      return response.success(res, result, result.message);

    } catch (error) {
      logger.error('Error copying portfolio:', error);
      return response.error(res, 'Failed to copy portfolio', 500, error);
    }
  }

  /**
   * Get user's leaderboard history
   */
  async getUserLeaderboardHistory(req, res) {
    try {
      if (!req.userId) {
        return response.error(res, 'Authentication required', 401);
      }

      logger.info('Getting user leaderboard history', { userId: req.userId });

      const history = await leaderboardService.getUserLeaderboardHistory(req.userId);

      return response.success(res, history, 'Leaderboard history retrieved successfully');

    } catch (error) {
      logger.error('Error getting user leaderboard history:', error);
      return response.error(res, 'Failed to get leaderboard history', 500, error);
    }
  }

  /**
   * Get portfolio copy history
   */
  async getPortfolioCopyHistory(req, res) {
    try {
      if (!req.userId) {
        return response.error(res, 'Authentication required', 401);
      }

      logger.info('Getting portfolio copy history', { userId: req.userId });

      const history = await leaderboardService.getPortfolioCopyHistory(req.userId);

      return response.success(res, history, 'Portfolio copy history retrieved successfully');

    } catch (error) {
      logger.error('Error getting portfolio copy history:', error);
      return response.error(res, 'Failed to get portfolio copy history', 500, error);
    }
  }

  /**
   * Manually trigger leaderboard generation (admin endpoint)
   */
  async generateLeaderboards(req, res) {
    try {
      if (!req.userId) {
        return response.error(res, 'Authentication required', 401);
      }

      logger.info('Manually generating leaderboards', { userId: req.userId });

      const results = await leaderboardService.generateAllLeaderboards();

      return response.success(res, results, 'Leaderboards generated successfully');

    } catch (error) {
      logger.error('Error generating leaderboards:', error);
      return response.error(res, 'Failed to generate leaderboards', 500, error);
    }
  }

  /**
   * Update XIRR for all portfolios (admin endpoint)
   */
  async updateAllXIRR(req, res) {
    try {
      if (!req.userId) {
        return response.error(res, 'Authentication required', 401);
      }

      logger.info('Updating XIRR for all portfolios', { userId: req.userId });

      const updatedCount = await leaderboardService.updateAllPortfolioXIRR();

      return response.success(res, { updatedCount }, `XIRR updated for ${updatedCount} portfolios`);

    } catch (error) {
      logger.error('Error updating XIRR:', error);
      return response.error(res, 'Failed to update XIRR', 500, error);
    }
  }

  /**
   * Get leaderboard statistics
   */
  async getLeaderboardStats(req, res) {
    try {
      const { duration } = req.params;
      
      if (!['1M', '3M', '6M', '1Y', '3Y'].includes(duration)) {
        return response.error(res, 'Invalid duration. Must be 1M, 3M, 6M, 1Y, or 3Y', 400);
      }

      logger.info('Getting leaderboard stats', { duration });

      const leaderboard = await leaderboardService.getLeaderboard(duration);

      if (!leaderboard) {
        return response.error(res, 'Leaderboard not found for this duration', 404);
      }

      const stats = {
        duration: leaderboard.duration,
        totalParticipants: leaderboard.totalParticipants,
        averageReturn: leaderboard.averageReturn,
        medianReturn: leaderboard.medianReturn,
        topReturn: leaderboard.leaders[0]?.returnPercent || 0,
        bottomReturn: leaderboard.leaders[leaderboard.leaders.length - 1]?.returnPercent || 0,
        generatedAt: leaderboard.generatedAt
      };

      return response.success(res, stats, 'Leaderboard statistics retrieved successfully');

    } catch (error) {
      logger.error('Error getting leaderboard stats:', error);
      return response.error(res, 'Failed to get leaderboard statistics', 500, error);
    }
  }

  /**
   * Get user's current rank in leaderboard
   */
  async getUserRank(req, res) {
    try {
      if (!req.userId) {
        return response.error(res, 'Authentication required', 401);
      }

      const { duration } = req.params;
      
      if (!['1M', '3M', '6M', '1Y', '3Y'].includes(duration)) {
        return response.error(res, 'Invalid duration. Must be 1M, 3M, 6M, 1Y, or 3Y', 400);
      }

      logger.info('Getting user rank', { userId: req.userId, duration });

      const leaderboard = await leaderboardService.getLeaderboard(duration);

      if (!leaderboard) {
        return response.error(res, 'Leaderboard not found for this duration', 404);
      }

      // Find user's rank
      const userEntry = leaderboard.leaders.find(leader => {
        // We need to get the user's secret code to match
        // This is a simplified approach - in production, you'd want to optimize this
        return leader.userId === req.userId;
      });

      if (!userEntry) {
        return response.success(res, { 
          rank: null, 
          returnPercent: 0, 
          totalParticipants: leaderboard.totalParticipants 
        }, 'User not found in leaderboard');
      }

      return response.success(res, {
        rank: userEntry.rank,
        returnPercent: userEntry.returnPercent,
        totalParticipants: leaderboard.totalParticipants,
        allocation: userEntry.allocation
      }, 'User rank retrieved successfully');

    } catch (error) {
      logger.error('Error getting user rank:', error);
      return response.error(res, 'Failed to get user rank', 500, error);
    }
  }

  /**
   * Get agent leaderboard
   */
  async getAgentLeaderboard(req, res) {
    try {
      return res.status(200).json({ message: 'Agent leaderboard retrieved successfully' });
    } catch (error) {
      logger.error('Error getting agent leaderboard:', error);
      return response.error(res, 'Failed to get agent leaderboard', 500, error);
    }
  }

  /**
   * Get performance leaderboard
   */
  async getPerformanceLeaderboard(req, res) {
    try {
      return res.status(200).json({ message: 'Performance leaderboard retrieved successfully' });
    } catch (error) {
      logger.error('Error getting performance leaderboard:', error);
      return response.error(res, 'Failed to get performance leaderboard', 500, error);
    }
  }
}

module.exports = new LeaderboardController(); 