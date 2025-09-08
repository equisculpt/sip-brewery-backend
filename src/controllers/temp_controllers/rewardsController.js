const rewardsService = require('../services/rewardsService');
const logger = require('../utils/logger');

class RewardsController {
  /**
   * Get user's reward summary
   * GET /api/rewards/summary
   */
  async getRewardSummary(req, res) {
    try {
      const userId = req.user.supabaseId;
      
      const summary = await rewardsService.getUserRewardSummary(userId);
      
      res.json({
        success: true,
        data: summary
      });
    } catch (error) {
      logger.error('Error in getRewardSummary:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to fetch reward summary',
        error: error.message
      });
    }
  }

  /**
   * Get paginated reward transactions
   * GET /api/rewards/transactions
   */
  async getRewardTransactions(req, res) {
    try {
      const userId = req.user.supabaseId;
      const { page, limit, type, startDate, endDate } = req.query;
      
      const options = {
        page: parseInt(page) || 1,
        limit: parseInt(limit) || 20,
        type,
        startDate,
        endDate
      };
      
      const result = await rewardsService.getRewardTransactions(userId, options);
      
      res.json({
        success: true,
        data: result
      });
    } catch (error) {
      logger.error('Error in getRewardTransactions:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to fetch reward transactions',
        error: error.message
      });
    }
  }

  /**
   * Simulate SIP reward (for testing)
   * POST /api/rewards/simulate-sip-reward
   */
  async simulateSipReward(req, res) {
    try {
      const userId = req.user.supabaseId;
      const { sipId, fundName, folioNumber } = req.body;
      
      if (!sipId) {
        return res.status(400).json({
          success: false,
          message: 'SIP ID is required'
        });
      }
      
      const result = await rewardsService.simulateSipReward(
        userId, 
        sipId, 
        fundName || 'Test Fund', 
        folioNumber || 'TEST123'
      );
      
      res.json({
        success: true,
        data: result
      });
    } catch (error) {
      logger.error('Error in simulateSipReward:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to simulate SIP reward',
        error: error.message
      });
    }
  }

  /**
   * Validate and award referral bonus
   * POST /api/rewards/validate-referral
   */
  async validateReferral(req, res) {
    try {
      const { referredId } = req.body;
      
      if (!referredId) {
        return res.status(400).json({
          success: false,
          message: 'Referred user ID is required'
        });
      }
      
      const result = await rewardsService.validateAndAwardReferralBonus(referredId);
      
      res.json({
        success: true,
        data: result
      });
    } catch (error) {
      logger.error('Error in validateReferral:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to validate referral',
        error: error.message
      });
    }
  }

  /**
   * Award SIP loyalty points (called by transaction system)
   * POST /api/rewards/award-sip-points
   */
  async awardSipPoints(req, res) {
    try {
      const { userId, sipId, fundName, folioNumber, bseConfirmationId } = req.body;
      
      if (!userId || !sipId || !fundName || !folioNumber) {
        return res.status(400).json({
          success: false,
          message: 'Missing required fields: userId, sipId, fundName, folioNumber'
        });
      }
      
      const result = await rewardsService.awardSipLoyaltyPoints(
        userId, 
        sipId, 
        fundName, 
        folioNumber, 
        bseConfirmationId
      );
      
      // Check for cashback eligibility
      const cashbackResult = await rewardsService.checkAndAwardCashback(
        userId, 
        fundName, 
        folioNumber
      );
      
      res.json({
        success: true,
        data: {
          sipReward: result,
          cashbackCheck: cashbackResult
        }
      });
    } catch (error) {
      logger.error('Error in awardSipPoints:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to award SIP points',
        error: error.message
      });
    }
  }

  /**
   * Get user's referral code
   * GET /api/rewards/referral-code
   */
  async getReferralCode(req, res) {
    try {
      const userId = req.user.supabaseId;
      
      // Get user's referral code and stats
      const { User } = require('../models');
      const user = await User.findOne({ supabaseId: userId })
        .select('referralCode referralCount totalReferralBonus');
      
      if (!user) {
        return res.status(404).json({
          success: false,
          message: 'User not found'
        });
      }
      
      res.json({
        success: true,
        data: {
          referralCode: user.referralCode,
          referralCount: user.referralCount,
          totalReferralBonus: user.totalReferralBonus
        }
      });
    } catch (error) {
      logger.error('Error in getReferralCode:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to fetch referral code',
        error: error.message
      });
    }
  }

  /**
   * Get referral leaderboard
   * GET /api/rewards/leaderboard
   */
  async getReferralLeaderboard(req, res) {
    try {
      const { limit } = req.query;
      const leaderboardLimit = parseInt(limit) || 10;
      
      const leaderboard = await rewardsService.getReferralLeaderboard(leaderboardLimit);
      
      res.json({
        success: true,
        data: leaderboard
      });
    } catch (error) {
      logger.error('Error in getReferralLeaderboard:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to fetch referral leaderboard',
        error: error.message
      });
    }
  }

  /**
   * Admin: Mark reward as paid
   * POST /api/admin/rewards/mark-paid
   */
  async markRewardAsPaid(req, res) {
    try {
      const { rewardId } = req.body;
      const adminId = req.admin.id;
      
      if (!rewardId) {
        return res.status(400).json({
          success: false,
          message: 'Reward ID is required'
        });
      }
      
      const result = await rewardsService.markRewardAsPaid(rewardId, adminId);
      
      res.json({
        success: true,
        data: result,
        message: 'Reward marked as paid successfully'
      });
    } catch (error) {
      logger.error('Error in markRewardAsPaid:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to mark reward as paid',
        error: error.message
      });
    }
  }

  /**
   * Admin: Get unpaid rewards for export
   * GET /api/admin/rewards/unpaid
   */
  async getUnpaidRewards(req, res) {
    try {
      const unpaidRewards = await rewardsService.getUnpaidRewards();
      
      res.json({
        success: true,
        data: unpaidRewards,
        count: unpaidRewards.length
      });
    } catch (error) {
      logger.error('Error in getUnpaidRewards:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to fetch unpaid rewards',
        error: error.message
      });
    }
  }

  /**
   * Admin: Get user's full reward history
   * GET /api/admin/rewards/user/:userId
   */
  async getUserRewardHistory(req, res) {
    try {
      const { userId } = req.params;
      
      const summary = await rewardsService.getUserRewardSummary(userId);
      const transactions = await rewardsService.getRewardTransactions(userId, { limit: 100 });
      
      res.json({
        success: true,
        data: {
          summary,
          transactions: transactions.transactions
        }
      });
    } catch (error) {
      logger.error('Error in getUserRewardHistory:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to fetch user reward history',
        error: error.message
      });
    }
  }

  /**
   * Admin: Export unpaid rewards as CSV
   * GET /api/admin/rewards/export-csv
   */
  async exportUnpaidRewardsCSV(req, res) {
    try {
      const unpaidRewards = await rewardsService.getUnpaidRewards();
      
      // Generate CSV content
      const csvHeader = 'User ID,Name,Email,Phone,Type,Amount,Description,Created At\n';
      const csvRows = unpaidRewards.map(reward => {
        const user = reward.userId;
        return `${user._id || user.supabaseId},"${user.name || ''}","${user.email || ''}","${user.phone || ''}","${reward.type}","${reward.amount}","${reward.description}","${reward.createdAt}"`;
      }).join('\n');
      
      const csvContent = csvHeader + csvRows;
      
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', 'attachment; filename="unpaid-rewards.csv"');
      res.send(csvContent);
    } catch (error) {
      logger.error('Error in exportUnpaidRewardsCSV:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to export unpaid rewards',
        error: error.message
      });
    }
  }

  /**
   * Revoke referral bonus if needed (called by transaction system)
   * POST /api/rewards/revoke-referral
   */
  async revokeReferralBonus(req, res) {
    try {
      const { referredId } = req.body;
      
      if (!referredId) {
        return res.status(400).json({
          success: false,
          message: 'Referred user ID is required'
        });
      }
      
      const result = await rewardsService.revokeReferralBonusIfNeeded(referredId);
      
      res.json({
        success: true,
        data: result
      });
    } catch (error) {
      logger.error('Error in revokeReferralBonus:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to revoke referral bonus',
        error: error.message
      });
    }
  }
}

module.exports = new RewardsController(); 