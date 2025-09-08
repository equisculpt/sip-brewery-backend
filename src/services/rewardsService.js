const { Reward, RewardSummary, Referral, User, Transaction } = require('../models');
const logger = require('../utils/logger');

class RewardsService {
  /**
   * Initialize the rewards service
   */
  async initialize() {
    try {
      logger.info('Initializing Rewards Service...');
      
      // Create indexes if they don't exist
      await Reward.createIndexes();
      await RewardSummary.createIndexes();
      await Referral.createIndexes();
      
      logger.info('✅ Rewards Service initialized successfully');
      return true;
    } catch (error) {
      logger.error('❌ Failed to initialize Rewards Service:', error);
      throw error;
    }
  }

  /**
   * Get service status
   */
  getStatus() {
    return 'ACTIVE';
  }

  /**
   * Calculate rewards for a user
   */
  async calculateRewards(userId) {
    try {
      logger.info(`Calculating rewards for user: ${userId}`);
      
      // Get user's reward summary
      const summary = await this.getUserRewardSummary(userId);
      
      // Calculate potential rewards
      const potentialRewards = await this.calculatePotentialRewards(userId);
      
      // Get recent activity
      const recentActivity = await this.getRecentActivity(userId);
      
      return {
        success: true,
        summary,
        potentialRewards,
        recentActivity,
        calculatedAt: new Date()
      };
    } catch (error) {
      logger.error('Error calculating rewards:', error);
      throw error;
    }
  }

  /**
   * Calculate potential rewards for user
   */
  async calculatePotentialRewards(userId) {
    try {
      const user = await User.findOne({ supabaseId: userId });
      if (!user) {
        throw new Error('User not found');
      }

      // Calculate potential SIP loyalty points
      const sipTransactions = await Transaction.find({
        userId: user._id,
        type: 'SIP',
        status: 'COMPLETED'
      });

      const potentialSipPoints = sipTransactions.length;

      // Calculate potential referral bonus
      const referrals = await Referral.find({
        referrerId: userId,
        status: 'PENDING'
      });

      const potentialReferralBonus = referrals.length * 1000; // ₹1000 per referral

      // Calculate potential cashback
      const fundSipCounts = {};
      sipTransactions.forEach(tx => {
        const fundKey = `${tx.fundName}-${tx.folioNumber}`;
        fundSipCounts[fundKey] = (fundSipCounts[fundKey] || 0) + 1;
      });

      let potentialCashback = 0;
      Object.values(fundSipCounts).forEach(count => {
        if (count >= 12) {
          potentialCashback += 500; // ₹500 for 12 SIPs
        }
      });

      return {
        potentialSipPoints,
        potentialReferralBonus,
        potentialCashback,
        totalPotential: potentialSipPoints + potentialReferralBonus + potentialCashback
      };
    } catch (error) {
      logger.error('Error calculating potential rewards:', error);
      throw error;
    }
  }

  /**
   * Get recent activity for user
   */
  async getRecentActivity(userId) {
    try {
      const user = await User.findOne({ supabaseId: userId });
      if (!user) {
        throw new Error('User not found');
      }

      const recentRewards = await Reward.find({ userId })
        .sort({ createdAt: -1 })
        .limit(5);

      const recentTransactions = await Transaction.find({
        userId: user._id
      })
        .sort({ createdAt: -1 })
        .limit(5);

      return {
        recentRewards,
        recentTransactions,
        lastActivity: recentRewards[0]?.createdAt || recentTransactions[0]?.createdAt
      };
    } catch (error) {
      logger.error('Error getting recent activity:', error);
      throw error;
    }
  }

  /**
   * Get user's reward summary
   */
  async getUserRewardSummary(userId) {
    try {
      let summary = await RewardSummary.findOne({ userId });
      
      if (!summary) {
        // Create summary if doesn't exist
        summary = new RewardSummary({ userId });
        await summary.save();
      }
      
      // Get recent transactions
      const recentTransactions = await Reward.find({ userId })
        .sort({ createdAt: -1 })
        .limit(10)
        .select('type amount points description status createdAt');
      
      return {
        totalPoints: summary.totalPoints,
        totalCashback: summary.totalCashback,
        totalReferralBonus: summary.totalReferralBonus,
        totalSipInstallments: summary.totalSipInstallments,
        pendingPayout: summary.pendingPayout,
        totalPaidOut: summary.totalPaidOut,
        recentTransactions,
        lastUpdated: summary.lastUpdated
      };
    } catch (error) {
      logger.error('Error getting user reward summary:', error);
      throw error;
    }
  }

  /**
   * Get paginated reward transactions
   */
  async getRewardTransactions(userId, options = {}) {
    try {
      const { page = 1, limit = 20, type, startDate, endDate } = options;
      const skip = (page - 1) * limit;
      
      const query = { userId };
      
      if (type) query.type = type;
      if (startDate || endDate) {
        query.createdAt = {};
        if (startDate) query.createdAt.$gte = new Date(startDate);
        if (endDate) query.createdAt.$lte = new Date(endDate);
      }
      
      const transactions = await Reward.find(query)
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(limit)
        .select('type amount points description status isPaid createdAt');
      
      const total = await Reward.countDocuments(query);
      
      return {
        transactions,
        pagination: {
          page,
          limit,
          total,
          pages: Math.ceil(total / limit)
        }
      };
    } catch (error) {
      logger.error('Error getting reward transactions:', error);
      throw error;
    }
  }

  /**
   * Award SIP loyalty points (1 point per successful SIP)
   */
  async awardSipLoyaltyPoints(userId, sipId, fundName, folioNumber, bseConfirmationId) {
    try {
      // Verify user exists and KYC is verified
      const user = await User.findOne({ supabaseId: userId });
      if (!user || user.kycStatus !== 'VERIFIED') {
        throw new Error('User not found or KYC not verified');
      }
      
      // Check if reward already exists for this SIP
      const existingReward = await Reward.findOne({ 
        userId, 
        sipId, 
        type: 'SIP_LOYALTY_POINTS' 
      });
      
      if (existingReward) {
        throw new Error('SIP loyalty points already awarded for this transaction');
      }
      
      // Create reward transaction
      const reward = new Reward({
        userId,
        type: 'SIP_LOYALTY_POINTS',
        amount: 0,
        points: 1,
        description: `SIP loyalty point for ${fundName}`,
        status: 'CREDITED',
        sipId,
        fundName,
        folioNumber,
        bseConfirmationId,
        transactionTimestamp: new Date()
      });
      
      await reward.save();
      
      // Update summary
      await this.updateRewardSummary(userId);
      
      logger.info(`Awarded SIP loyalty point to user ${userId} for SIP ${sipId}`);
      
      return {
        success: true,
        message: "🎉 You earned 1 loyalty point for your SIP! Keep investing regularly.",
        reward
      };
    } catch (error) {
      logger.error('Error awarding SIP loyalty points:', error);
      throw error;
    }
  }

  /**
   * Award cashback for completing 12 SIPs in one fund
   */
  async checkAndAwardCashback(userId, fundName, folioNumber) {
    try {
      // Count SIPs for this fund
      const sipCount = await Reward.countDocuments({
        userId,
        type: 'SIP_LOYALTY_POINTS',
        fundName,
        folioNumber,
        status: 'CREDITED'
      });
      
      if (sipCount === 12) {
        // Check if cashback already awarded
        const existingCashback = await Reward.findOne({
          userId,
          type: 'CASHBACK_12_SIPS',
          fundName,
          folioNumber
        });
        
        if (existingCashback) {
          return { success: false, message: 'Cashback already awarded for this fund' };
        }
        
        // Award cashback
        const cashback = new Reward({
          userId,
          type: 'CASHBACK_12_SIPS',
          amount: 500,
          points: 0,
          description: `₹500 cashback for completing 12 SIPs in ${fundName}`,
          status: 'CREDITED',
          fundName,
          folioNumber,
          transactionTimestamp: new Date()
        });
        
        await cashback.save();
        
        // Update summary
        await this.updateRewardSummary(userId);
        
        logger.info(`Awarded ₹500 cashback to user ${userId} for completing 12 SIPs in ${fundName}`);
        
        return {
          success: true,
          message: "🎉 Congratulations! You've earned ₹500 cashback for completing 12 SIPs!",
          cashback
        };
      }
      
      return { success: false, message: 'Not enough SIPs for cashback yet' };
    } catch (error) {
      logger.error('Error checking/awarding cashback:', error);
      throw error;
    }
  }

  /**
   * Validate and award referral bonus
   */
  async validateAndAwardReferralBonus(referredId) {
    try {
      // Find the referred user
      const referredUser = await User.findOne({ supabaseId: referredId });
      if (!referredUser || !referredUser.referredBy) {
        throw new Error('Referred user not found or no referrer');
      }
      
      const referrerId = referredUser.referredBy;
      
      // Check if referral bonus already paid
      const existingReferral = await Referral.findOne({
        referrerId,
        referredId,
        status: 'BONUS_PAID'
      });
      
      if (existingReferral) {
        throw new Error('Referral bonus already paid');
      }
      
      // Verify referrer exists and is active
      const referrer = await User.findOne({ supabaseId: referrerId });
      if (!referrer || !referrer.isActive) {
        throw new Error('Referrer not found or inactive');
      }
      
      // Anti-abuse checks
      if (referrerId === referredId) {
        throw new Error('Self-referral not allowed');
      }
      
      // Check referral limit (50 per year)
      const currentYear = new Date().getFullYear();
      const yearStart = new Date(currentYear, 0, 1);
      const yearEnd = new Date(currentYear, 11, 31);
      
      const yearlyReferrals = await Referral.countDocuments({
        referrerId,
        referredAt: { $gte: yearStart, $lte: yearEnd },
        status: 'BONUS_PAID'
      });
      
      if (yearlyReferrals >= 50) {
        throw new Error('Referral limit reached for this year');
      }
      
      // Verify referred user has completed KYC and started SIP
      if (referredUser.kycStatus !== 'VERIFIED') {
        throw new Error('Referred user KYC not verified');
      }
      
      // Check if referred user has at least one successful SIP
      const sipCount = await Reward.countDocuments({
        userId: referredId,
        type: 'SIP_LOYALTY_POINTS',
        status: 'CREDITED'
      });
      
      if (sipCount === 0) {
        throw new Error('Referred user has not started SIP yet');
      }
      
      // Create referral record
      const referral = new Referral({
        referrerId,
        referredId,
        referralCode: referredUser.referralCode,
        status: 'BONUS_PAID',
        kycCompletedAt: new Date(),
        sipStartedAt: new Date(),
        bonusPaidAt: new Date(),
        bonusAmount: 100,
        bonusPaid: true
      });
      
      await referral.save();
      
      // Award referral bonus
      const bonus = new Reward({
        userId: referrerId,
        type: 'REFERRAL_BONUS',
        amount: 100,
        points: 0,
        description: `₹100 referral bonus for ${referredUser.name}`,
        status: 'CREDITED',
        referralId: referral._id,
        transactionTimestamp: new Date()
      });
      
      await bonus.save();
      
      // Update referral record with bonus transaction ID
      referral.bonusTransactionId = bonus._id;
      await referral.save();
      
      // Update summaries
      await this.updateRewardSummary(referrerId);
      await this.updateUserReferralStats(referrerId);
      
      logger.info(`Awarded ₹100 referral bonus to ${referrerId} for referring ${referredId}`);
      
      return {
        success: true,
        message: "🎉 You just earned ₹100 for referring a new investor! Keep going.",
        bonus,
        referral
      };
    } catch (error) {
      logger.error('Error validating/awarding referral bonus:', error);
      throw error;
    }
  }

  /**
   * Simulate SIP reward (for testing)
   */
  async simulateSipReward(userId, sipId, fundName = 'Test Fund', folioNumber = 'TEST123') {
    try {
      const result = await this.awardSipLoyaltyPoints(
        userId, 
        sipId, 
        fundName, 
        folioNumber, 
        'SIMULATED_BSE_ID'
      );
      
      // Check for cashback eligibility
      const cashbackResult = await this.checkAndAwardCashback(userId, fundName, folioNumber);
      
      return {
        sipReward: result,
        cashbackCheck: cashbackResult
      };
    } catch (error) {
      logger.error('Error simulating SIP reward:', error);
      throw error;
    }
  }

  /**
   * Update reward summary for user
   */
  async updateRewardSummary(userId) {
    try {
      const rewards = await Reward.find({ 
        userId, 
        status: { $in: ['CREDITED', 'REDEEMED'] } 
      });
      
      const summary = await RewardSummary.findOne({ userId }) || new RewardSummary({ userId });
      
      // Calculate totals
      summary.totalPoints = rewards
        .filter(r => r.type === 'SIP_LOYALTY_POINTS')
        .reduce((sum, r) => sum + r.points, 0);
      
      summary.totalCashback = rewards
        .filter(r => r.type === 'CASHBACK_12_SIPS')
        .reduce((sum, r) => sum + r.amount, 0);
      
      summary.totalReferralBonus = rewards
        .filter(r => r.type === 'REFERRAL_BONUS')
        .reduce((sum, r) => sum + r.amount, 0);
      
      summary.totalSipInstallments = rewards
        .filter(r => r.type === 'SIP_LOYALTY_POINTS')
        .length;
      
      // Calculate pending payout
      const unpaidRewards = await Reward.find({
        userId,
        isPaid: false,
        status: 'CREDITED',
        type: { $in: ['CASHBACK_12_SIPS', 'REFERRAL_BONUS'] }
      });
      
      summary.pendingPayout = unpaidRewards.reduce((sum, r) => sum + r.amount, 0);
      
      // Calculate total paid out
      const paidRewards = await Reward.find({
        userId,
        isPaid: true,
        type: { $in: ['CASHBACK_12_SIPS', 'REFERRAL_BONUS'] }
      });
      
      summary.totalPaidOut = paidRewards.reduce((sum, r) => sum + r.amount, 0);
      
      summary.lastUpdated = new Date();
      await summary.save();
      
      return summary;
    } catch (error) {
      logger.error('Error updating reward summary:', error);
      throw error;
    }
  }

  /**
   * Update user referral statistics
   */
  async updateUserReferralStats(userId) {
    try {
      const referrals = await Referral.find({ 
        referrerId: userId, 
        status: 'BONUS_PAID' 
      });
      
      const user = await User.findOne({ supabaseId: userId });
      if (user) {
        user.referralCount = referrals.length;
        user.totalReferralBonus = referrals.reduce((sum, r) => sum + r.bonusAmount, 0);
        await user.save();
      }
      
      return { referralCount: referrals.length, totalBonus: user.totalReferralBonus };
    } catch (error) {
      logger.error('Error updating user referral stats:', error);
      throw error;
    }
  }

  /**
   * Mark reward as paid (admin function)
   */
  async markRewardAsPaid(rewardId, adminId) {
    try {
      const reward = await Reward.findById(rewardId);
      if (!reward) {
        throw new Error('Reward not found');
      }
      
      reward.isPaid = true;
      reward.paidAt = new Date();
      reward.paidBy = adminId;
      await reward.save();
      
      // Update summary
      await this.updateRewardSummary(reward.userId);
      
      logger.info(`Reward ${rewardId} marked as paid by admin ${adminId}`);
      
      return reward;
    } catch (error) {
      logger.error('Error marking reward as paid:', error);
      throw error;
    }
  }

  /**
   * Get unpaid rewards for admin export
   */
  async getUnpaidRewards() {
    try {
      const unpaidRewards = await Reward.find({
        isPaid: false,
        status: 'CREDITED',
        type: { $in: ['CASHBACK_12_SIPS', 'REFERRAL_BONUS'] }
      }).populate('userId', 'name email phone');
      
      return unpaidRewards;
    } catch (error) {
      logger.error('Error getting unpaid rewards:', error);
      throw error;
    }
  }

  /**
   * Get referral leaderboard
   */
  async getReferralLeaderboard(limit = 10) {
    try {
      const leaderboard = await User.find({ isActive: true })
        .sort({ referralCount: -1, totalReferralBonus: -1 })
        .limit(limit)
        .select('name referralCount totalReferralBonus');
      
      return leaderboard;
    } catch (error) {
      logger.error('Error getting referral leaderboard:', error);
      throw error;
    }
  }

  /**
   * Revoke referral bonus if SIP cancelled within 3 months
   */
  async revokeReferralBonusIfNeeded(referredId) {
    try {
      const referral = await Referral.findOne({
        referredId,
        status: 'BONUS_PAID',
        bonusPaid: true
      });
      
      if (!referral) return null;
      
      const threeMonthsAgo = new Date();
      threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3);
      
      // Check if SIP was cancelled within 3 months
      if (referral.sipCancelledAt && referral.sipCancelledAt > threeMonthsAgo) {
        // Revoke the bonus
        const reward = await Reward.findById(referral.bonusTransactionId);
        if (reward) {
          reward.status = 'REVOKED';
          reward.description += ' (Revoked - SIP cancelled within 3 months)';
          await reward.save();
        }
        
        referral.status = 'CANCELLED';
        referral.validationNotes = 'SIP cancelled within 3 months';
        await referral.save();
        
        // Update summaries
        await this.updateRewardSummary(referral.referrerId);
        await this.updateUserReferralStats(referral.referrerId);
        
        logger.info(`Revoked referral bonus for ${referral.referrerId} due to early SIP cancellation`);
        
        return { revoked: true, referral, reward };
      }
      
      return { revoked: false };
    } catch (error) {
      logger.error('Error revoking referral bonus:', error);
      throw error;
    }
  }
}

module.exports = new RewardsService(); 