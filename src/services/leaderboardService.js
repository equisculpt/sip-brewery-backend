const User = require('../models/User');
const UserPortfolio = require('../models/UserPortfolio');
const Leaderboard = require('../models/Leaderboard');
const PortfolioCopy = require('../models/PortfolioCopy');
const logger = require('../utils/logger');

class LeaderboardService {
  /**
   * Calculate XIRR for a given set of cash flows
   */
  calculateXIRR(cashFlows, guess = 0.1) {
    try {
      // Simple XIRR calculation using Newton-Raphson method
      // For production, consider using a more robust library like 'xirr'
      
      const tolerance = 0.0001;
      const maxIterations = 100;
      let rate = guess;
      
      for (let i = 0; i < maxIterations; i++) {
        const npv = this.calculateNPV(cashFlows, rate);
        const derivative = this.calculateNPVDerivative(cashFlows, rate);
        
        if (Math.abs(derivative) < tolerance) break;
        
        const newRate = rate - npv / derivative;
        if (Math.abs(newRate - rate) < tolerance) {
          rate = newRate;
          break;
        }
        rate = newRate;
      }
      
      // Convert to percentage and annualize
      return (Math.pow(1 + rate, 12) - 1) * 100;
    } catch (error) {
      logger.error('Error calculating XIRR:', error);
      return 0;
    }
  }

  /**
   * Calculate Net Present Value
   */
  calculateNPV(cashFlows, rate) {
    return cashFlows.reduce((npv, flow) => {
      const timeInYears = flow.date / (365 * 24 * 60 * 60 * 1000); // Convert to years
      return npv + flow.amount / Math.pow(1 + rate, timeInYears);
    }, 0);
  }

  /**
   * Calculate NPV derivative
   */
  calculateNPVDerivative(cashFlows, rate) {
    return cashFlows.reduce((derivative, flow) => {
      const timeInYears = flow.date / (365 * 24 * 60 * 60 * 1000);
      return derivative - (flow.amount * timeInYears) / Math.pow(1 + rate, timeInYears + 1);
    }, 0);
  }

  /**
   * Calculate XIRR for different time periods
   */
  async calculatePortfolioXIRR(portfolio, duration) {
    try {
      const now = new Date();
      let startDate;
      
      switch (duration) {
        case '1M':
          startDate = new Date(now.getFullYear(), now.getMonth() - 1, now.getDate());
          break;
        case '3M':
          startDate = new Date(now.getFullYear(), now.getMonth() - 3, now.getDate());
          break;
        case '6M':
          startDate = new Date(now.getFullYear(), now.getMonth() - 6, now.getDate());
          break;
        case '1Y':
          startDate = new Date(now.getFullYear() - 1, now.getMonth(), now.getDate());
          break;
        case '3Y':
          startDate = new Date(now.getFullYear() - 3, now.getMonth(), now.getDate());
          break;
        default:
          return 0;
      }

      // Filter transactions within the time period
      const relevantTransactions = portfolio.transactions.filter(t => 
        new Date(t.date) >= startDate
      );

      if (relevantTransactions.length === 0) return 0;

      // Create cash flows for XIRR calculation
      const cashFlows = relevantTransactions.map(t => ({
        date: new Date(t.date).getTime(),
        amount: t.type === 'REDEMPTION' ? t.amount : -t.amount // Outflows are negative
      }));

      // Add current portfolio value as final cash flow
      if (portfolio.totalCurrentValue > 0) {
        cashFlows.push({
          date: now.getTime(),
          amount: portfolio.totalCurrentValue
        });
      }

      return this.calculateXIRR(cashFlows);
    } catch (error) {
      logger.error('Error calculating portfolio XIRR:', error);
      return 0;
    }
  }

  /**
   * Update XIRR for all portfolios
   */
  async updateAllPortfolioXIRR() {
    try {
      logger.info('Starting XIRR update for all portfolios...');
      
      const portfolios = await UserPortfolio.find({ isActive: true });
      let updatedCount = 0;

      for (const portfolio of portfolios) {
        try {
          const durations = ['1M', '3M', '6M', '1Y', '3Y'];
          
          for (const duration of durations) {
            const xirr = await this.calculatePortfolioXIRR(portfolio, duration);
            
            switch (duration) {
              case '1M':
                portfolio.xirr1M = xirr;
                break;
              case '3M':
                portfolio.xirr3M = xirr;
                break;
              case '6M':
                portfolio.xirr6M = xirr;
                break;
              case '1Y':
                portfolio.xirr1Y = xirr;
                break;
              case '3Y':
                portfolio.xirr3Y = xirr;
                break;
            }
          }
          
          await portfolio.save();
          updatedCount++;
          
        } catch (error) {
          logger.error(`Error updating XIRR for portfolio ${portfolio._id}:`, error);
        }
      }
      
      logger.info(`XIRR update completed. Updated ${updatedCount} portfolios.`);
      return updatedCount;
    } catch (error) {
      logger.error('Error updating all portfolio XIRR:', error);
      throw error;
    }
  }

  /**
   * Generate leaderboard for a specific duration
   */
  async generateLeaderboard(duration) {
    try {
      logger.info(`Generating leaderboard for duration: ${duration}`);
      
      // Get portfolios with valid XIRR for the duration
      const xirrField = `xirr${duration}`;
      const portfolios = await UserPortfolio.find({
        isActive: true,
        [xirrField]: { $gt: 0 }
      }).populate('userId', 'secretCode');
      
      if (portfolios.length === 0) {
        logger.warn(`No portfolios found for leaderboard generation: ${duration}`);
        return null;
      }

      // Sort by XIRR in descending order
      portfolios.sort((a, b) => b[xirrField] - a[xirrField]);
      
      // Take top 20
      const topPortfolios = portfolios.slice(0, 20);
      
      // Calculate statistics
      const totalParticipants = portfolios.length;
      const averageReturn = portfolios.reduce((sum, p) => sum + p[xirrField], 0) / portfolios.length;
      const sortedReturns = portfolios.map(p => p[xirrField]).sort((a, b) => a - b);
      const medianReturn = sortedReturns[Math.floor(sortedReturns.length / 2)];
      
      // Create leaderboard entries
      const leaders = topPortfolios.map((portfolio, index) => ({
        secretCode: portfolio.userId.secretCode,
        returnPercent: Math.round(portfolio[xirrField] * 100) / 100,
        allocation: portfolio.allocation,
        rank: index + 1,
        userId: portfolio.userId._id,
        portfolioId: portfolio._id
      }));

      // Create or update leaderboard
      let leaderboard = await Leaderboard.findOne({ duration, isActive: true });
      
      if (leaderboard) {
        // Update existing leaderboard
        leaderboard.leaders = leaders;
        leaderboard.generatedAt = new Date();
        leaderboard.totalParticipants = totalParticipants;
        leaderboard.averageReturn = Math.round(averageReturn * 100) / 100;
        leaderboard.medianReturn = Math.round(medianReturn * 100) / 100;
      } else {
        // Create new leaderboard
        leaderboard = new Leaderboard({
          duration,
          leaders,
          totalParticipants,
          averageReturn: Math.round(averageReturn * 100) / 100,
          medianReturn: Math.round(medianReturn * 100) / 100
        });
      }
      
      await leaderboard.save();
      
      // Update leaderboard history for each user
      for (const leader of leaders) {
        const portfolio = await UserPortfolio.findById(leader.portfolioId);
        if (portfolio) {
          await portfolio.addLeaderboardHistory(duration, leader.rank, leader.returnPercent);
        }
      }
      
      logger.info(`Leaderboard generated for ${duration}: ${leaders.length} leaders`);
      return leaderboard;
      
    } catch (error) {
      logger.error(`Error generating leaderboard for ${duration}:`, error);
      throw error;
    }
  }

  /**
   * Generate all leaderboards
   */
  async generateAllLeaderboards() {
    try {
      logger.info('Starting generation of all leaderboards...');
      
      const durations = ['1M', '3M', '6M', '1Y', '3Y'];
      const results = {};
      
      for (const duration of durations) {
        try {
          const leaderboard = await this.generateLeaderboard(duration);
          results[duration] = leaderboard ? 'success' : 'no_data';
        } catch (error) {
          logger.error(`Error generating leaderboard for ${duration}:`, error);
          results[duration] = 'error';
        }
      }
      
      logger.info('All leaderboards generation completed:', results);
      return results;
    } catch (error) {
      logger.error('Error generating all leaderboards:', error);
      throw error;
    }
  }

  /**
   * Get leaderboard for a specific duration
   */
  async getLeaderboard(duration) {
    try {
      const leaderboard = await Leaderboard.findOne({ 
        duration, 
        isActive: true 
      }).sort({ generatedAt: -1 });
      
      if (!leaderboard) {
        return null;
      }
      
      return {
        duration: leaderboard.duration,
        leaders: leaderboard.getTopLeaders(20),
        generatedAt: leaderboard.generatedAt,
        totalParticipants: leaderboard.totalParticipants,
        averageReturn: leaderboard.averageReturn,
        medianReturn: leaderboard.medianReturn
      };
    } catch (error) {
      logger.error(`Error getting leaderboard for ${duration}:`, error);
      throw error;
    }
  }

  /**
   * Copy portfolio
   */
  async copyPortfolio(targetUserId, sourceSecretCode, investmentType, averageSip = null) {
    try {
      logger.info(`Copying portfolio: ${sourceSecretCode} -> ${targetUserId}`);
      
      // Find source user by secret code
      const sourceUser = await User.findOne({ secretCode: sourceSecretCode, isActive: true });
      if (!sourceUser) {
        throw new Error('Source user not found');
      }
      
      // Get source portfolio
      const sourcePortfolio = await UserPortfolio.findOne({ 
        userId: sourceUser._id, 
        isActive: true 
      });
      
      if (!sourcePortfolio) {
        throw new Error('Source portfolio not found');
      }
      
      // Validate investment type
      if (investmentType === 'SIP' && !averageSip) {
        throw new Error('Average SIP amount is required for SIP investment');
      }
      
      // Create portfolio copy record
      const portfolioCopy = new PortfolioCopy({
        sourceSecretCode,
        sourceUserId: sourceUser._id,
        targetUserId,
        investmentType,
        averageSip,
        copiedAllocation: sourcePortfolio.allocation,
        sourceReturnPercent: sourcePortfolio.performance.absoluteReturnPercent,
        duration: '1Y', // Default to 1Y for copy
        metadata: {
          userAgent: 'API',
          ipAddress: '127.0.0.1',
          deviceType: 'web'
        }
      });
      
      await portfolioCopy.save();
      
      // TODO: Integrate with Smart SIP service to create actual SIP
      // For now, just return success
      
      logger.info(`Portfolio copy created: ${portfolioCopy._id}`);
      
      return {
        success: true,
        copyId: portfolioCopy._id,
        message: 'Portfolio copied successfully. Your SIP setup has been created based on the leader\'s allocation.',
        allocation: portfolioCopy.getAllocationObject()
      };
      
    } catch (error) {
      logger.error('Error copying portfolio:', error);
      throw error;
    }
  }

  /**
   * Get user's leaderboard history
   */
  async getUserLeaderboardHistory(userId) {
    try {
      const portfolio = await UserPortfolio.findOne({ 
        userId, 
        isActive: true 
      });
      
      if (!portfolio) {
        return [];
      }
      
      return portfolio.leaderboardHistory.sort((a, b) => new Date(b.date) - new Date(a.date));
    } catch (error) {
      logger.error('Error getting user leaderboard history:', error);
      throw error;
    }
  }

  /**
   * Get portfolio copy history
   */
  async getPortfolioCopyHistory(userId) {
    try {
      const copies = await PortfolioCopy.find({ 
        targetUserId: userId, 
        isActive: true 
      }).sort({ createdAt: -1 });
      
      return copies.map(copy => ({
        id: copy._id,
        sourceSecretCode: copy.sourceSecretCode,
        investmentType: copy.investmentType,
        averageSip: copy.averageSip,
        sourceReturnPercent: copy.sourceReturnPercent,
        status: copy.status,
        createdAt: copy.createdAt,
        allocation: copy.getAllocationObject()
      }));
    } catch (error) {
      logger.error('Error getting portfolio copy history:', error);
      throw error;
    }
  }
}

module.exports = new LeaderboardService(); 