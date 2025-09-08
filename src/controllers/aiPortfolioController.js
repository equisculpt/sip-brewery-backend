const aiPortfolioOptimizer = require('../services/aiPortfolioOptimizer');
const logger = require('../utils/logger');
const { successResponse, errorResponse } = require('../utils/response');
const { User, UserPortfolio, Holding } = require('../models');

class AIPortfolioController {
  /**
   * Optimize user portfolio using AI
   */
  async optimizePortfolio(req, res) {
    try {
      const { userId } = req.user;
      const { riskTolerance, goals } = req.body;

      // Validate input
      if (!riskTolerance || !goals || !Array.isArray(goals)) {
        return errorResponse(res, 'Missing required fields: riskTolerance and goals array', null, 400);
      }

      // Get user profile
      const user = await User.findById(userId);
      if (!user) {
        return errorResponse(res, 'User not found', null, 404);
      }

      // Check if user has portfolio
      const portfolio = await UserPortfolio.findOne({ userId });
      if (!portfolio) {
        return errorResponse(res, 'Portfolio not found. Please create a portfolio first.', null, 404);
      }

      // Create user profile for optimization
      const userProfile = {
        userId,
        age: user.age || 30,
        income: user.income || 500000,
        taxSlab: user.taxSlab || 0.3,
        investmentHorizon: user.investmentHorizon || 'LONG_TERM'
      };

      // Run AI optimization
      let result;
      try {
        result = await aiPortfolioOptimizer.optimizePortfolio(userProfile, riskTolerance, goals);
      } catch (serviceError) {
        logger.error('AI Portfolio Optimizer service error', { error: serviceError.message });
        return errorResponse(res, 'AI service temporarily unavailable', null, 503);
      }

      if (!result || !result.success) {
        return errorResponse(res, result?.message || 'Optimization failed', null, 500);
      }

      logger.info('Portfolio optimization completed', { userId, riskTolerance });

      return successResponse(res, 'Portfolio optimization completed', result.data);
    } catch (error) {
      logger.error('Portfolio optimization error', { error: error.message });
      return errorResponse(res, 'Failed to optimize portfolio', { error: error.message });
    }
  }
}

module.exports = new AIPortfolioController();
