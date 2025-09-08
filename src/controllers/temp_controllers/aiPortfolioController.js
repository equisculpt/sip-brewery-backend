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

  /**
   * Get portfolio analysis
   */
  async getPortfolioAnalysis(req, res) {
    try {
      const { userId } = req.user;

      // Get user portfolio
      const portfolio = await UserPortfolio.findOne({ userId });
      if (!portfolio) {
        return errorResponse(res, 'Portfolio not found', null, 404);
      }

      // Get holdings
      const holdings = await Holding.find({ userId, isActive: true });

      // Calculate analysis
      const analysis = {
        totalValue: portfolio.totalValue,
        totalInvested: portfolio.totalInvested,
        currentGain: portfolio.totalValue - portfolio.totalInvested,
        gainPercentage: ((portfolio.totalValue - portfolio.totalInvested) / portfolio.totalInvested) * 100,
        assetAllocation: this.calculateAssetAllocation(holdings),
        performance: {
          xirr1M: portfolio.xirr1M || 0,
          xirr3M: portfolio.xirr3M || 0,
          xirr6M: portfolio.xirr6M || 0,
          xirr1Y: portfolio.xirr1Y || 0,
          xirr3Y: portfolio.xirr3Y || 0
        },
        riskMetrics: await this.calculateRiskMetrics(holdings),
        diversification: await this.calculateDiversification(holdings)
      };

      return successResponse(res, 'Portfolio analysis retrieved', analysis);
    } catch (error) {
      logger.error('Portfolio analysis error', { error: error.message });
      return errorResponse(res, 'Failed to get portfolio analysis', { error: error.message });
    }
  }

  /**
   * Get AI recommendations
   */
  async getRecommendations(req, res) {
    try {
      const { userId } = req.user;
      const { riskTolerance = 'MODERATE' } = req.query;

      // Get user profile
      const user = await User.findById(userId);
      if (!user) {
        return errorResponse(res, 'User not found', null, 404);
      }

      // Get portfolio
      const portfolio = await UserPortfolio.findOne({ userId });
      if (!portfolio) {
        return errorResponse(res, 'Portfolio not found', null, 404);
      }

      // Create user profile
      const userProfile = {
        userId,
        age: user.age || 30,
        income: user.income || 500000,
        taxSlab: user.taxSlab || 0.3,
        investmentHorizon: user.investmentHorizon || 'LONG_TERM'
      };

      // Get portfolio analysis
      let portfolioAnalysis;
      try {
        portfolioAnalysis = await aiPortfolioOptimizer.analyzeCurrentPortfolio(portfolio);
      } catch (serviceError) {
        logger.error('AI Portfolio Optimizer analysis error', { error: serviceError.message });
        return errorResponse(res, 'AI service temporarily unavailable', null, 503);
      }

      // Generate recommendations
      let recommendations;
      try {
        recommendations = await aiPortfolioOptimizer.generateRecommendations(
          userProfile,
          riskTolerance,
          ['WEALTH_CREATION', 'RETIREMENT'], // Default goals
          portfolioAnalysis
        );
      } catch (serviceError) {
        logger.error('AI Portfolio Optimizer recommendations error', { error: serviceError.message });
        return errorResponse(res, 'AI service temporarily unavailable', null, 503);
      }

      return successResponse(res, 'Recommendations generated', recommendations);
    } catch (error) {
      logger.error('Get recommendations error', { error: error.message });
      return errorResponse(res, 'Failed to get recommendations', { error: error.message });
    }
  }

  /**
   * Predict fund performance
   */
  async predictFundPerformance(req, res) {
    try {
      const { schemeCode, fundHouse, category, historicalReturns, nav, aum } = req.body;

      // Validate input
      if (!schemeCode || !fundHouse || !category) {
        return errorResponse(res, 'Missing required fields: schemeCode, fundHouse, category', null, 400);
      }

      // Mock market conditions (in real implementation, get from market data service)
      const marketConditions = {
        trend: 'BULLISH',
        economicIndicators: {
          gdp: 7.2,
          inflation: 5.5,
          interestRate: 6.5
        },
        sectorPerformance: {
          technology: 0.15,
          healthcare: 0.12,
          finance: 0.08
        }
      };

      const fundData = {
        schemeCode,
        fundHouse,
        category,
        historicalReturns: historicalReturns || [0.12, 0.15, 0.10],
        nav: nav || 100,
        aum: aum || 1000000000
      };

      // Get AI prediction
      let prediction;
      try {
        prediction = await aiPortfolioOptimizer.predictPerformance(fundData, marketConditions);
      } catch (serviceError) {
        logger.error('AI Portfolio Optimizer prediction error', { error: serviceError.message });
        return errorResponse(res, 'AI service temporarily unavailable', null, 503);
      }

      return successResponse(res, 'Performance prediction generated', prediction);
    } catch (error) {
      logger.error('Fund performance prediction error', { error: error.message });
      return errorResponse(res, 'Failed to predict fund performance', { error: error.message });
    }
  }

  /**
   * Get risk assessment
   */
  async getRiskAssessment(req, res) {
    try {
      const { userId } = req.user;
      const { allocation } = req.body;

      if (!allocation) {
        return errorResponse(res, 'Missing allocation data', null, 400);
      }

      // Get user profile
      const user = await User.findById(userId);
      if (!user) {
        return errorResponse(res, 'User not found', null, 404);
      }

      // Calculate risk assessment
      const riskAssessment = await aiPortfolioOptimizer.assessRisk(allocation, {
        age: user.age || 30,
        income: user.income || 500000,
        investmentHorizon: user.investmentHorizon || 'LONG_TERM'
      });

      return successResponse(res, 'Risk assessment completed', riskAssessment);
    } catch (error) {
      logger.error('Risk assessment error', { error: error.message });
      return errorResponse(res, 'Failed to assess risk', { error: error.message });
    }
  }

  /**
   * Get tax strategies
   */
  async getTaxStrategies(req, res) {
    try {
      const { userId } = req.user;
      const { portfolioValue, annualIncome } = req.body;

      if (!portfolioValue || !annualIncome) {
        return errorResponse(res, 'Missing required fields: portfolioValue, annualIncome', null, 400);
      }

      // Get user profile
      const user = await User.findById(userId);
      if (!user) {
        return errorResponse(res, 'User not found', null, 404);
      }

      // Get tax strategies
      const taxStrategies = await aiPortfolioOptimizer.generateTaxStrategies({
        portfolioValue,
        annualIncome,
        taxSlab: user.taxSlab || 0.3,
        age: user.age || 30,
        investmentHorizon: user.investmentHorizon || 'LONG_TERM'
      });

      return successResponse(res, 'Tax strategies generated', taxStrategies);
    } catch (error) {
      logger.error('Tax strategies error', { error: error.message });
      return errorResponse(res, 'Failed to generate tax strategies', { error: error.message });
    }
  }

  /**
   * Get expected returns
   */
  async getExpectedReturns(req, res) {
    try {
      const { userId } = req.user;
      const { allocation, timeHorizon } = req.body;

      if (!allocation || !timeHorizon) {
        return errorResponse(res, 'Missing required fields: allocation, timeHorizon', null, 400);
      }

      // Get expected returns
      const expectedReturns = await aiPortfolioOptimizer.calculateExpectedReturns(allocation, timeHorizon);

      return successResponse(res, 'Expected returns calculated', expectedReturns);
    } catch (error) {
      logger.error('Expected returns error', { error: error.message });
      return errorResponse(res, 'Failed to calculate expected returns', { error: error.message });
    }
  }

  // Helper methods
  calculateAssetAllocation(holdings) {
    const totalValue = holdings.reduce((sum, holding) => sum + holding.currentValue, 0);
    const allocation = {};

    holdings.forEach(holding => {
      const category = holding.fundCategory || 'others';
      allocation[category] = (allocation[category] || 0) + (holding.currentValue / totalValue);
    });

    return allocation;
  }

  async calculateRiskMetrics(holdings) {
    const totalValue = holdings.reduce((sum, holding) => sum + holding.currentValue, 0);
    const weightedRisk = holdings.reduce((sum, holding) => {
      const riskWeight = this.getFundRiskWeight(holding.fundCategory);
      return sum + (holding.currentValue / totalValue) * riskWeight;
    }, 0);

    return {
      overallRisk: weightedRisk,
      maxDrawdown: 0.15, // Simplified calculation
      sharpeRatio: 1.2,  // Simplified calculation
      beta: 0.9          // Simplified calculation
    };
  }

  async calculateDiversification(holdings) {
    const categories = new Set(holdings.map(h => h.fundCategory));
    const fundHouses = new Set(holdings.map(h => h.fundHouse));
    
    return {
      categoryDiversification: categories.size / 10,
      fundHouseDiversification: fundHouses.size / 20,
      overallScore: (categories.size / 10 + fundHouses.size / 20) / 2
    };
  }

  getFundRiskWeight(category) {
    const riskWeights = {
      'Equity': 0.8,
      'Debt': 0.2,
      'Hybrid': 0.5,
      'Liquid': 0.1,
      'others': 0.6
    };
    return riskWeights[category] || 0.6;
  }
}

module.exports = new AIPortfolioController(); 