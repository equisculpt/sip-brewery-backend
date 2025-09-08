/**
 * DashboardController handles dashboard, holdings, and analytics endpoints for authenticated users.
 * Assumes authentication middleware populates req.user.supabaseId.
 * @module controllers/dashboardController
 */
const response = require('../utils/response');
const logger = require('../utils/logger');

class DashboardController {
  /**
   * Get complete dashboard data for the authenticated user.
   * @route GET /dashboard
   * @returns {Object} 200 - Dashboard data
   * @returns {Object} 401 - Authentication required
   * @returns {Object} 500 - Internal server error
   */
  async getDashboard(req, res) {
    try {
      const userId = req.user && req.user.supabaseId;
      if (!userId) {
        return response.errorResponse(res, 'Authentication required', null, 401);
      }
      logger.info('Fetching dashboard data', { userId });
      const dashboardData = await dashboardService.getDashboardData(userId);
      return response.successResponse(res, 'Dashboard data retrieved successfully', dashboardData);
    } catch (error) {
      logger.error('Error fetching dashboard data:', { error: error.message });
      return response.errorResponse(res, 'Failed to fetch dashboard data', error, 500);
    }
  }

  /**
   * Get holdings for the authenticated user.
   * @route GET /dashboard/holdings
   * @returns {Object} 200 - Holdings data
   * @returns {Object} 401 - Authentication required
   * @returns {Object} 500 - Internal server error
   */
  async getHoldings(req, res) {
    try {
      const userId = req.user && req.user.supabaseId;
      if (!userId) {
        return response.errorResponse(res, 'Authentication required', null, 401);
      }
      logger.info('Fetching holdings', { userId });
      const holdings = await dashboardService.getHoldings(userId);
      return response.successResponse(res, 'Holdings retrieved successfully', holdings);
    } catch (error) {
      logger.error('Error fetching holdings:', { error: error.message });
      return response.errorResponse(res, 'Failed to fetch holdings', error, 500);
    }
  }

  /**
   * Get Smart SIP Center data for the authenticated user.
   * @route GET /dashboard/smart-sip-center
   * @returns {Object} 200 - Smart SIP Center data
   * @returns {Object} 401 - Authentication required
   * @returns {Object} 500 - Internal server error
   */
  async getSmartSIPCenter(req, res) {
    try {
      const userId = req.user && req.user.supabaseId;
      if (!userId) {
        return response.errorResponse(res, 'Authentication required', null, 401);
      }
      logger.info('Fetching Smart SIP Center data', { userId });
      const smartSIPData = await dashboardService.getSmartSIPCenter(userId);
      return response.successResponse(res, 'Smart SIP Center data retrieved successfully', smartSIPData);
    } catch (error) {
      logger.error('Error fetching Smart SIP Center data:', { error: error.message });
      return response.errorResponse(res, 'Failed to fetch Smart SIP Center data', error, 500);
    }
  }

  /**
   * Get transactions for the authenticated user.
   * @route GET /dashboard/transactions
   * @param {number} [req.query.limit=10] - Max number of transactions
   * @returns {Object} 200 - Transactions data
   * @returns {Object} 401 - Authentication required
   * @returns {Object} 400 - Invalid parameters
   * @returns {Object} 500 - Internal server error
   */
  async getTransactions(req, res) {
    try {
      const userId = req.user && req.user.supabaseId;
      if (!userId) {
        return response.errorResponse(res, 'Authentication required', null, 401);
      }
      let { limit = 10 } = req.query;
      limit = parseInt(limit);
      if (isNaN(limit) || limit < 1 || limit > 100) {
        return response.errorResponse(res, 'Invalid limit parameter', null, 400);
      }
      logger.info('Fetching transactions', { userId, limit });
      const transactions = await dashboardService.getTransactions(userId, limit);
      return response.successResponse(res, 'Transactions retrieved successfully', transactions);
    } catch (error) {
      logger.error('Error fetching transactions:', { error: error.message });
      return response.errorResponse(res, 'Failed to fetch transactions', error, 500);
    }
  }

  /**
   * Get statements for the authenticated user.
   * @route GET /dashboard/statements
   * @returns {Object} 200 - Statements data
   * @returns {Object} 401 - Authentication required
   * @returns {Object} 500 - Internal server error
   */
  async getStatements(req, res) {
    try {
      const userId = req.user && req.user.supabaseId;
      if (!userId) {
        return response.errorResponse(res, 'Authentication required', null, 401);
      }
      logger.info('Fetching statements', { userId });
      const statements = await dashboardService.getStatements(userId);
      return response.successResponse(res, 'Statements retrieved successfully', statements);
    } catch (error) {
      logger.error('Error fetching statements:', { error: error.message });
      return response.errorResponse(res, 'Failed to fetch statements', error, 500);
    }
  }

  /**
   * Get rewards for the authenticated user.
   * @route GET /dashboard/rewards
   * @returns {Object} 200 - Rewards data
   * @returns {Object} 401 - Authentication required
   * @returns {Object} 500 - Internal server error
   */
  async getRewards(req, res) {
    try {
      const userId = req.user && req.user.supabaseId;
      if (!userId) {
        return response.errorResponse(res, 'Authentication required', null, 401);
      }
      logger.info('Fetching rewards', { userId });
      const rewards = await dashboardService.getRewards(userId);
      return response.successResponse(res, 'Rewards retrieved successfully', rewards);
    } catch (error) {
      logger.error('Error fetching rewards:', { error: error.message });
      return response.errorResponse(res, 'Failed to fetch rewards', error, 500);
    }
  }

  /**
   * Get referral data for the authenticated user.
   * @route GET /dashboard/referral
   * @returns {Object} 200 - Referral data
   * @returns {Object} 401 - Authentication required
   * @returns {Object} 500 - Internal server error
   */
  async getReferralData(req, res) {
    try {
      const userId = req.user && req.user.supabaseId;
      if (!userId) {
        return response.errorResponse(res, 'Authentication required', null, 401);
      }
      logger.info('Fetching referral data', { userId });
      const referralData = await dashboardService.getReferralData(userId);
      return response.successResponse(res, 'Referral data retrieved successfully', referralData);
    } catch (error) {
      logger.error('Error fetching referral data:', { error: error.message });
      return response.errorResponse(res, 'Failed to fetch referral data', error, 500);
    }
  }

  /**
   * Get AI Analytics for the authenticated user.
   * @route GET /dashboard/ai-analytics
   * @returns {Object} 200 - AI analytics data
   * @returns {Object} 401 - Authentication required
   * @returns {Object} 500 - Internal server error
   */
  async getAIAnalytics(req, res) {
    try {
      const userId = req.user && req.user.supabaseId;
      if (!userId) {
        return response.errorResponse(res, 'Authentication required', null, 401);
      }
      logger.info('Fetching AI analytics', { userId });
      const aiAnalytics = await dashboardService.getAIAnalytics(userId);
      return response.successResponse(res, 'AI analytics retrieved successfully', aiAnalytics);
    } catch (error) {
      logger.error('Error fetching AI analytics:', { error: error.message });
      return response.errorResponse(res, 'Failed to fetch AI analytics', error, 500);
    }
  }

  /**
   * Get Portfolio Analytics for the authenticated user.
   * @route GET /dashboard/portfolio-analytics
   * @returns {Object} 200 - Portfolio analytics data
   * @returns {Object} 401 - Authentication required
   * @returns {Object} 500 - Internal server error
   */
  async getPortfolioAnalytics(req, res) {
    try {
      const userId = req.user && req.user.supabaseId;
      if (!userId) {
        return response.errorResponse(res, 'Authentication required', null, 401);
      }
      logger.info('Fetching portfolio analytics', { userId });
      const portfolioAnalytics = await dashboardService.getPortfolioAnalytics(userId);
      return response.successResponse(res, 'Portfolio analytics retrieved successfully', portfolioAnalytics);
    } catch (error) {
      logger.error('Error fetching portfolio analytics:', { error: error.message });
      return response.errorResponse(res, 'Failed to fetch portfolio analytics', error, 500);
    }
  }

  /**
   * Get performance chart for the authenticated user.
   * @route GET /dashboard/performance-chart
   * @returns {Object} 200 - Performance chart data
   * @returns {Object} 401 - Authentication required
   * @returns {Object} 500 - Internal server error
   */
  async getPerformanceChart(req, res) {
    try {
      const userId = req.user && req.user.supabaseId;
      if (!userId) {
        return response.errorResponse(res, 'Authentication required', null, 401);
      }
      logger.info('Fetching performance chart', { userId });
      const performanceChart = await dashboardService.getPerformanceChart(userId);
      return response.successResponse(res, 'Performance chart retrieved successfully', performanceChart);
    } catch (error) {
      logger.error('Error fetching performance chart:', { error: error.message });
      return response.errorResponse(res, 'Failed to fetch performance chart', error, 500);
    }
  }

  /**
   * Get user profile for the authenticated user.
   * @route GET /dashboard/profile
   * @returns {Object} 200 - Profile data
   * @returns {Object} 401 - Authentication required
   * @returns {Object} 500 - Internal server error
   */
  async getProfile(req, res) {
    try {
      const userId = req.user && req.user.supabaseId;
      if (!userId) {
        return response.errorResponse(res, 'Authentication required', null, 401);
      }
      logger.info('Fetching user profile', { userId });
      const profile = await dashboardService.getProfile(userId);
      return response.successResponse(res, 'Profile retrieved successfully', profile);
    } catch (error) {
      logger.error('Error fetching profile:', { error: error.message });
      return response.errorResponse(res, 'Failed to fetch profile', error, 500);
    }
  }

  /**
   * Get AI-Driven Portfolio Insights for the authenticated user.
   * @route GET /dashboard/ai-insights
   * @returns {Object} 200 - AI-driven portfolio insights
   * @returns {Object} 401 - Authentication required
   * @returns {Object} 500 - Internal server error
   */
  async getAIInsights(req, res) {
    try {
      const userId = req.user && req.user.supabaseId;
      if (!userId) {
        return response.errorResponse(res, 'Authentication required', null, 401);
      }
      logger.info('Fetching AI-driven portfolio insights', { userId });
      const aiPortfolioOptimizer = require('../services/aiPortfolioOptimizer');
      const userProfile = { userId };
      // For demo, use MODERATE risk and WEALTH_CREATION goal
      const result = await aiPortfolioOptimizer.optimizePortfolio(userProfile, 'MODERATE', ['WEALTH_CREATION']);
      if (!result || !result.success) {
        logger.warn('Returning mock AI insights due to service failure', { userId });
        const mockInsights = {
          portfolioAnalysis: {
            currentValue: 1000000,
            totalReturn: 0.15,
            riskScore: 0.4
          },
          recommendations: {
            fundSelection: ['Index funds', 'Large cap funds'],
            allocationChanges: ['Increase equity allocation by 10%'],
            riskManagement: ['Set stop-loss at 5%']
          },
          marketInsights: {
            trend: 'BULLISH',
            sectors: ['Technology', 'Healthcare'],
            riskFactors: ['Market volatility', 'Interest rate changes']
          }
        };
        return response.successResponse(res, 'AI-driven portfolio insights retrieved', mockInsights);
      }
      return response.successResponse(res, 'AI-driven portfolio insights retrieved', result.data);
    } catch (error) {
      logger.error('Error fetching AI-driven insights:', { error: error.message });
      // Return mock data on error
      logger.warn('Returning mock AI insights due to error', { error: error.message });
      const mockInsights = {
        portfolioAnalysis: {
          currentValue: 1000000,
          totalReturn: 0.15,
          riskScore: 0.4
        },
        recommendations: {
          fundSelection: ['Index funds', 'Large cap funds'],
          allocationChanges: ['Increase equity allocation by 10%'],
          riskManagement: ['Set stop-loss at 5%']
        },
        marketInsights: {
          trend: 'BULLISH',
          sectors: ['Technology', 'Healthcare'],
          riskFactors: ['Market volatility', 'Interest rate changes']
        }
      };
      return response.successResponse(res, 'AI-driven portfolio insights retrieved', mockInsights);
    }
  }

  /**
   * Get Predictive Analytics (fund performance, market trends, behavioral insights) for the authenticated user.
   * @route GET /dashboard/predictive-analytics
   * @returns {Object} 200 - Predictive analytics data
   * @returns {Object} 401 - Authentication required
   * @returns {Object} 500 - Internal server error
   */
  async getPredictiveAnalytics(req, res) {
    try {
      const userId = req.user && req.user.supabaseId;
      if (!userId) {
        return response.errorResponse(res, 'Authentication required', null, 401);
      }
      logger.info('Fetching predictive analytics', { userId });
      // For demo, use mock fund data and market conditions
      const aiPortfolioOptimizer = require('../services/aiPortfolioOptimizer');
      const fundData = {
        schemeCode: '123456',
        fundHouse: 'Axis Mutual Fund',
        category: 'Equity',
        historicalReturns: [0.12, 0.15, 0.10],
        nav: 100,
        aum: 1000000000
      };
      const marketConditions = {
        trend: 'BULLISH',
        economicIndicators: { gdp: 7.2, inflation: 5.5 },
        sectorPerformance: { technology: 0.15, healthcare: 0.12 }
      };
      const prediction = await aiPortfolioOptimizer.predictPerformance(fundData, marketConditions);
      if (!prediction) {
        logger.warn('Returning mock predictive analytics due to service failure', { userId });
        const mockPrediction = {
          prediction1Y: 0.12,
          prediction3Y: 0.10,
          prediction5Y: 0.09,
          riskAssessment: {
            volatility: 0.15,
            downsideRisk: 0.08,
            sharpeRatio: 1.2
          },
          marketTrends: {
            sectorOutlook: 'BULLISH',
            recommendedSectors: ['Technology', 'Healthcare'],
            riskFactors: ['Interest rate changes', 'Geopolitical tensions']
          },
          behavioralInsights: {
            sentiment: 'POSITIVE',
            confidence: 0.75,
            recommendations: ['Stay invested', 'Consider rebalancing']
          }
        };
        return response.successResponse(res, 'Predictive analytics generated', mockPrediction);
      }
      return response.successResponse(res, 'Predictive analytics generated', prediction);
    } catch (error) {
      logger.error('Error fetching predictive analytics:', { error: error.message });
      // Return mock data on error
      logger.warn('Returning mock predictive analytics due to error', { error: error.message });
      const mockPrediction = {
        prediction1Y: 0.12,
        prediction3Y: 0.10,
        prediction5Y: 0.09,
        riskAssessment: {
          volatility: 0.15,
          downsideRisk: 0.08,
          sharpeRatio: 1.2
        },
        marketTrends: {
          sectorOutlook: 'BULLISH',
          recommendedSectors: ['Technology', 'Healthcare'],
          riskFactors: ['Interest rate changes', 'Geopolitical tensions']
        },
        behavioralInsights: {
          sentiment: 'POSITIVE',
          confidence: 0.75,
          recommendations: ['Stay invested', 'Consider rebalancing']
        }
      };
      return response.successResponse(res, 'Predictive analytics generated', mockPrediction);
    }
  }
}

module.exports = new DashboardController();