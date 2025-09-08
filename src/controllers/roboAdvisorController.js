const logger = require('../utils/logger');
const roboAdvisor = require('../services/roboAdvisor');
const { authenticateUser } = require('../middleware/auth');

class RoboAdvisorController {
  /**
   * Perform portfolio review
   */
  async performPortfolioReview(req, res) {
    try {
      logger.info('Portfolio review request received');

      const { userId } = req.user;
      const { reviewType } = req.body;

      const result = await roboAdvisor.performPortfolioReview(userId, reviewType);

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'Portfolio review completed successfully',
          data: result.data
        });
      } else {
        res.status(500).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Portfolio review controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to perform portfolio review',
        error: error.message
      });
    }
  }

  /**
   * Generate switch recommendations
   */
  async generateSwitchRecommendations(req, res) {
    try {
      logger.info('Switch recommendations request received');

      const { userId } = req.user;
      const { fundName } = req.body;

      const result = await roboAdvisor.generateSwitchRecommendations(userId, fundName);

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'Switch recommendations generated successfully',
          data: result.data
        });
      } else {
        res.status(500).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Switch recommendations controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to generate switch recommendations',
        error: error.message
      });
    }
  }

  /**
   * Check tax harvesting opportunities
   */
  async checkTaxHarvestingOpportunities(req, res) {
    try {
      logger.info('Tax harvesting opportunities request received');

      const { userId } = req.user;

      // Get user holdings
      const { Holding } = require('../models');
      const holdings = await Holding.find({ userId, isActive: true });

      const opportunities = await roboAdvisor.checkTaxHarvestingOpportunities(holdings);

      res.status(200).json({
        success: true,
        message: 'Tax harvesting opportunities retrieved successfully',
        data: {
          opportunities,
          totalOpportunities: opportunities.length,
          highPriorityOpportunities: opportunities.filter(o => o.priority === 'HIGH').length
        }
      });
    } catch (error) {
      logger.error('Tax harvesting opportunities controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to check tax harvesting opportunities',
        error: error.message
      });
    }
  }

  /**
   * Generate STP/SWP plans
   */
  async generateSTPSWPPlan(req, res) {
    try {
      logger.info('STP/SWP plan generation request received');

      const { userId } = req.user;

      const result = await roboAdvisor.generateSTPSWPPlan(userId);

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'STP/SWP plans generated successfully',
          data: result.data
        });
      } else {
        res.status(500).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('STP/SWP plan generation controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to generate STP/SWP plans',
        error: error.message
      });
    }
  }

  /**
   * Check SIP goal deviations
   */
  async checkSIPGoalDeviations(req, res) {
    try {
      logger.info('SIP goal deviations check request received');

      const { userId } = req.user;

      const result = await roboAdvisor.checkSIPGoalDeviations(userId);

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'SIP goal deviations checked successfully',
          data: result.data
        });
      } else {
        res.status(500).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('SIP goal deviations controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to check SIP goal deviations',
        error: error.message
      });
    }
  }

  /**
   * Get robo-advisory suggestions
   */
  async getRoboAdvisorySuggestions(req, res) {
    try {
      logger.info('Robo-advisory suggestions request received');

      const { userId } = req.user;
      const { suggestionType } = req.query;

      const result = await roboAdvisor.getRoboAdvisorySuggestions(userId, suggestionType);

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'Robo-advisory suggestions retrieved successfully',
          data: result.data
        });
      } else {
        res.status(500).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Robo-advisory suggestions controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get robo-advisory suggestions',
        error: error.message
      });
    }
  }

  /**
   * Get portfolio health summary
   */
  async getPortfolioHealthSummary(req, res) {
    try {
      logger.info('Portfolio health summary request received');

      const { userId } = req.user;

      // Perform portfolio review to get health data
      const review = await roboAdvisor.performPortfolioReview(userId);

      if (!review.success) {
        return res.status(500).json({
          success: false,
          message: 'Failed to get portfolio health data',
          error: review.error
        });
      }

      const healthSummary = {
        status: review.data.portfolioHealth.status,
        score: review.data.portfolioHealth.score,
        metrics: review.data.portfolioHealth.metrics,
        recommendations: review.data.recommendations.filter(r => r.type === 'RISK_REDUCE'),
        alerts: review.data.alerts,
        lastReviewDate: review.data.reviewDate
      };

      res.status(200).json({
        success: true,
        message: 'Portfolio health summary retrieved successfully',
        data: healthSummary
      });
    } catch (error) {
      logger.error('Portfolio health summary controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get portfolio health summary',
        error: error.message
      });
    }
  }

  /**
   * Get rebalancing recommendations
   */
  async getRebalancingRecommendations(req, res) {
    try {
      logger.info('Rebalancing recommendations request received');

      const { userId } = req.user;

      // Perform portfolio review to get rebalancing data
      const review = await roboAdvisor.performPortfolioReview(userId);

      if (!review.success) {
        return res.status(500).json({
          success: false,
          message: 'Failed to get rebalancing data',
          error: review.error
        });
      }

      const rebalancingData = {
        needs: review.data.rebalancingNeeds,
        recommendations: review.data.recommendations.filter(r => r.type === 'REBALANCE'),
        totalNeeds: review.data.rebalancingNeeds.length,
        highPriorityNeeds: review.data.rebalancingNeeds.filter(n => n.priority === 'HIGH').length,
        lastReviewDate: review.data.reviewDate
      };

      res.status(200).json({
        success: true,
        message: 'Rebalancing recommendations retrieved successfully',
        data: rebalancingData
      });
    } catch (error) {
      logger.error('Rebalancing recommendations controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get rebalancing recommendations',
        error: error.message
      });
    }
  }

  /**
   * Get goal progress summary
   */
  async getGoalProgressSummary(req, res) {
    try {
      logger.info('Goal progress summary request received');

      const { userId } = req.user;

      // Perform portfolio review to get goal data
      const review = await roboAdvisor.performPortfolioReview(userId);

      if (!review.success) {
        return res.status(500).json({
          success: false,
          message: 'Failed to get goal progress data',
          error: review.error
        });
      }

      const goalSummary = {
        goals: review.data.goalProgress,
        atRiskGoals: review.data.goalProgress.filter(g => g.status === 'AT_RISK'),
        onTrackGoals: review.data.goalProgress.filter(g => g.status === 'ON_TRACK'),
        completedGoals: review.data.goalProgress.filter(g => g.status === 'COMPLETED'),
        totalGoals: review.data.goalProgress.length,
        averageProgress: this.calculateAverageProgress(review.data.goalProgress),
        lastReviewDate: review.data.reviewDate
      };

      res.status(200).json({
        success: true,
        message: 'Goal progress summary retrieved successfully',
        data: goalSummary
      });
    } catch (error) {
      logger.error('Goal progress summary controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get goal progress summary',
        error: error.message
      });
    }
  }

  /**
   * Get risk assessment summary
   */
  async getRiskAssessmentSummary(req, res) {
    try {
      logger.info('Risk assessment summary request received');

      const { userId } = req.user;

      // Perform portfolio review to get risk data
      const review = await roboAdvisor.performPortfolioReview(userId);

      if (!review.success) {
        return res.status(500).json({
          success: false,
          message: 'Failed to get risk assessment data',
          error: review.error
        });
      }

      const riskSummary = {
        level: review.data.riskAssessment.level,
        metrics: review.data.riskAssessment.metrics,
        recommendation: review.data.riskAssessment.recommendation,
        riskAlerts: review.data.alerts.filter(a => a.type === 'HIGH_RISK'),
        lastAssessmentDate: review.data.reviewDate
      };

      res.status(200).json({
        success: true,
        message: 'Risk assessment summary retrieved successfully',
        data: riskSummary
      });
    } catch (error) {
      logger.error('Risk assessment summary controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get risk assessment summary',
        error: error.message
      });
    }
  }

  /**
   * Get market opportunities
   */
  async getMarketOpportunities(req, res) {
    try {
      logger.info('Market opportunities request received');

      const { userId } = req.user;

      // Perform portfolio review to get market data
      const review = await roboAdvisor.performPortfolioReview(userId);

      if (!review.success) {
        return res.status(500).json({
          success: false,
          message: 'Failed to get market opportunities data',
          error: review.error
        });
      }

      const opportunitiesSummary = {
        opportunities: review.data.marketOpportunities,
        sectorOpportunities: review.data.marketOpportunities.filter(o => o.type === 'SECTOR_OPPORTUNITY'),
        marketOpportunities: review.data.marketOpportunities.filter(o => o.type === 'MARKET_OPPORTUNITY'),
        totalOpportunities: review.data.marketOpportunities.length,
        highPriorityOpportunities: review.data.marketOpportunities.filter(o => o.priority === 'HIGH').length,
        lastReviewDate: review.data.reviewDate
      };

      res.status(200).json({
        success: true,
        message: 'Market opportunities retrieved successfully',
        data: opportunitiesSummary
      });
    } catch (error) {
      logger.error('Market opportunities controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get market opportunities',
        error: error.message
      });
    }
  }

  /**
   * Get comprehensive robo-advisory dashboard
   */
  async getRoboAdvisoryDashboard(req, res) {
    try {
      logger.info('Robo-advisory dashboard request received');

      const { userId } = req.user;

      // Get all robo-advisory data
      const [
        portfolioReview,
        switchRecommendations,
        taxOpportunities,
        stpswpPlans,
        sipDeviations
      ] = await Promise.all([
        roboAdvisor.performPortfolioReview(userId),
        roboAdvisor.generateSwitchRecommendations(userId),
        roboAdvisor.checkTaxHarvestingOpportunities([]), // Will be populated with actual holdings
        roboAdvisor.generateSTPSWPPlan(userId),
        roboAdvisor.checkSIPGoalDeviations(userId)
      ]);

      const dashboardData = {
        portfolioHealth: portfolioReview.success ? portfolioReview.data.portfolioHealth : null,
        switchRecommendations: switchRecommendations.success ? switchRecommendations.data : null,
        taxOpportunities: taxOpportunities,
        stpswpPlans: stpswpPlans.success ? stpswpPlans.data : null,
        sipDeviations: sipDeviations.success ? sipDeviations.data : null,
        recommendations: portfolioReview.success ? portfolioReview.data.recommendations : [],
        alerts: portfolioReview.success ? portfolioReview.data.alerts : [],
        lastUpdated: new Date().toISOString()
      };

      res.status(200).json({
        success: true,
        message: 'Robo-advisory dashboard data retrieved successfully',
        data: dashboardData
      });
    } catch (error) {
      logger.error('Robo-advisory dashboard controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get robo-advisory dashboard data',
        error: error.message
      });
    }
  }

  /**
   * Execute robo-advisory action
   */
  async executeRoboAdvisoryAction(req, res) {
    try {
      logger.info('Robo-advisory action execution request received');

      const { userId } = req.user;
      const { actionType, actionData } = req.body;

      let result;

      switch (actionType) {
        case 'PORTFOLIO_REVIEW':
          result = await roboAdvisor.performPortfolioReview(userId);
          break;
        case 'SWITCH_RECOMMENDATION':
          result = await roboAdvisor.generateSwitchRecommendations(userId, actionData.fundName);
          break;
        case 'TAX_HARVESTING':
          // Get holdings and check tax opportunities
          const { Holding } = require('../models');
          const holdings = await Holding.find({ userId, isActive: true });
          result = { success: true, data: await roboAdvisor.checkTaxHarvestingOpportunities(holdings) };
          break;
        case 'STP_SWP_PLAN':
          result = await roboAdvisor.generateSTPSWPPlan(userId);
          break;
        case 'SIP_DEVIATION_CHECK':
          result = await roboAdvisor.checkSIPGoalDeviations(userId);
          break;
        default:
          return res.status(400).json({
            success: false,
            message: 'Invalid action type',
            error: 'Unsupported action type'
          });
      }

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'Robo-advisory action executed successfully',
          data: result.data
        });
      } else {
        res.status(500).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Robo-advisory action execution controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to execute robo-advisory action',
        error: error.message
      });
    }
  }

  // Helper methods
  calculateAverageProgress(goalProgress) {
    try {
      if (goalProgress.length === 0) return 0;
      
      const totalProgress = goalProgress.reduce((sum, goal) => sum + goal.progressPercentage, 0);
      return totalProgress / goalProgress.length;
    } catch (error) {
      logger.error('Average progress calculation failed', { error: error.message });
      return 0;
    }
  }
}

module.exports = new RoboAdvisorController(); 