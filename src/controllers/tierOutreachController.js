const tierOutreachService = require('../services/tierOutreachService');
const logger = require('../utils/logger');

class TierOutreachController {
  /**
   * Determine user tier
   * @route POST /api/tier/determine
   */
  async determineUserTier(req, res) {
    try {
      const { userId } = req.user;
      const { location } = req.body;

      logger.info('Determining user tier', { userId, location });

      const result = await tierOutreachService.determineUserTier(userId, location);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'User tier determined successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in determineUserTier controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Create simplified onboarding
   * @route POST /api/tier/simplified-onboarding
   */
  async createSimplifiedOnboarding(req, res) {
    try {
      const { userId } = req.user;
      const { languageCode } = req.body;

      logger.info('Creating simplified onboarding', { userId, languageCode });

      const result = await tierOutreachService.createSimplifiedOnboarding(userId, languageCode);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Simplified onboarding created successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in createSimplifiedOnboarding controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Get vernacular content
   * @route GET /api/tier/vernacular-content
   */
  async getVernacularContent(req, res) {
    try {
      const { userId } = req.user;
      const { contentType, languageCode } = req.query;

      logger.info('Getting vernacular content', { userId, contentType, languageCode });

      if (!contentType || !languageCode) {
        return res.status(400).json({
          success: false,
          message: 'Content type and language code are required'
        });
      }

      const result = await tierOutreachService.getVernacularContent(userId, contentType, languageCode);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Vernacular content retrieved successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in getVernacularContent controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Create micro-investment options
   * @route POST /api/tier/micro-investments
   */
  async createMicroInvestmentOptions(req, res) {
    try {
      const { userId } = req.user;

      logger.info('Creating micro-investment options', { userId });

      const result = await tierOutreachService.createMicroInvestmentOptions(userId);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Micro-investment options created successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in createMicroInvestmentOptions controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Create community features
   * @route POST /api/tier/community-features
   */
  async createCommunityFeatures(req, res) {
    try {
      const { userId } = req.user;
      const { location } = req.body;

      logger.info('Creating community features', { userId, location });

      const result = await tierOutreachService.createCommunityFeatures(userId, location);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Community features created successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in createCommunityFeatures controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Create financial literacy program
   * @route POST /api/tier/financial-literacy
   */
  async createFinancialLiteracyProgram(req, res) {
    try {
      const { userId } = req.user;

      logger.info('Creating financial literacy program', { userId });

      const result = await tierOutreachService.createFinancialLiteracyProgram(userId);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Financial literacy program created successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in createFinancialLiteracyProgram controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Get tier-specific features
   * @route GET /api/tier/features
   */
  async getTierSpecificFeatures(req, res) {
    try {
      const { userId } = req.user;

      logger.info('Getting tier-specific features', { userId });

      const result = await tierOutreachService.getTierSpecificFeatures(userId);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Tier-specific features retrieved successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in getTierSpecificFeatures controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Get tier categories
   * @route GET /api/tier/categories
   */
  async getTierCategories(req, res) {
    try {
      logger.info('Getting tier categories');

      const tierCategories = {
        TIER_1: {
          name: 'Metro Cities',
          description: 'Major metropolitan areas with advanced features',
          cities: ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad'],
          features: ['full_features', 'advanced_analytics', 'premium_support'],
          minInvestment: 1000,
          literacyLevel: 'advanced'
        },
        TIER_2: {
          name: 'Tier 2 Cities',
          description: 'Emerging cities with simplified features',
          cities: ['Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam'],
          features: ['simplified_ui', 'basic_analytics', 'community_support'],
          minInvestment: 500,
          literacyLevel: 'intermediate'
        },
        TIER_3: {
          name: 'Tier 3 Cities',
          description: 'Smaller cities with basic features',
          cities: ['Patna', 'Vadodara', 'Ghaziabad', 'Ludhiana', 'Agra', 'Nashik', 'Faridabad', 'Meerut'],
          features: ['basic_ui', 'minimal_analytics', 'peer_support'],
          minInvestment: 100,
          literacyLevel: 'basic'
        },
        RURAL: {
          name: 'Rural Areas',
          description: 'Villages and small towns with ultra-simple features',
          cities: ['villages', 'small_towns', 'district_centers'],
          features: ['ultra_simple_ui', 'voice_interface', 'community_learning'],
          minInvestment: 50,
          literacyLevel: 'beginner'
        }
      };

      res.status(200).json({
        success: true,
        data: {
          tierCategories,
          totalTiers: Object.keys(tierCategories).length
        },
        message: 'Tier categories retrieved successfully'
      });
    } catch (error) {
      logger.error('Error in getTierCategories controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Get micro-investment options
   * @route GET /api/tier/micro-investment-options
   */
  async getMicroInvestmentOptions(req, res) {
    try {
      logger.info('Getting micro-investment options');

      const microInvestmentOptions = {
        DAILY_SIP: {
          name: 'Daily SIP',
          minAmount: 10,
          maxAmount: 1000,
          frequency: 'daily',
          description: 'Start with as little as ₹10 per day'
        },
        WEEKLY_SIP: {
          name: 'Weekly SIP',
          minAmount: 50,
          maxAmount: 5000,
          frequency: 'weekly',
          description: 'Invest ₹50-5000 every week'
        },
        GOAL_BASED: {
          name: 'Goal Based',
          minAmount: 100,
          maxAmount: 10000,
          frequency: 'monthly',
          description: 'Save for specific goals'
        },
        FESTIVAL_SAVINGS: {
          name: 'Festival Savings',
          minAmount: 25,
          maxAmount: 2000,
          frequency: 'monthly',
          description: 'Save for festivals and celebrations'
        }
      };

      res.status(200).json({
        success: true,
        data: {
          microInvestmentOptions,
          totalOptions: Object.keys(microInvestmentOptions).length
        },
        message: 'Micro-investment options retrieved successfully'
      });
    } catch (error) {
      logger.error('Error in getMicroInvestmentOptions controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Get financial literacy modules
   * @route GET /api/tier/financial-literacy-modules
   */
  async getFinancialLiteracyModules(req, res) {
    try {
      logger.info('Getting financial literacy modules');

      const financialLiteracyModules = {
        BASIC: {
          title: 'Basic Financial Literacy',
          modules: [
            'What is Money?',
            'Saving vs Spending',
            'Introduction to Banking',
            'Understanding Interest',
            'Basic Budgeting'
          ],
          duration: '2 weeks',
          difficulty: 'beginner'
        },
        INTERMEDIATE: {
          title: 'Investment Basics',
          modules: [
            'What are Mutual Funds?',
            'Types of Mutual Funds',
            'Risk and Returns',
            'SIP vs Lump Sum',
            'Tax Benefits'
          ],
          duration: '4 weeks',
          difficulty: 'intermediate'
        },
        ADVANCED: {
          title: 'Advanced Investment',
          modules: [
            'Portfolio Diversification',
            'Market Analysis',
            'Fund Selection',
            'Tax Planning',
            'Retirement Planning'
          ],
          duration: '6 weeks',
          difficulty: 'advanced'
        }
      };

      res.status(200).json({
        success: true,
        data: {
          financialLiteracyModules,
          totalLevels: Object.keys(financialLiteracyModules).length
        },
        message: 'Financial literacy modules retrieved successfully'
      });
    } catch (error) {
      logger.error('Error in getFinancialLiteracyModules controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }
}

module.exports = new TierOutreachController(); 