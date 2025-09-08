const socialInvestingService = require('../services/socialInvestingService');
const logger = require('../utils/logger');

class SocialInvestingController {
  /**
   * Create social profile
   * @route POST /api/social/profile
   */
  async createSocialProfile(req, res) {
    try {
      const { userId } = req.user;

      logger.info('Creating social profile', { userId });

      const result = await socialInvestingService.createSocialProfile(userId);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Social profile created successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in createSocialProfile controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Follow another investor
   * @route POST /api/social/follow
   */
  async followInvestor(req, res) {
    try {
      const { userId } = req.user;
      const { followingId } = req.body;

      logger.info('Following investor', { userId, followingId });

      if (!followingId) {
        return res.status(400).json({
          success: false,
          message: 'Following ID is required'
        });
      }

      const result = await socialInvestingService.followInvestor(userId, followingId);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Successfully followed investor'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in followInvestor controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Share portfolio performance
   * @route POST /api/social/share-portfolio
   */
  async sharePortfolioPerformance(req, res) {
    try {
      const { userId } = req.user;
      const { shareType } = req.body;

      logger.info('Sharing portfolio performance', { userId, shareType });

      const result = await socialInvestingService.sharePortfolioPerformance(userId, shareType);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Portfolio performance shared successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in sharePortfolioPerformance controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Create investment challenge
   * @route POST /api/social/challenge
   */
  async createInvestmentChallenge(req, res) {
    try {
      const { userId } = req.user;
      const { challengeType } = req.body;

      logger.info('Creating investment challenge', { userId, challengeType });

      if (!challengeType) {
        return res.status(400).json({
          success: false,
          message: 'Challenge type is required'
        });
      }

      const result = await socialInvestingService.createInvestmentChallenge(userId, challengeType);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Investment challenge created successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in createInvestmentChallenge controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Award points to user
   * @route POST /api/social/award-points
   */
  async awardPoints(req, res) {
    try {
      const { userId } = req.user;
      const { points, reason } = req.body;

      logger.info('Awarding points', { userId, points, reason });

      if (!points || !reason) {
        return res.status(400).json({
          success: false,
          message: 'Points and reason are required'
        });
      }

      const result = await socialInvestingService.awardPoints(userId, points, reason);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Points awarded successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in awardPoints controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Get leaderboard
   * @route GET /api/social/leaderboard
   */
  async getLeaderboard(req, res) {
    try {
      const { category, timeFrame } = req.query;

      logger.info('Getting leaderboard', { category, timeFrame });

      const result = await socialInvestingService.getLeaderboard(category, timeFrame);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Leaderboard retrieved successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in getLeaderboard controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Create educational game
   * @route POST /api/social/educational-game
   */
  async createEducationalGame(req, res) {
    try {
      const { userId } = req.user;
      const { gameType } = req.body;

      logger.info('Creating educational game', { userId, gameType });

      if (!gameType) {
        return res.status(400).json({
          success: false,
          message: 'Game type is required'
        });
      }

      const result = await socialInvestingService.createEducationalGame(userId, gameType);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Educational game created successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in createEducationalGame controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Submit game answer
   * @route POST /api/social/game-answer
   */
  async submitGameAnswer(req, res) {
    try {
      const { userId } = req.user;
      const { gameId, answer } = req.body;

      logger.info('Submitting game answer', { userId, gameId, answer });

      if (!gameId || answer === undefined) {
        return res.status(400).json({
          success: false,
          message: 'Game ID and answer are required'
        });
      }

      const result = await socialInvestingService.submitGameAnswer(userId, gameId, answer);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Game answer submitted successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in submitGameAnswer controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Get user achievements
   * @route GET /api/social/achievements
   */
  async getUserAchievements(req, res) {
    try {
      const { userId } = req.user;

      logger.info('Getting user achievements', { userId });

      const result = await socialInvestingService.getUserAchievements(userId);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'User achievements retrieved successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in getUserAchievements controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Create community event
   * @route POST /api/social/community-event
   */
  async createCommunityEvent(req, res) {
    try {
      const { userId } = req.user;
      const { eventData } = req.body;

      logger.info('Creating community event', { userId, eventData });

      if (!eventData || !eventData.title || !eventData.description) {
        return res.status(400).json({
          success: false,
          message: 'Event data with title and description is required'
        });
      }

      const result = await socialInvestingService.createCommunityEvent(userId, eventData);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Community event created successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in createCommunityEvent controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Get social features
   * @route GET /api/social/features
   */
  async getSocialFeatures(req, res) {
    try {
      logger.info('Getting social features');

      const socialFeatures = {
        SOCIAL_TRADING: {
          name: 'Social Trading',
          description: 'Follow and copy successful investors',
          features: ['portfolio_sharing', 'copy_trading', 'social_signals', 'performance_tracking']
        },
        COMMUNITY_FEATURES: {
          name: 'Community Features',
          description: 'Connect with fellow investors',
          features: ['discussions', 'forums', 'groups', 'mentorship']
        },
        GAMIFICATION: {
          name: 'Gamification',
          description: 'Learn and earn through games',
          features: ['challenges', 'achievements', 'rewards', 'leaderboards']
        },
        EDUCATIONAL_GAMES: {
          name: 'Educational Games',
          description: 'Learn investing through interactive games',
          features: ['simulation_games', 'quiz_games', 'strategy_games', 'risk_games']
        }
      };

      res.status(200).json({
        success: true,
        data: {
          socialFeatures,
          totalFeatures: Object.keys(socialFeatures).length
        },
        message: 'Social features retrieved successfully'
      });
    } catch (error) {
      logger.error('Error in getSocialFeatures controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Get achievement types
   * @route GET /api/social/achievement-types
   */
  async getAchievementTypes(req, res) {
    try {
      logger.info('Getting achievement types');

      const achievementTypes = {
        INVESTMENT_ACHIEVEMENTS: {
          first_investment: {
            name: 'First Investment',
            description: 'Made your first investment',
            points: 100,
            icon: '🎯',
            category: 'investment'
          },
          sip_streak: {
            name: 'SIP Streak',
            description: 'Maintained SIP for 6 months',
            points: 500,
            icon: '🔥',
            category: 'consistency'
          },
          portfolio_growth: {
            name: 'Portfolio Growth',
            description: 'Achieved 20% portfolio growth',
            points: 1000,
            icon: '📈',
            category: 'performance'
          },
          diversification: {
            name: 'Diversification Master',
            description: 'Invested in 5 different fund categories',
            points: 750,
            icon: '🎲',
            category: 'strategy'
          }
        },
        LEARNING_ACHIEVEMENTS: {
          course_completion: {
            name: 'Course Completion',
            description: 'Completed financial literacy course',
            points: 300,
            icon: '📚',
            category: 'education'
          },
          quiz_master: {
            name: 'Quiz Master',
            description: 'Scored 90%+ in investment quiz',
            points: 200,
            icon: '🧠',
            category: 'knowledge'
          },
          mentor: {
            name: 'Mentor',
            description: 'Helped 5 new investors',
            points: 400,
            icon: '👨‍🏫',
            category: 'community'
          }
        },
        SOCIAL_ACHIEVEMENTS: {
          community_leader: {
            name: 'Community Leader',
            description: 'Active community member for 3 months',
            points: 600,
            icon: '👑',
            category: 'social'
          },
          influencer: {
            name: 'Influencer',
            description: 'Gained 100 followers',
            points: 800,
            icon: '⭐',
            category: 'influence'
          },
          team_player: {
            name: 'Team Player',
            description: 'Participated in 10 community events',
            points: 350,
            icon: '🤝',
            category: 'participation'
          }
        }
      };

      res.status(200).json({
        success: true,
        data: {
          achievementTypes,
          totalCategories: Object.keys(achievementTypes).length
        },
        message: 'Achievement types retrieved successfully'
      });
    } catch (error) {
      logger.error('Error in getAchievementTypes controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Get challenge types
   * @route GET /api/social/challenge-types
   */
  async getChallengeTypes(req, res) {
    try {
      logger.info('Getting challenge types');

      const challengeTypes = {
        INVESTMENT_CHALLENGES: {
          sip_challenge: {
            name: 'SIP Challenge',
            description: 'Start and maintain SIP for 3 months',
            duration: '3 months',
            reward: 500,
            difficulty: 'easy',
            category: 'investment'
          },
          diversification_challenge: {
            name: 'Diversification Challenge',
            description: 'Invest in 3 different fund categories',
            duration: '1 month',
            reward: 300,
            difficulty: 'medium',
            category: 'strategy'
          },
          goal_achievement: {
            name: 'Goal Achievement',
            description: 'Achieve your first investment goal',
            duration: '6 months',
            reward: 1000,
            difficulty: 'hard',
            category: 'goals'
          }
        },
        LEARNING_CHALLENGES: {
          course_challenge: {
            name: 'Course Challenge',
            description: 'Complete 3 financial literacy courses',
            duration: '2 months',
            reward: 400,
            difficulty: 'medium',
            category: 'education'
          },
          quiz_challenge: {
            name: 'Quiz Challenge',
            description: 'Score 100% in 5 investment quizzes',
            duration: '1 month',
            reward: 250,
            difficulty: 'easy',
            category: 'knowledge'
          },
          research_challenge: {
            name: 'Research Challenge',
            description: 'Analyze and compare 10 mutual funds',
            duration: '2 weeks',
            reward: 350,
            difficulty: 'hard',
            category: 'analysis'
          }
        },
        SOCIAL_CHALLENGES: {
          community_challenge: {
            name: 'Community Challenge',
            description: 'Help 10 new investors get started',
            duration: '3 months',
            reward: 600,
            difficulty: 'medium',
            category: 'community'
          },
          sharing_challenge: {
            name: 'Sharing Challenge',
            description: 'Share 5 educational posts',
            duration: '1 month',
            reward: 200,
            difficulty: 'easy',
            category: 'sharing'
          },
          event_challenge: {
            name: 'Event Challenge',
            description: 'Participate in 5 community events',
            duration: '2 months',
            reward: 300,
            difficulty: 'medium',
            category: 'participation'
          }
        }
      };

      res.status(200).json({
        success: true,
        data: {
          challengeTypes,
          totalCategories: Object.keys(challengeTypes).length
        },
        message: 'Challenge types retrieved successfully'
      });
    } catch (error) {
      logger.error('Error in getChallengeTypes controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }
}

module.exports = new SocialInvestingController(); 