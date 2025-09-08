const logger = require('../utils/logger');
const { User, UserPortfolio, SocialProfile, Leaderboard, Challenge, Reward, Community } = require('../models');

class SocialInvestingService {
  constructor() {
    this.socialFeatures = {
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

    this.achievementTypes = {
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

    this.challengeTypes = {
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

    this.rewardTypes = {
      POINTS: {
        name: 'Points',
        description: 'Earn points for achievements',
        value: 1,
        redeemable: true
      },
      BADGES: {
        name: 'Badges',
        description: 'Collect badges for milestones',
        value: 0,
        redeemable: false
      },
      CASHBACK: {
        name: 'Cashback',
        description: 'Get cashback on investments',
        value: 0.5,
        redeemable: true
      },
      FEATURES: {
        name: 'Premium Features',
        description: 'Unlock premium features',
        value: 0,
        redeemable: false
      }
    };
  }

  /**
   * Create social profile for user
   */
  async createSocialProfile(userId) {
    try {
      logger.info('Creating social profile', { userId });

      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      const socialProfile = await SocialProfile.findOneAndUpdate(
        { userId },
        {
          username: user.username || `investor_${userId.slice(-6)}`,
          displayName: user.name || 'Anonymous Investor',
          bio: 'Passionate about investing and financial growth',
          avatar: user.avatar || 'default_avatar.png',
          followers: [],
          following: [],
          posts: [],
          achievements: [],
          points: 0,
          level: 1,
          reputation: 0,
          isPublic: true,
          privacySettings: {
            showPortfolio: true,
            showPerformance: true,
            allowFollow: true,
            allowMessages: true
          },
          createdAt: new Date(),
          lastActive: new Date()
        },
        { upsert: true, new: true }
      );

      return {
        success: true,
        data: {
          socialProfile,
          message: 'Social profile created successfully'
        }
      };
    } catch (error) {
      logger.error('Failed to create social profile', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to create social profile',
        error: error.message
      };
    }
  }

  /**
   * Follow another investor
   */
  async followInvestor(followerId, followingId) {
    try {
      logger.info('Following investor', { followerId, followingId });

      if (followerId === followingId) {
        throw new Error('Cannot follow yourself');
      }

      const followerProfile = await SocialProfile.findOne({ userId: followerId });
      const followingProfile = await SocialProfile.findOne({ userId: followingId });

      if (!followerProfile || !followingProfile) {
        throw new Error('User profile not found');
      }

      if (followingProfile.followers.includes(followerId)) {
        throw new Error('Already following this user');
      }

      // Add to following list
      await SocialProfile.findByIdAndUpdate(followerProfile._id, {
        $push: { following: followingId },
        $inc: { followingCount: 1 }
      });

      // Add to followers list
      await SocialProfile.findByIdAndUpdate(followingProfile._id, {
        $push: { followers: followerId },
        $inc: { followersCount: 1 }
      });

      // Award points for following
      await this.awardPoints(followerId, 10, 'following_user');

      return {
        success: true,
        data: {
          message: 'Successfully followed investor',
          followingCount: followerProfile.followingCount + 1,
          followersCount: followingProfile.followersCount + 1
        }
      };
    } catch (error) {
      logger.error('Failed to follow investor', { error: error.message, followerId, followingId });
      return {
        success: false,
        message: 'Failed to follow investor',
        error: error.message
      };
    }
  }

  /**
   * Share portfolio performance
   */
  async sharePortfolioPerformance(userId, shareType = 'public') {
    try {
      logger.info('Sharing portfolio performance', { userId, shareType });

      const userPortfolio = await UserPortfolio.findOne({ userId });
      if (!userPortfolio) {
        throw new Error('Portfolio not found');
      }

      const socialProfile = await SocialProfile.findOne({ userId });
      if (!socialProfile) {
        throw new Error('Social profile not found');
      }

      const performanceData = {
        totalValue: userPortfolio.totalValue,
        totalReturn: userPortfolio.totalReturn,
        returnPercentage: userPortfolio.returnPercentage,
        topHoldings: userPortfolio.holdings.slice(0, 3),
        riskScore: userPortfolio.riskScore,
        diversificationScore: userPortfolio.diversificationScore
      };

      const post = {
        type: 'portfolio_share',
        content: `My portfolio performance: ${performanceData.returnPercentage}% return`,
        data: performanceData,
        shareType: shareType,
        likes: 0,
        comments: [],
        shares: 0,
        createdAt: new Date()
      };

      // Add post to social profile
      await SocialProfile.findByIdAndUpdate(socialProfile._id, {
        $push: { posts: post },
        $inc: { postCount: 1 }
      });

      // Award points for sharing
      await this.awardPoints(userId, 25, 'sharing_portfolio');

      return {
        success: true,
        data: {
          post,
          message: 'Portfolio performance shared successfully'
        }
      };
    } catch (error) {
      logger.error('Failed to share portfolio performance', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to share portfolio performance',
        error: error.message
      };
    }
  }

  /**
   * Create investment challenge
   */
  async createInvestmentChallenge(userId, challengeType) {
    try {
      logger.info('Creating investment challenge', { userId, challengeType });

      const challengeTemplate = this.challengeTypes.INVESTMENT_CHALLENGES[challengeType];
      if (!challengeTemplate) {
        throw new Error(`Invalid challenge type: ${challengeType}`);
      }

      const challenge = await Challenge.create({
        userId,
        type: challengeType,
        name: challengeTemplate.name,
        description: challengeTemplate.description,
        category: challengeTemplate.category,
        difficulty: challengeTemplate.difficulty,
        duration: challengeTemplate.duration,
        reward: challengeTemplate.reward,
        startDate: new Date(),
        endDate: this.calculateEndDate(challengeTemplate.duration),
        progress: 0,
        status: 'active',
        milestones: this.getChallengeMilestones(challengeType),
        createdAt: new Date()
      });

      return {
        success: true,
        data: {
          challenge,
          message: 'Challenge created successfully'
        }
      };
    } catch (error) {
      logger.error('Failed to create investment challenge', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to create investment challenge',
        error: error.message
      };
    }
  }

  /**
   * Award points to user
   */
  async awardPoints(userId, points, reason) {
    try {
      logger.info('Awarding points', { userId, points, reason });

      const socialProfile = await SocialProfile.findOne({ userId });
      if (!socialProfile) {
        throw new Error('Social profile not found');
      }

      const newPoints = socialProfile.points + points;
      const newLevel = this.calculateLevel(newPoints);

      await SocialProfile.findByIdAndUpdate(socialProfile._id, {
        points: newPoints,
        level: newLevel,
        $push: {
          pointHistory: {
            points,
            reason,
            timestamp: new Date()
          }
        }
      });

      // Check for level up
      if (newLevel > socialProfile.level) {
        await this.handleLevelUp(userId, newLevel);
      }

      return {
        success: true,
        data: {
          points: newPoints,
          level: newLevel,
          pointsEarned: points,
          reason
        }
      };
    } catch (error) {
      logger.error('Failed to award points', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to award points',
        error: error.message
      };
    }
  }

  /**
   * Get leaderboard
   */
  async getLeaderboard(category = 'overall', timeFrame = 'monthly') {
    try {
      logger.info('Getting leaderboard', { category, timeFrame });

      let leaderboardQuery = {};

      switch (category) {
        case 'points':
          leaderboardQuery = { points: { $exists: true } };
          break;
        case 'performance':
          leaderboardQuery = { 'portfolio.returnPercentage': { $exists: true } };
          break;
        case 'consistency':
          leaderboardQuery = { 'portfolio.sipStreak': { $exists: true } };
          break;
        case 'community':
          leaderboardQuery = { followersCount: { $exists: true } };
          break;
        default:
          leaderboardQuery = {};
      }

      const socialProfiles = await SocialProfile.find(leaderboardQuery)
        .sort(this.getSortCriteria(category))
        .limit(50)
        .populate('userId', 'name username');

      const leaderboard = socialProfiles.map((profile, index) => ({
        rank: index + 1,
        userId: profile.userId._id,
        name: profile.userId.name,
        username: profile.userId.username,
        avatar: profile.avatar,
        score: this.getScore(profile, category),
        level: profile.level,
        achievements: profile.achievements.length
      }));

      return {
        success: true,
        data: {
          leaderboard,
          category,
          timeFrame,
          totalParticipants: leaderboard.length
        }
      };
    } catch (error) {
      logger.error('Failed to get leaderboard', { error: error.message });
      return {
        success: false,
        message: 'Failed to get leaderboard',
        error: error.message
      };
    }
  }

  /**
   * Create educational game
   */
  async createEducationalGame(userId, gameType) {
    try {
      logger.info('Creating educational game', { userId, gameType });

      const game = {
        type: gameType,
        userId,
        startTime: new Date(),
        score: 0,
        level: 1,
        questions: this.getGameQuestions(gameType),
        currentQuestion: 0,
        status: 'active',
        rewards: {
          points: 0,
          badges: [],
          experience: 0
        }
      };

      // Store game in user's social profile
      await SocialProfile.findOneAndUpdate(
        { userId },
        {
          $push: { activeGames: game },
          $inc: { gamesPlayed: 1 }
        }
      );

      return {
        success: true,
        data: {
          game,
          message: 'Educational game started successfully'
        }
      };
    } catch (error) {
      logger.error('Failed to create educational game', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to create educational game',
        error: error.message
      };
    }
  }

  /**
   * Submit game answer
   */
  async submitGameAnswer(userId, gameId, answer) {
    try {
      logger.info('Submitting game answer', { userId, gameId, answer });

      const socialProfile = await SocialProfile.findOne({ userId });
      if (!socialProfile) {
        throw new Error('Social profile not found');
      }

      const game = socialProfile.activeGames.find(g => g._id.toString() === gameId);
      if (!game) {
        throw new Error('Game not found');
      }

      const currentQuestion = game.questions[game.currentQuestion];
      const isCorrect = this.checkAnswer(currentQuestion, answer);

      if (isCorrect) {
        game.score += 10;
        game.rewards.points += 5;
        game.rewards.experience += 10;
      }

      game.currentQuestion++;

      // Check if game is complete
      if (game.currentQuestion >= game.questions.length) {
        game.status = 'completed';
        game.endTime = new Date();
        
        // Award completion rewards
        const completionRewards = this.getCompletionRewards(game);
        game.rewards.points += completionRewards.points;
        game.rewards.badges.push(...completionRewards.badges);
        game.rewards.experience += completionRewards.experience;

        // Award points to user
        await this.awardPoints(userId, game.rewards.points, 'game_completion');
      }

      // Update game in social profile
      await SocialProfile.findByIdAndUpdate(socialProfile._id, {
        $set: { 'activeGames.$[gameId]': game }
      }, {
        arrayFilters: [{ 'gameId._id': gameId }]
      });

      return {
        success: true,
        data: {
          isCorrect,
          score: game.score,
          currentQuestion: game.currentQuestion,
          gameComplete: game.status === 'completed',
          rewards: game.rewards
        }
      };
    } catch (error) {
      logger.error('Failed to submit game answer', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to submit game answer',
        error: error.message
      };
    }
  }

  /**
   * Get user achievements
   */
  async getUserAchievements(userId) {
    try {
      logger.info('Getting user achievements', { userId });

      const socialProfile = await SocialProfile.findOne({ userId });
      if (!socialProfile) {
        throw new Error('Social profile not found');
      }

      const achievements = socialProfile.achievements.map(achievement => ({
        ...achievement,
        details: this.achievementTypes[achievement.category]?.[achievement.type] || {}
      }));

      const totalPoints = socialProfile.points;
      const level = socialProfile.level;
      const nextLevelPoints = this.getNextLevelPoints(level);

      return {
        success: true,
        data: {
          achievements,
          totalPoints,
          level,
          nextLevelPoints,
          progressToNextLevel: ((totalPoints % 1000) / 1000) * 100,
          totalAchievements: achievements.length
        }
      };
    } catch (error) {
      logger.error('Failed to get user achievements', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to get user achievements',
        error: error.message
      };
    }
  }

  /**
   * Create community event
   */
  async createCommunityEvent(userId, eventData) {
    try {
      logger.info('Creating community event', { userId, eventData });

      const event = await Community.create({
        organizerId: userId,
        title: eventData.title,
        description: eventData.description,
        type: eventData.type,
        category: eventData.category,
        startDate: eventData.startDate,
        endDate: eventData.endDate,
        location: eventData.location,
        maxParticipants: eventData.maxParticipants || 100,
        participants: [userId],
        status: 'upcoming',
        rewards: {
          points: eventData.rewardPoints || 100,
          badges: eventData.rewardBadges || []
        },
        createdAt: new Date()
      });

      return {
        success: true,
        data: {
          event,
          message: 'Community event created successfully'
        }
      };
    } catch (error) {
      logger.error('Failed to create community event', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to create community event',
        error: error.message
      };
    }
  }

  // Helper methods
  calculateEndDate(duration) {
    const now = new Date();
    const [value, unit] = duration.split(' ');
    
    switch (unit) {
      case 'days':
        return new Date(now.getTime() + parseInt(value) * 24 * 60 * 60 * 1000);
      case 'weeks':
        return new Date(now.getTime() + parseInt(value) * 7 * 24 * 60 * 60 * 1000);
      case 'months':
        return new Date(now.getTime() + parseInt(value) * 30 * 24 * 60 * 60 * 1000);
      default:
        return new Date(now.getTime() + 30 * 24 * 60 * 60 * 1000); // Default 30 days
    }
  }

  getChallengeMilestones(challengeType) {
    const milestones = {
      sip_challenge: [
        { name: 'Start SIP', progress: 0, reward: 50 },
        { name: '1 Month', progress: 33, reward: 100 },
        { name: '2 Months', progress: 66, reward: 150 },
        { name: '3 Months', progress: 100, reward: 200 }
      ],
      diversification_challenge: [
        { name: '1 Category', progress: 33, reward: 100 },
        { name: '2 Categories', progress: 66, reward: 150 },
        { name: '3 Categories', progress: 100, reward: 200 }
      ]
    };

    return milestones[challengeType] || [];
  }

  calculateLevel(points) {
    return Math.floor(points / 1000) + 1;
  }

  async handleLevelUp(userId, newLevel) {
    try {
      // Award level up bonus
      await this.awardPoints(userId, 100, 'level_up');
      
      // Unlock new features based on level
      const unlockedFeatures = this.getUnlockedFeatures(newLevel);
      
      logger.info('User leveled up', { userId, newLevel, unlockedFeatures });
    } catch (error) {
      logger.error('Failed to handle level up', { error: error.message, userId });
    }
  }

  getUnlockedFeatures(level) {
    const features = {
      1: ['basic_social', 'basic_games'],
      2: ['portfolio_sharing', 'basic_challenges'],
      3: ['advanced_games', 'community_events'],
      4: ['mentorship', 'premium_features'],
      5: ['expert_status', 'all_features']
    };

    return features[level] || [];
  }

  getSortCriteria(category) {
    const sortCriteria = {
      points: { points: -1 },
      performance: { 'portfolio.returnPercentage': -1 },
      consistency: { 'portfolio.sipStreak': -1 },
      community: { followersCount: -1 },
      overall: { points: -1, 'portfolio.returnPercentage': -1 }
    };

    return sortCriteria[category] || { points: -1 };
  }

  getScore(profile, category) {
    const scores = {
      points: profile.points,
      performance: profile.portfolio?.returnPercentage || 0,
      consistency: profile.portfolio?.sipStreak || 0,
      community: profile.followersCount || 0,
      overall: (profile.points + (profile.portfolio?.returnPercentage || 0) * 10 + (profile.followersCount || 0) * 5)
    };

    return scores[category] || 0;
  }

  getGameQuestions(gameType) {
    const questions = {
      investment_quiz: [
        {
          question: 'What is a mutual fund?',
          options: ['A type of bank account', 'A pool of investments', 'A type of insurance', 'A loan product'],
          correct: 1,
          explanation: 'A mutual fund is a pool of investments managed by professionals.'
        },
        {
          question: 'What does SIP stand for?',
          options: ['Systematic Investment Plan', 'Simple Investment Process', 'Smart Investment Program', 'Secure Investment Plan'],
          correct: 0,
          explanation: 'SIP stands for Systematic Investment Plan.'
        }
      ],
      risk_assessment: [
        {
          question: 'How would you react if your investment lost 20% in a month?',
          options: ['Panic and withdraw', 'Wait and watch', 'Invest more', 'Switch to safer funds'],
          correct: 1,
          explanation: 'Market volatility is normal. Stay invested for long-term gains.'
        }
      ]
    };

    return questions[gameType] || questions.investment_quiz;
  }

  checkAnswer(question, answer) {
    return question.correct === answer;
  }

  getCompletionRewards(game) {
    const baseRewards = {
      points: 50,
      badges: ['game_completion'],
      experience: 100
    };

    // Bonus for high scores
    if (game.score >= 80) {
      baseRewards.points += 25;
      baseRewards.badges.push('high_scorer');
    }

    return baseRewards;
  }

  getNextLevelPoints(level) {
    return level * 1000;
  }
}

module.exports = new SocialInvestingService(); 