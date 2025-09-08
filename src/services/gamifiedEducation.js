const logger = require('../utils/logger');
const { User, UserLearning, Achievement, Leaderboard, TriviaBattle, Streak, Badge, Coupon } = require('../models');
const learningModule = require('./learningModule');

class GamifiedEducation {
  constructor() {
    this.badgeTypes = {
      DAILY_LEARNER: {
        name: 'Daily Learner',
        description: 'Complete lessons for 7 consecutive days',
        icon: '📚',
        points: 50,
        requirement: { type: 'streak', value: 7 }
      },
      QUIZ_MASTER: {
        name: 'Quiz Master',
        description: 'Score 100% in 5 quizzes',
        icon: '🏆',
        points: 100,
        requirement: { type: 'perfect_quizzes', value: 5 }
      },
      TOPIC_EXPERT: {
        name: 'Topic Expert',
        description: 'Complete all lessons in a topic with 90%+ quiz scores',
        icon: '🎯',
        points: 150,
        requirement: { type: 'topic_mastery', value: 1 }
      },
      STREAK_CHAMPION: {
        name: 'Streak Champion',
        description: 'Maintain 30-day learning streak',
        icon: '🔥',
        points: 200,
        requirement: { type: 'streak', value: 30 }
      },
      COMMUNITY_HELPER: {
        name: 'Community Helper',
        description: 'Help 10 other users in trivia battles',
        icon: '🤝',
        points: 75,
        requirement: { type: 'help_others', value: 10 }
      },
      EARLY_ADOPTER: {
        name: 'Early Adopter',
        description: 'Join the platform in first 1000 users',
        icon: '⭐',
        points: 100,
        requirement: { type: 'early_user', value: 1000 }
      },
      GOAL_ACHIEVER: {
        name: 'Goal Achiever',
        description: 'Achieve your first investment goal',
        icon: '🎉',
        points: 300,
        requirement: { type: 'goal_achievement', value: 1 }
      },
      TAX_SAVER: {
        name: 'Tax Saver',
        description: 'Complete tax optimization lessons and implement strategies',
        icon: '💰',
        points: 125,
        requirement: { type: 'tax_optimization', value: 1 }
      }
    };

    this.titles = {
      NOVICE_INVESTOR: {
        name: 'Novice Investor',
        description: 'Just starting the investment journey',
        requirement: { type: 'lessons_completed', value: 5 },
        color: '#6B7280'
      },
      FUND_LEARNER: {
        name: 'Fund Learner',
        description: 'Understanding mutual fund basics',
        requirement: { type: 'lessons_completed', value: 15 },
        color: '#3B82F6'
      },
      ASSET_ALLOCATOR: {
        name: 'Asset Allocator',
        description: 'Mastered portfolio allocation strategies',
        requirement: { type: 'topics_mastered', value: 3 },
        color: '#10B981'
      },
      ELSS_CHAMP: {
        name: 'ELSS Champ',
        description: 'Expert in tax-saving investments',
        requirement: { type: 'tax_lessons_completed', value: 10 },
        color: '#F59E0B'
      },
      SIP_MASTER: {
        name: 'SIP Master',
        description: 'Systematic investment planning expert',
        requirement: { type: 'sip_lessons_completed', value: 8 },
        color: '#8B5CF6'
      },
      PORTFOLIO_GURU: {
        name: 'Portfolio Guru',
        description: 'Advanced portfolio management skills',
        requirement: { type: 'advanced_lessons', value: 20 },
        color: '#EF4444'
      },
      INVESTMENT_SAGE: {
        name: 'Investment Sage',
        description: 'Complete mastery of investment concepts',
        requirement: { type: 'all_lessons_completed', value: 1 },
        color: '#000000'
      }
    };

    this.couponTypes = {
      SIP_BOOST: {
        name: 'SIP Boost',
        description: '10% extra units on next SIP',
        value: 10,
        type: 'percentage',
        validity: 30, // days
        requirement: { type: 'streak', value: 7 }
      },
      ZERO_EXIT_LOAD: {
        name: 'Zero Exit Load',
        description: 'No exit load on fund switches',
        value: 100,
        type: 'percentage',
        validity: 7,
        requirement: { type: 'quiz_mastery', value: 5 }
      },
      LEARNING_BONUS: {
        name: 'Learning Bonus',
        description: '₹100 bonus for completing topic',
        value: 100,
        type: 'fixed',
        validity: 15,
        requirement: { type: 'topic_completion', value: 1 }
      },
      STREAK_REWARD: {
        name: 'Streak Reward',
        description: '₹50 for 30-day streak',
        value: 50,
        type: 'fixed',
        validity: 7,
        requirement: { type: 'streak', value: 30 }
      }
    };

    this.triviaCategories = {
      MUTUAL_FUNDS: {
        name: 'Mutual Funds',
        questions: [
          {
            question: 'What does NAV stand for?',
            options: ['Net Asset Value', 'New Asset Value', 'Net Annual Value', 'New Annual Value'],
            correctAnswer: 0,
            explanation: 'NAV stands for Net Asset Value, which represents the per-unit market value of a mutual fund.'
          },
          {
            question: 'Which type of mutual fund invests primarily in government securities?',
            options: ['Equity Fund', 'Debt Fund', 'Balanced Fund', 'Money Market Fund'],
            correctAnswer: 1,
            explanation: 'Debt funds primarily invest in government securities, corporate bonds, and other debt instruments.'
          }
        ]
      },
      SIP: {
        name: 'Systematic Investment Planning',
        questions: [
          {
            question: 'What is the main advantage of SIP?',
            options: ['Higher returns', 'Rupee cost averaging', 'Tax benefits', 'Lower risk'],
            correctAnswer: 1,
            explanation: 'SIP helps in rupee cost averaging by buying more units when prices are low and fewer when prices are high.'
          }
        ]
      },
      TAX: {
        name: 'Taxation',
        questions: [
          {
            question: 'What is the tax rate for LTCG on equity mutual funds?',
            options: ['5%', '10%', '15%', '20%'],
            correctAnswer: 1,
            explanation: 'LTCG on equity mutual funds is taxed at 10% without indexation benefit.'
          }
        ]
      }
    };
  }

  /**
   * Initialize gamification for user
   */
  async initializeGamification(userId) {
    try {
      logger.info('Initializing gamification', { userId });

      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      // Create initial streak record
      const streak = new Streak({
        userId,
        currentStreak: 0,
        longestStreak: 0,
        lastActivityDate: null,
        streakHistory: []
      });

      await streak.save();

      // Assign initial title
      const initialTitle = await this.assignInitialTitle(userId);

      // Create initial badges
      const initialBadges = await this.createInitialBadges(userId);

      return {
        success: true,
        data: {
          streak,
          title: initialTitle,
          badges: initialBadges,
          availableRewards: await this.getAvailableRewards(userId)
        }
      };
    } catch (error) {
      logger.error('Failed to initialize gamification', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to initialize gamification',
        error: error.message
      };
    }
  }

  /**
   * Update daily streak
   */
  async updateDailyStreak(userId) {
    try {
      logger.info('Updating daily streak', { userId });

      const streak = await Streak.findOne({ userId });
      if (!streak) {
        throw new Error('Streak record not found');
      }

      const today = new Date().toDateString();
      const lastActivity = streak.lastActivityDate ? new Date(streak.lastActivityDate).toDateString() : null;

      if (lastActivity === today) {
        // Already updated today
        return {
          success: true,
          data: {
            currentStreak: streak.currentStreak,
            message: 'Streak already updated today'
          }
        };
      }

      if (lastActivity === new Date(Date.now() - 24 * 60 * 60 * 1000).toDateString()) {
        // Consecutive day
        streak.currentStreak++;
      } else {
        // Streak broken
        streak.streakHistory.push({
          streak: streak.currentStreak,
          endDate: streak.lastActivityDate
        });
        streak.currentStreak = 1;
      }

      if (streak.currentStreak > streak.longestStreak) {
        streak.longestStreak = streak.currentStreak;
      }

      streak.lastActivityDate = new Date();
      await streak.save();

      // Check for streak-based achievements
      const achievements = await this.checkStreakAchievements(userId, streak.currentStreak);

      // Check for streak-based coupons
      const coupons = await this.checkStreakCoupons(userId, streak.currentStreak);

      return {
        success: true,
        data: {
          currentStreak: streak.currentStreak,
          longestStreak: streak.longestStreak,
          achievements,
          coupons,
          nextMilestone: this.getNextStreakMilestone(streak.currentStreak)
        }
      };
    } catch (error) {
      logger.error('Failed to update daily streak', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to update daily streak',
        error: error.message
      };
    }
  }

  /**
   * Award badge to user
   */
  async awardBadge(userId, badgeType, context = {}) {
    try {
      logger.info('Awarding badge', { userId, badgeType });

      const badgeInfo = this.badgeTypes[badgeType];
      if (!badgeInfo) {
        throw new Error('Invalid badge type');
      }

      // Check if user already has this badge
      const existingBadge = await Badge.findOne({ userId, type: badgeType });
      if (existingBadge) {
        return {
          success: true,
          data: {
            message: 'Badge already awarded',
            badge: existingBadge
          }
        };
      }

      // Create new badge
      const badge = new Badge({
        userId,
        type: badgeType,
        name: badgeInfo.name,
        description: badgeInfo.description,
        icon: badgeInfo.icon,
        points: badgeInfo.points,
        context,
        awardedAt: new Date()
      });

      await badge.save();

      // Update user points
      await this.updateUserPoints(userId, badgeInfo.points);

      return {
        success: true,
        data: {
          badge,
          pointsEarned: badgeInfo.points,
          message: `Congratulations! You earned the ${badgeInfo.name} badge!`
        }
      };
    } catch (error) {
      logger.error('Failed to award badge', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to award badge',
        error: error.message
      };
    }
  }

  /**
   * Update user title
   */
  async updateUserTitle(userId) {
    try {
      logger.info('Updating user title', { userId });

      const userLearning = await UserLearning.findOne({ userId });
      const achievements = await Achievement.find({ userId });

      if (!userLearning) {
        throw new Error('User learning data not found');
      }

      // Calculate user progress
      const progress = await this.calculateUserProgress(userId);

      // Find appropriate title
      const newTitle = this.determineUserTitle(progress, achievements);

      // Update user title
      const user = await User.findByIdAndUpdate(
        userId,
        { title: newTitle },
        { new: true }
      );

      return {
        success: true,
        data: {
          title: newTitle,
          progress,
          message: `Your title has been updated to: ${newTitle}`
        }
      };
    } catch (error) {
      logger.error('Failed to update user title', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to update user title',
        error: error.message
      };
    }
  }

  /**
   * Generate coupon for user
   */
  async generateCoupon(userId, couponType) {
    try {
      logger.info('Generating coupon', { userId, couponType });

      const couponInfo = this.couponTypes[couponType];
      if (!couponInfo) {
        throw new Error('Invalid coupon type');
      }

      // Check if user is eligible
      const isEligible = await this.checkCouponEligibility(userId, couponType);
      if (!isEligible) {
        throw new Error('User not eligible for this coupon');
      }

      // Create coupon
      const coupon = new Coupon({
        userId,
        type: couponType,
        name: couponInfo.name,
        description: couponInfo.description,
        value: couponInfo.value,
        valueType: couponInfo.type,
        validityDays: couponInfo.validity,
        validFrom: new Date(),
        validUntil: new Date(Date.now() + couponInfo.validity * 24 * 60 * 60 * 1000),
        isUsed: false,
        usedAt: null
      });

      await coupon.save();

      return {
        success: true,
        data: {
          coupon,
          message: `You earned a ${couponInfo.name} coupon!`
        }
      };
    } catch (error) {
      logger.error('Failed to generate coupon', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to generate coupon',
        error: error.message
      };
    }
  }

  /**
   * Get leaderboard
   */
  async getLeaderboard(category = 'overall', limit = 50) {
    try {
      logger.info('Getting leaderboard', { category, limit });

      let leaderboard;

      switch (category) {
        case 'streak':
          leaderboard = await this.getStreakLeaderboard(limit);
          break;
        case 'points':
          leaderboard = await this.getPointsLeaderboard(limit);
          break;
        case 'achievements':
          leaderboard = await this.getAchievementsLeaderboard(limit);
          break;
        case 'quizzes':
          leaderboard = await this.getQuizLeaderboard(limit);
          break;
        default:
          leaderboard = await this.getOverallLeaderboard(limit);
      }

      return {
        success: true,
        data: {
          leaderboard,
          category,
          userRank: await this.getUserRank(category),
          totalParticipants: await this.getTotalParticipants()
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
   * Start trivia battle
   */
  async startTriviaBattle(userId, category, opponentId = null) {
    try {
      logger.info('Starting trivia battle', { userId, category, opponentId });

      const categoryInfo = this.triviaCategories[category];
      if (!categoryInfo) {
        throw new Error('Invalid trivia category');
      }

      // Generate battle questions
      const questions = this.generateBattleQuestions(category, 5);

      // Create battle record
      const battle = new TriviaBattle({
        category,
        questions,
        participants: [userId],
        scores: {},
        status: 'waiting',
        startTime: new Date(),
        endTime: null
      });

      if (opponentId) {
        battle.participants.push(opponentId);
        battle.status = 'active';
      }

      await battle.save();

      return {
        success: true,
        data: {
          battleId: battle._id,
          category,
          questions: questions.map(q => ({
            id: q.id,
            question: q.question,
            options: q.options
          })),
          timeLimit: 300, // 5 minutes
          participants: battle.participants
        }
      };
    } catch (error) {
      logger.error('Failed to start trivia battle', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to start trivia battle',
        error: error.message
      };
    }
  }

  /**
   * Submit trivia battle answers
   */
  async submitTriviaBattle(battleId, userId, answers) {
    try {
      logger.info('Submitting trivia battle', { battleId, userId });

      const battle = await TriviaBattle.findById(battleId);
      if (!battle) {
        throw new Error('Battle not found');
      }

      if (!battle.participants.includes(userId)) {
        throw new Error('User not part of this battle');
      }

      // Evaluate answers
      const results = this.evaluateBattleAnswers(battle.questions, answers);
      
      // Update battle scores
      battle.scores[userId] = results.score;
      battle.answers = battle.answers || {};
      battle.answers[userId] = answers;

      // Check if all participants have answered
      if (Object.keys(battle.scores).length === battle.participants.length) {
        battle.status = 'completed';
        battle.endTime = new Date();
        
        // Determine winner
        const winner = this.determineBattleWinner(battle.scores);
        battle.winner = winner;
      }

      await battle.save();

      // Award points for participation
      await this.awardBattlePoints(userId, results.score);

      return {
        success: true,
        data: {
          score: results.score,
          correctAnswers: results.correctAnswers,
          totalQuestions: results.totalQuestions,
          battleStatus: battle.status,
          winner: battle.winner
        }
      };
    } catch (error) {
      logger.error('Failed to submit trivia battle', { error: error.message, battleId });
      return {
        success: false,
        message: 'Failed to submit trivia battle',
        error: error.message
      };
    }
  }

  /**
   * Get user gamification profile
   */
  async getUserGamificationProfile(userId) {
    try {
      logger.info('Getting user gamification profile', { userId });

      const streak = await Streak.findOne({ userId });
      const badges = await Badge.find({ userId });
      const achievements = await Achievement.find({ userId });
      const coupons = await Coupon.find({ userId, isUsed: false });
      const user = await User.findById(userId);

      const profile = {
        user: {
          name: user.name,
          title: user.title || 'Novice Investor',
          totalPoints: await this.getUserTotalPoints(userId),
          rank: await this.getUserRank('overall')
        },
        streak: {
          current: streak?.currentStreak || 0,
          longest: streak?.longestStreak || 0,
          nextMilestone: this.getNextStreakMilestone(streak?.currentStreak || 0)
        },
        badges: {
          total: badges.length,
          recent: badges.slice(-5),
          categories: this.categorizeBadges(badges)
        },
        achievements: {
          total: achievements.length,
          recent: achievements.slice(-5)
        },
        coupons: {
          active: coupons.length,
          total: coupons
        },
        leaderboards: {
          streakRank: await this.getUserRank('streak'),
          pointsRank: await this.getUserRank('points'),
          achievementsRank: await this.getUserRank('achievements')
        }
      };

      return {
        success: true,
        data: profile
      };
    } catch (error) {
      logger.error('Failed to get user gamification profile', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to get user gamification profile',
        error: error.message
      };
    }
  }

  // Helper methods
  async assignInitialTitle(userId) {
    const user = await User.findByIdAndUpdate(
      userId,
      { title: 'Novice Investor' },
      { new: true }
    );
    return user.title;
  }

  async createInitialBadges(userId) {
    const badges = [];
    
    // Award early adopter badge if user is among first 1000
    const userCount = await User.countDocuments();
    if (userCount <= 1000) {
      const badge = await this.awardBadge(userId, 'EARLY_ADOPTER');
      if (badge.success) {
        badges.push(badge.data.badge);
      }
    }

    return badges;
  }

  async getAvailableRewards(userId) {
    const rewards = [];
    
    // Check for available badges
    for (const [badgeType, badgeInfo] of Object.entries(this.badgeTypes)) {
      const isEligible = await this.checkBadgeEligibility(userId, badgeType);
      if (isEligible) {
        rewards.push({
          type: 'badge',
          name: badgeInfo.name,
          description: badgeInfo.description,
          icon: badgeInfo.icon
        });
      }
    }

    // Check for available coupons
    for (const [couponType, couponInfo] of Object.entries(this.couponTypes)) {
      const isEligible = await this.checkCouponEligibility(userId, couponType);
      if (isEligible) {
        rewards.push({
          type: 'coupon',
          name: couponInfo.name,
          description: couponInfo.description,
          value: couponInfo.value
        });
      }
    }

    return rewards;
  }

  async checkStreakAchievements(userId, currentStreak) {
    const achievements = [];

    if (currentStreak >= 7) {
      const badge = await this.awardBadge(userId, 'DAILY_LEARNER');
      if (badge.success) {
        achievements.push(badge.data.badge);
      }
    }

    if (currentStreak >= 30) {
      const badge = await this.awardBadge(userId, 'STREAK_CHAMPION');
      if (badge.success) {
        achievements.push(badge.data.badge);
      }
    }

    return achievements;
  }

  async checkStreakCoupons(userId, currentStreak) {
    const coupons = [];

    if (currentStreak >= 7) {
      const coupon = await this.generateCoupon(userId, 'SIP_BOOST');
      if (coupon.success) {
        coupons.push(coupon.data.coupon);
      }
    }

    if (currentStreak >= 30) {
      const coupon = await this.generateCoupon(userId, 'STREAK_REWARD');
      if (coupon.success) {
        coupons.push(coupon.data.coupon);
      }
    }

    return coupons;
  }

  getNextStreakMilestone(currentStreak) {
    const milestones = [7, 30, 60, 90, 180, 365];
    const nextMilestone = milestones.find(m => m > currentStreak);
    
    return {
      target: nextMilestone,
      current: currentStreak,
      remaining: nextMilestone - currentStreak,
      reward: nextMilestone === 7 ? 'SIP Boost Coupon' : 
              nextMilestone === 30 ? '₹50 Cash Reward' : 'Special Badge'
    };
  }

  async updateUserPoints(userId, points) {
    const user = await User.findById(userId);
    if (user) {
      user.points = (user.points || 0) + points;
      await user.save();
    }
  }

  async calculateUserProgress(userId) {
    const userLearning = await UserLearning.findOne({ userId });
    const achievements = await Achievement.find({ userId });
    const quizzes = await this.getUserQuizzes(userId);

    return {
      lessonsCompleted: userLearning?.learningHistory?.filter(h => h.action === 'lesson_completed').length || 0,
      topicsMastered: userLearning?.completedTopics || 0,
      perfectQuizzes: quizzes.filter(q => q.score === 100).length,
      totalPoints: achievements.reduce((sum, a) => sum + (a.points || 0), 0),
      achievementsCount: achievements.length
    };
  }

  determineUserTitle(progress, achievements) {
    if (progress.lessonsCompleted >= 50 && progress.topicsMastered >= 5) {
      return 'Investment Sage';
    } else if (progress.lessonsCompleted >= 30 && progress.topicsMastered >= 3) {
      return 'Portfolio Guru';
    } else if (progress.lessonsCompleted >= 20 && progress.topicsMastered >= 2) {
      return 'SIP Master';
    } else if (progress.lessonsCompleted >= 15) {
      return 'Asset Allocator';
    } else if (progress.lessonsCompleted >= 10) {
      return 'Fund Learner';
    } else if (progress.lessonsCompleted >= 5) {
      return 'Novice Investor';
    } else {
      return 'New Investor';
    }
  }

  async checkCouponEligibility(userId, couponType) {
    const streak = await Streak.findOne({ userId });
    const achievements = await Achievement.find({ userId });

    switch (couponType) {
      case 'SIP_BOOST':
        return streak?.currentStreak >= 7;
      case 'ZERO_EXIT_LOAD':
        return achievements.filter(a => a.type === 'QUIZ_MASTERY').length >= 5;
      case 'LEARNING_BONUS':
        return true; // Topic completion will be checked when generating
      case 'STREAK_REWARD':
        return streak?.currentStreak >= 30;
      default:
        return false;
    }
  }

  async getStreakLeaderboard(limit) {
    const streaks = await Streak.find().sort({ currentStreak: -1, longestStreak: -1 }).limit(limit);
    return await this.populateLeaderboardData(streaks, 'streak');
  }

  async getPointsLeaderboard(limit) {
    const users = await User.find().sort({ points: -1 }).limit(limit);
    return await this.populateLeaderboardData(users, 'points');
  }

  async getAchievementsLeaderboard(limit) {
    const achievements = await Achievement.aggregate([
      { $group: { _id: '$userId', count: { $sum: 1 } } },
      { $sort: { count: -1 } },
      { $limit: limit }
    ]);
    return await this.populateLeaderboardData(achievements, 'achievements');
  }

  async getQuizLeaderboard(limit) {
    // Implementation for quiz leaderboard
    return [];
  }

  async getOverallLeaderboard(limit) {
    const users = await User.find().sort({ points: -1 }).limit(limit);
    return await this.populateLeaderboardData(users, 'overall');
  }

  async populateLeaderboardData(data, category) {
    const leaderboard = [];
    
    for (let i = 0; i < data.length; i++) {
      const item = data[i];
      const user = await User.findById(item.userId || item._id);
      
      if (user) {
        leaderboard.push({
          rank: i + 1,
          userId: user._id,
          name: user.name,
          title: user.title,
          score: this.getScoreForCategory(item, category),
          avatar: user.avatar
        });
      }
    }

    return leaderboard;
  }

  getScoreForCategory(item, category) {
    switch (category) {
      case 'streak':
        return item.currentStreak;
      case 'points':
        return item.points;
      case 'achievements':
        return item.count;
      default:
        return item.points || 0;
    }
  }

  async getUserRank(category) {
    // Implementation to get user's rank in specific category
    return 0;
  }

  async getTotalParticipants() {
    return await User.countDocuments();
  }

  generateBattleQuestions(category, count) {
    const categoryInfo = this.triviaCategories[category];
    if (!categoryInfo) return [];

    const questions = categoryInfo.questions;
    const selectedQuestions = [];

    for (let i = 0; i < Math.min(count, questions.length); i++) {
      selectedQuestions.push({
        id: i + 1,
        ...questions[i]
      });
    }

    return selectedQuestions;
  }

  evaluateBattleAnswers(questions, answers) {
    let correctAnswers = 0;
    const results = [];

    questions.forEach((question, index) => {
      const userAnswer = answers[index];
      const isCorrect = userAnswer === question.correctAnswer;
      
      if (isCorrect) correctAnswers++;

      results.push({
        questionId: question.id,
        userAnswer,
        correctAnswer: question.correctAnswer,
        isCorrect,
        explanation: question.explanation
      });
    });

    const score = (correctAnswers / questions.length) * 100;

    return {
      score: Math.round(score),
      correctAnswers,
      totalQuestions: questions.length,
      results
    };
  }

  determineBattleWinner(scores) {
    const entries = Object.entries(scores);
    if (entries.length === 0) return null;

    return entries.reduce((winner, [userId, score]) => 
      score > winner.score ? { userId, score } : winner
    ).userId;
  }

  async awardBattlePoints(userId, score) {
    const points = Math.round(score / 10); // 1 point per 10% score
    await this.updateUserPoints(userId, points);
  }

  async getUserTotalPoints(userId) {
    const user = await User.findById(userId);
    return user?.points || 0;
  }

  categorizeBadges(badges) {
    const categories = {};
    badges.forEach(badge => {
      const category = badge.type.split('_')[0];
      categories[category] = (categories[category] || 0) + 1;
    });
    return categories;
  }

  async checkBadgeEligibility(userId, badgeType) {
    // Implementation to check if user is eligible for specific badge
    return false;
  }

  async getUserQuizzes(userId) {
    // Implementation to get user's quiz history
    return [];
  }
}

module.exports = new GamifiedEducation(); 