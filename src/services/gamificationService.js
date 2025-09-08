const logger = require('../utils/logger');
const User = require('../models/User');
const Achievement = require('../models/Achievement');
const Challenge = require('../models/Challenge');

class GamificationService {
  constructor() {
    this.achievements = new Map();
    this.challenges = new Map();
    this.behavioralNudges = new Map();
    this.engagementMetrics = new Map();
  }

  /**
   * Initialize gamification service
   */
  async initialize() {
    try {
      await this.loadAchievements();
      await this.loadChallenges();
      await this.loadBehavioralNudges();
      logger.info('Gamification Service initialized successfully');
      return true;
    } catch (error) {
      logger.error('Failed to initialize Gamification Service:', error);
      return false;
    }
  }

  /**
   * Load predefined achievements
   */
  async loadAchievements() {
    const achievementDefinitions = [
      {
        id: 'first_sip',
        name: 'First Steps',
        description: 'Start your first SIP',
        category: 'INVESTMENT',
        icon: '🎯',
        points: 100,
        criteria: { type: 'SIP_STARTED', count: 1 }
      },
      {
        id: 'sip_streak_7',
        name: 'Week Warrior',
        description: 'Complete 7 consecutive SIPs',
        category: 'CONSISTENCY',
        icon: '🔥',
        points: 250,
        criteria: { type: 'SIP_STREAK', count: 7 }
      },
      {
        id: 'sip_streak_30',
        name: 'Monthly Master',
        description: 'Complete 30 consecutive SIPs',
        category: 'CONSISTENCY',
        icon: '⭐',
        points: 500,
        criteria: { type: 'SIP_STREAK', count: 30 }
      },
      {
        id: 'portfolio_10k',
        name: 'Portfolio Pioneer',
        description: 'Reach ₹10,000 portfolio value',
        category: 'MILESTONE',
        icon: '💰',
        points: 300,
        criteria: { type: 'PORTFOLIO_VALUE', amount: 10000 }
      },
      {
        id: 'portfolio_1lakh',
        name: 'Lakhpati',
        description: 'Reach ₹1,00,000 portfolio value',
        category: 'MILESTONE',
        icon: '🏆',
        points: 1000,
        criteria: { type: 'PORTFOLIO_VALUE', amount: 100000 }
      },
      {
        id: 'diversification_5',
        name: 'Diversification Expert',
        description: 'Invest in 5 different fund categories',
        category: 'KNOWLEDGE',
        icon: '🌐',
        points: 400,
        criteria: { type: 'FUND_CATEGORIES', count: 5 }
      },
      {
        id: 'referral_5',
        name: 'Social Butterfly',
        description: 'Refer 5 friends successfully',
        category: 'SOCIAL',
        icon: '🦋',
        points: 600,
        criteria: { type: 'SUCCESSFUL_REFERRALS', count: 5 }
      },
      {
        id: 'leaderboard_top10',
        name: 'Top Performer',
        description: 'Reach top 10 in any leaderboard',
        category: 'PERFORMANCE',
        icon: '🥇',
        points: 800,
        criteria: { type: 'LEADERBOARD_RANK', rank: 10 }
      },
      {
        id: 'tax_savings_10k',
        name: 'Tax Saver',
        description: 'Save ₹10,000 in taxes through investments',
        category: 'KNOWLEDGE',
        icon: '📊',
        points: 700,
        criteria: { type: 'TAX_SAVINGS', amount: 10000 }
      },
      {
        id: 'learning_complete',
        name: 'Knowledge Seeker',
        description: 'Complete all investment education modules',
        category: 'EDUCATION',
        icon: '📚',
        points: 500,
        criteria: { type: 'EDUCATION_MODULES', count: 10 }
      }
    ];

    achievementDefinitions.forEach(achievement => {
      this.achievements.set(achievement.id, achievement);
    });

    logger.info(`Loaded ${achievementDefinitions.length} achievements`);
  }

  /**
   * Load predefined challenges
   */
  async loadChallenges() {
    const challengeDefinitions = [
      {
        id: 'sip_challenge_30',
        name: '30-Day SIP Challenge',
        description: 'Complete SIPs for 30 consecutive days',
        category: 'CONSISTENCY',
        duration: 30,
        reward: { points: 1000, badge: 'SIP_MASTER' },
        criteria: { type: 'DAILY_SIP', count: 30 }
      },
      {
        id: 'diversification_challenge',
        name: 'Diversification Challenge',
        description: 'Invest in 7 different fund categories',
        category: 'KNOWLEDGE',
        duration: 90,
        reward: { points: 800, badge: 'DIVERSIFICATION_EXPERT' },
        criteria: { type: 'FUND_CATEGORIES', count: 7 }
      },
      {
        id: 'referral_challenge',
        name: 'Referral Challenge',
        description: 'Refer 10 friends in 60 days',
        category: 'SOCIAL',
        duration: 60,
        reward: { points: 1200, badge: 'SOCIAL_INFLUENCER' },
        criteria: { type: 'SUCCESSFUL_REFERRALS', count: 10 }
      }
    ];

    challengeDefinitions.forEach(challenge => {
      this.challenges.set(challenge.id, challenge);
    });

    logger.info(`Loaded ${challengeDefinitions.length} challenges`);
  }

  /**
   * Load behavioral nudges
   */
  async loadBehavioralNudges() {
    const nudgeDefinitions = [
      {
        id: 'sip_reminder',
        name: 'SIP Reminder',
        description: 'Gentle reminder to continue your SIP',
        type: 'REMINDER',
        trigger: { type: 'SIP_MISSED', days: 7 },
        message: 'Don\'t break your investment streak! Continue your SIP to build wealth.',
        action: 'CONTINUE_SIP'
      },
      {
        id: 'market_dip_opportunity',
        name: 'Market Opportunity',
        description: 'Alert about market dip for increased investment',
        type: 'OPPORTUNITY',
        trigger: { type: 'MARKET_DIP', percentage: 5 },
        message: 'Market is down! Consider increasing your SIP to buy more units.',
        action: 'INCREASE_SIP'
      },
      {
        id: 'diversification_nudge',
        name: 'Diversification Nudge',
        description: 'Suggest portfolio diversification',
        type: 'EDUCATION',
        trigger: { type: 'LOW_DIVERSIFICATION', categories: 2 },
        message: 'Your portfolio could benefit from more diversification. Consider adding different fund categories.',
        action: 'LEARN_DIVERSIFICATION'
      },
      {
        id: 'goal_progress',
        name: 'Goal Progress',
        description: 'Celebrate progress towards financial goals',
        type: 'CELEBRATION',
        trigger: { type: 'GOAL_MILESTONE', percentage: 25 },
        message: 'Great progress! You\'re 25% closer to your financial goal.',
        action: 'VIEW_GOAL_PROGRESS'
      }
    ];

    nudgeDefinitions.forEach(nudge => {
      this.behavioralNudges.set(nudge.id, nudge);
    });

    logger.info(`Loaded ${nudgeDefinitions.length} behavioral nudges`);
  }

  /**
   * Check and award achievements
   */
  async checkAchievements(userId, action, data) {
    try {
      const user = await User.findOne({ supabaseId: userId });
      if (!user) return [];

      const awardedAchievements = [];
      const userAchievements = user.achievements || [];

      for (const [achievementId, achievement] of this.achievements) {
        // Skip if already awarded
        if (userAchievements.includes(achievementId)) continue;

        // Check if achievement criteria are met
        if (this.checkAchievementCriteria(achievement, action, data, user)) {
          await this.awardAchievement(userId, achievement);
          awardedAchievements.push(achievement);
        }
      }

      return awardedAchievements;
    } catch (error) {
      logger.error('Error checking achievements:', error);
      return [];
    }
  }

  /**
   * Check if achievement criteria are met
   */
  checkAchievementCriteria(achievement, action, data, user) {
    const criteria = achievement.criteria;

    switch (criteria.type) {
      case 'SIP_STARTED':
        return this.checkSIPStartedCriteria(criteria, action, data);
      
      case 'SIP_STREAK':
        return this.checkSIPStreakCriteria(criteria, user);
      
      case 'PORTFOLIO_VALUE':
        return this.checkPortfolioValueCriteria(criteria, user);
      
      case 'FUND_CATEGORIES':
        return this.checkFundCategoriesCriteria(criteria, user);
      
      case 'SUCCESSFUL_REFERRALS':
        return this.checkReferralsCriteria(criteria, user);
      
      case 'LEADERBOARD_RANK':
        return this.checkLeaderboardRankCriteria(criteria, user);
      
      case 'TAX_SAVINGS':
        return this.checkTaxSavingsCriteria(criteria, user);
      
      case 'EDUCATION_MODULES':
        return this.checkEducationModulesCriteria(criteria, user);
      
      default:
        return false;
    }
  }

  /**
   * Check SIP started criteria
   */
  checkSIPStartedCriteria(criteria, action, data) {
    return action === 'SIP_STARTED' && data.count >= criteria.count;
  }

  /**
   * Check SIP streak criteria
   */
  checkSIPStreakCriteria(criteria, user) {
    const sipHistory = user.sipHistory || [];
    let currentStreak = 0;
    let maxStreak = 0;

    for (let i = sipHistory.length - 1; i >= 0; i--) {
      const sip = sipHistory[i];
      const daysDiff = Math.floor((new Date() - new Date(sip.date)) / (1000 * 60 * 60 * 24));
      
      if (daysDiff <= 1) {
        currentStreak++;
        maxStreak = Math.max(maxStreak, currentStreak);
      } else {
        break;
      }
    }

    return maxStreak >= criteria.count;
  }

  /**
   * Check portfolio value criteria
   */
  checkPortfolioValueCriteria(criteria, user) {
    const portfolioValue = user.portfolioValue || 0;
    return portfolioValue >= criteria.amount;
  }

  /**
   * Check fund categories criteria
   */
  checkFundCategoriesCriteria(criteria, user) {
    const holdings = user.holdings || [];
    const categories = new Set(holdings.map(h => h.category));
    return categories.size >= criteria.count;
  }

  /**
   * Check referrals criteria
   */
  checkReferralsCriteria(criteria, user) {
    const successfulReferrals = user.successfulReferrals || 0;
    return successfulReferrals >= criteria.count;
  }

  /**
   * Check leaderboard rank criteria
   */
  checkLeaderboardRankCriteria(criteria, user) {
    const bestRank = user.bestLeaderboardRank || 999;
    return bestRank <= criteria.rank;
  }

  /**
   * Check tax savings criteria
   */
  checkTaxSavingsCriteria(criteria, user) {
    const taxSavings = user.taxSavings || 0;
    return taxSavings >= criteria.amount;
  }

  /**
   * Check education modules criteria
   */
  checkEducationModulesCriteria(criteria, user) {
    const completedModules = user.completedEducationModules || 0;
    return completedModules >= criteria.count;
  }

  /**
   * Award achievement to user
   */
  async awardAchievement(userId, achievement) {
    try {
      const user = await User.findOne({ supabaseId: userId });
      if (!user) return;

      // Add achievement to user
      user.achievements = user.achievements || [];
      user.achievements.push(achievement.id);

      // Add points
      user.points = (user.points || 0) + achievement.points;

      // Update level
      user.level = this.calculateLevel(user.points);

      await user.save();

      // Create achievement record
      const achievementRecord = new Achievement({
        userId,
        achievementId: achievement.id,
        name: achievement.name,
        description: achievement.description,
        category: achievement.category,
        points: achievement.points,
        awardedAt: new Date()
      });

      await achievementRecord.save();

      logger.info(`Achievement awarded: ${achievement.id} to user: ${userId}`);
    } catch (error) {
      logger.error('Error awarding achievement:', error);
    }
  }

  /**
   * Calculate user level based on points
   */
  calculateLevel(points) {
    if (points < 100) return 1;
    if (points < 500) return 2;
    if (points < 1000) return 3;
    if (points < 2000) return 4;
    if (points < 5000) return 5;
    if (points < 10000) return 6;
    if (points < 20000) return 7;
    if (points < 50000) return 8;
    if (points < 100000) return 9;
    return 10;
  }

  /**
   * Generate behavioral nudges
   */
  async generateBehavioralNudges(userId, context) {
    try {
      const user = await User.findOne({ supabaseId: userId });
      if (!user) return [];

      const nudges = [];

      for (const [nudgeId, nudge] of this.behavioralNudges) {
        if (this.shouldTriggerNudge(nudge, user, context)) {
          nudges.push({
            id: nudgeId,
            name: nudge.name,
            description: nudge.description,
            message: nudge.message,
            action: nudge.action,
            type: nudge.type,
            priority: this.calculateNudgePriority(nudge, user)
          });
        }
      }

      // Sort by priority
      nudges.sort((a, b) => b.priority - a.priority);

      return nudges;
    } catch (error) {
      logger.error('Error generating behavioral nudges:', error);
      return [];
    }
  }

  /**
   * Check if nudge should be triggered
   */
  shouldTriggerNudge(nudge, user, context) {
    const trigger = nudge.trigger;

    switch (trigger.type) {
      case 'SIP_MISSED':
        return this.checkSIPMissedTrigger(trigger, user);
      
      case 'MARKET_DIP':
        return this.checkMarketDipTrigger(trigger, context);
      
      case 'LOW_DIVERSIFICATION':
        return this.checkLowDiversificationTrigger(trigger, user);
      
      case 'GOAL_MILESTONE':
        return this.checkGoalMilestoneTrigger(trigger, user);
      
      default:
        return false;
    }
  }

  /**
   * Check SIP missed trigger
   */
  checkSIPMissedTrigger(trigger, user) {
    const lastSIP = user.lastSIPDate;
    if (!lastSIP) return false;

    const daysSinceLastSIP = Math.floor((new Date() - new Date(lastSIP)) / (1000 * 60 * 60 * 24));
    return daysSinceLastSIP >= trigger.days;
  }

  /**
   * Check market dip trigger
   */
  checkMarketDipTrigger(trigger, context) {
    const marketData = context.marketData;
    if (!marketData) return false;

    const marketChange = marketData.changePercent || 0;
    return marketChange <= -trigger.percentage;
  }

  /**
   * Check low diversification trigger
   */
  checkLowDiversificationTrigger(trigger, user) {
    const holdings = user.holdings || [];
    const categories = new Set(holdings.map(h => h.category));
    return categories.size <= trigger.categories;
  }

  /**
   * Check goal milestone trigger
   */
  checkGoalMilestoneTrigger(trigger, user) {
    const goals = user.financialGoals || [];
    
    for (const goal of goals) {
      const progress = (goal.currentAmount / goal.targetAmount) * 100;
      if (progress >= trigger.percentage && progress < trigger.percentage + 25) {
        return true;
      }
    }
    
    return false;
  }

  /**
   * Calculate nudge priority
   */
  calculateNudgePriority(nudge, user) {
    let priority = 50; // Base priority

    // Adjust based on user behavior
    switch (nudge.type) {
      case 'REMINDER':
        priority += 20;
        break;
      case 'OPPORTUNITY':
        priority += 30;
        break;
      case 'EDUCATION':
        priority += 10;
        break;
      case 'CELEBRATION':
        priority += 15;
        break;
    }

    // Adjust based on user level
    const userLevel = user.level || 1;
    if (userLevel < 3) priority += 10; // Higher priority for new users

    return priority;
  }

  /**
   * Track user engagement
   */
  async trackEngagement(userId, action, data) {
    try {
      const user = await User.findOne({ supabaseId: userId });
      if (!user) return;

      // Update engagement metrics
      user.engagementMetrics = user.engagementMetrics || {};
      user.engagementMetrics.lastActivity = new Date();
      user.engagementMetrics.totalActions = (user.engagementMetrics.totalActions || 0) + 1;

      // Track specific actions
      user.engagementMetrics.actions = user.engagementMetrics.actions || {};
      user.engagementMetrics.actions[action] = (user.engagementMetrics.actions[action] || 0) + 1;

      // Update streak
      await this.updateEngagementStreak(user);

      await user.save();

      logger.info(`Engagement tracked: ${action} for user: ${userId}`);
    } catch (error) {
      logger.error('Error tracking engagement:', error);
    }
  }

  /**
   * Update engagement streak
   */
  async updateEngagementStreak(user) {
    const lastActivity = user.engagementMetrics?.lastActivity;
    const currentStreak = user.engagementMetrics?.currentStreak || 0;

    if (lastActivity) {
      const daysDiff = Math.floor((new Date() - new Date(lastActivity)) / (1000 * 60 * 60 * 24));
      
      if (daysDiff <= 1) {
        user.engagementMetrics.currentStreak = currentStreak + 1;
        user.engagementMetrics.maxStreak = Math.max(
          user.engagementMetrics.maxStreak || 0,
          user.engagementMetrics.currentStreak
        );
      } else if (daysDiff > 1) {
        user.engagementMetrics.currentStreak = 1;
      }
    }
  }

  /**
   * Get user gamification profile
   */
  async getUserProfile(userId) {
    try {
      const user = await User.findOne({ supabaseId: userId });
      if (!user) return null;

      const profile = {
        level: user.level || 1,
        points: user.points || 0,
        achievements: user.achievements || [],
        badges: await this.getUserBadges(user),
        streak: user.engagementMetrics?.currentStreak || 0,
        maxStreak: user.engagementMetrics?.maxStreak || 0,
        totalActions: user.engagementMetrics?.totalActions || 0,
        rank: await this.getUserRank(userId),
        nextLevel: this.getNextLevelInfo(user.level || 1, user.points || 0)
      };

      return profile;
    } catch (error) {
      logger.error('Error getting user profile:', error);
      return null;
    }
  }

  /**
   * Get user badges
   */
  async getUserBadges(user) {
    const badges = [];
    const achievements = user.achievements || [];

    // Map achievements to badges
    for (const achievementId of achievements) {
      const achievement = this.achievements.get(achievementId);
      if (achievement) {
        badges.push({
          id: achievementId,
          name: achievement.name,
          icon: achievement.icon,
          category: achievement.category,
          description: achievement.description
        });
      }
    }

    return badges;
  }

  /**
   * Get user rank
   */
  async getUserRank(userId) {
    try {
      const users = await User.find({}).sort({ points: -1 });
      const userIndex = users.findIndex(user => user.supabaseId === userId);
      return userIndex >= 0 ? userIndex + 1 : null;
    } catch (error) {
      logger.error('Error getting user rank:', error);
      return null;
    }
  }

  /**
   * Get next level information
   */
  getNextLevelInfo(currentLevel, currentPoints) {
    const levelThresholds = [0, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000];
    const nextLevel = currentLevel + 1;
    
    if (nextLevel > 10) {
      return { level: 10, pointsNeeded: 0, isMaxLevel: true };
    }

    const pointsNeeded = levelThresholds[nextLevel - 1] - currentPoints;
    
    return {
      level: nextLevel,
      pointsNeeded: Math.max(0, pointsNeeded),
      isMaxLevel: false
    };
  }

  /**
   * Get leaderboard
   */
  async getLeaderboard(limit = 50) {
    try {
      const users = await User.find({})
        .sort({ points: -1 })
        .limit(limit)
        .select('supabaseId name points level achievements');

      return users.map((user, index) => ({
        rank: index + 1,
        userId: user.supabaseId,
        name: user.name,
        points: user.points || 0,
        level: user.level || 1,
        achievements: user.achievements?.length || 0
      }));
    } catch (error) {
      logger.error('Error getting leaderboard:', error);
      return [];
    }
  }
}

module.exports = GamificationService; 