const logger = require('../utils/logger');
const User = require('../models/User');
const UserPortfolio = require('../models/UserPortfolio');
const PortfolioCopy = require('../models/PortfolioCopy');

class SocialTradingService {
  constructor() {
    this.communityFeatures = {
      discussions: new Map(),
      challenges: new Map(),
      expertAdvisors: new Map(),
      socialFeed: []
    };
  }

  /**
   * Initialize social trading service
   */
  async initialize() {
    try {
      await this.loadCommunityData();
      logger.info('Social Trading Service initialized successfully');
      return true;
    } catch (error) {
      logger.error('Failed to initialize Social Trading Service:', error);
      return false;
    }
  }

  /**
   * Load community data
   */
  async loadCommunityData() {
    // Load discussions, challenges, and expert advisors
    this.communityFeatures.discussions = new Map();
    this.communityFeatures.challenges = new Map();
    this.communityFeatures.expertAdvisors = new Map();
    
    logger.info('Community data loaded successfully');
  }

  /**
   * Create a new discussion
   */
  async createDiscussion(userId, discussionData) {
    try {
      const discussion = {
        id: this.generateId(),
        userId,
        title: discussionData.title,
        content: discussionData.content,
        category: discussionData.category,
        tags: discussionData.tags || [],
        createdAt: new Date(),
        updatedAt: new Date(),
        likes: 0,
        replies: [],
        isActive: true
      };

      this.communityFeatures.discussions.set(discussion.id, discussion);
      
      // Add to social feed
      this.addToSocialFeed({
        type: 'DISCUSSION_CREATED',
        userId,
        discussionId: discussion.id,
        title: discussion.title,
        timestamp: new Date()
      });

      logger.info(`Discussion created: ${discussion.id} by user: ${userId}`);
      return discussion;
    } catch (error) {
      logger.error('Error creating discussion:', error);
      return null;
    }
  }

  /**
   * Reply to a discussion
   */
  async replyToDiscussion(discussionId, userId, replyData) {
    try {
      const discussion = this.communityFeatures.discussions.get(discussionId);
      if (!discussion) {
        throw new Error('Discussion not found');
      }

      const reply = {
        id: this.generateId(),
        userId,
        content: replyData.content,
        createdAt: new Date(),
        likes: 0,
        isExpertReply: replyData.isExpertReply || false
      };

      discussion.replies.push(reply);
      discussion.updatedAt = new Date();

      // Add to social feed
      this.addToSocialFeed({
        type: 'DISCUSSION_REPLY',
        userId,
        discussionId,
        replyId: reply.id,
        timestamp: new Date()
      });

      logger.info(`Reply added to discussion: ${discussionId} by user: ${userId}`);
      return reply;
    } catch (error) {
      logger.error('Error replying to discussion:', error);
      return null;
    }
  }

  /**
   * Like a discussion or reply
   */
  async likeContent(contentId, userId, contentType = 'discussion') {
    try {
      if (contentType === 'discussion') {
        const discussion = this.communityFeatures.discussions.get(contentId);
        if (discussion) {
          discussion.likes++;
        }
      } else if (contentType === 'reply') {
        // Find reply in discussions
        for (const discussion of this.communityFeatures.discussions.values()) {
          const reply = discussion.replies.find(r => r.id === contentId);
          if (reply) {
            reply.likes++;
            break;
          }
        }
      }

      logger.info(`Content liked: ${contentId} by user: ${userId}`);
      return { success: true };
    } catch (error) {
      logger.error('Error liking content:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Create an investment challenge
   */
  async createChallenge(userId, challengeData) {
    try {
      const challenge = {
        id: this.generateId(),
        creatorId: userId,
        title: challengeData.title,
        description: challengeData.description,
        startDate: new Date(challengeData.startDate),
        endDate: new Date(challengeData.endDate),
        initialAmount: challengeData.initialAmount,
        maxParticipants: challengeData.maxParticipants || 100,
        participants: [],
        leaderboard: [],
        rules: challengeData.rules || [],
        category: challengeData.category,
        isActive: true,
        createdAt: new Date()
      };

      this.communityFeatures.challenges.set(challenge.id, challenge);
      
      // Add to social feed
      this.addToSocialFeed({
        type: 'CHALLENGE_CREATED',
        userId,
        challengeId: challenge.id,
        title: challenge.title,
        timestamp: new Date()
      });

      logger.info(`Challenge created: ${challenge.id} by user: ${userId}`);
      return challenge;
    } catch (error) {
      logger.error('Error creating challenge:', error);
      return null;
    }
  }

  /**
   * Join an investment challenge
   */
  async joinChallenge(challengeId, userId) {
    try {
      const challenge = this.communityFeatures.challenges.get(challengeId);
      if (!challenge) {
        throw new Error('Challenge not found');
      }

      if (!challenge.isActive) {
        throw new Error('Challenge is not active');
      }

      if (challenge.participants.length >= challenge.maxParticipants) {
        throw new Error('Challenge is full');
      }

      if (challenge.participants.includes(userId)) {
        throw new Error('Already participating in this challenge');
      }

      const participant = {
        userId,
        joinedAt: new Date(),
        currentValue: challenge.initialAmount,
        transactions: [],
        performance: 0
      };

      challenge.participants.push(userId);
      challenge.leaderboard.push(participant);

      // Add to social feed
      this.addToSocialFeed({
        type: 'CHALLENGE_JOINED',
        userId,
        challengeId,
        timestamp: new Date()
      });

      logger.info(`User ${userId} joined challenge: ${challengeId}`);
      return participant;
    } catch (error) {
      logger.error('Error joining challenge:', error);
      return null;
    }
  }

  /**
   * Update challenge leaderboard
   */
  async updateChallengeLeaderboard(challengeId) {
    try {
      const challenge = this.communityFeatures.challenges.get(challengeId);
      if (!challenge) return;

      // Sort participants by performance
      challenge.leaderboard.sort((a, b) => b.performance - a.performance);

      // Update rankings
      challenge.leaderboard.forEach((participant, index) => {
        participant.rank = index + 1;
      });

      logger.info(`Challenge leaderboard updated: ${challengeId}`);
    } catch (error) {
      logger.error('Error updating challenge leaderboard:', error);
    }
  }

  /**
   * Register as expert advisor
   */
  async registerExpertAdvisor(userId, expertData) {
    try {
      const expert = {
        id: this.generateId(),
        userId,
        name: expertData.name,
        bio: expertData.bio,
        expertise: expertData.expertise || [],
        credentials: expertData.credentials || [],
        rating: 0,
        totalReviews: 0,
        followers: [],
        portfolio: expertData.portfolio || {},
        isVerified: false,
        isActive: true,
        createdAt: new Date()
      };

      this.communityFeatures.expertAdvisors.set(expert.id, expert);
      
      // Add to social feed
      this.addToSocialFeed({
        type: 'EXPERT_REGISTERED',
        userId,
        expertId: expert.id,
        name: expert.name,
        timestamp: new Date()
      });

      logger.info(`Expert advisor registered: ${expert.id} by user: ${userId}`);
      return expert;
    } catch (error) {
      logger.error('Error registering expert advisor:', error);
      return null;
    }
  }

  /**
   * Follow an expert advisor
   */
  async followExpert(expertId, userId) {
    try {
      const expert = this.communityFeatures.expertAdvisors.get(expertId);
      if (!expert) {
        throw new Error('Expert advisor not found');
      }

      if (expert.followers.includes(userId)) {
        throw new Error('Already following this expert');
      }

      expert.followers.push(userId);

      // Add to social feed
      this.addToSocialFeed({
        type: 'EXPERT_FOLLOWED',
        userId,
        expertId,
        expertName: expert.name,
        timestamp: new Date()
      });

      logger.info(`User ${userId} started following expert: ${expertId}`);
      return { success: true };
    } catch (error) {
      logger.error('Error following expert:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Rate an expert advisor
   */
  async rateExpert(expertId, userId, rating, review) {
    try {
      const expert = this.communityFeatures.expertAdvisors.get(expertId);
      if (!expert) {
        throw new Error('Expert advisor not found');
      }

      // Update rating
      const totalRating = expert.rating * expert.totalReviews + rating;
      expert.totalReviews++;
      expert.rating = totalRating / expert.totalReviews;

      // Add review
      expert.reviews = expert.reviews || [];
      expert.reviews.push({
        userId,
        rating,
        review,
        createdAt: new Date()
      });

      logger.info(`Expert rated: ${expertId} by user: ${userId} with rating: ${rating}`);
      return { success: true };
    } catch (error) {
      logger.error('Error rating expert:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Copy a portfolio
   */
  async copyPortfolio(sourceUserId, targetUserId, copyData) {
    try {
      // Get source portfolio
      const sourcePortfolio = await UserPortfolio.findOne({ userId: sourceUserId });
      if (!sourcePortfolio) {
        throw new Error('Source portfolio not found');
      }

      // Create copy record
      const portfolioCopy = new PortfolioCopy({
        sourceUserId,
        targetUserId,
        investmentType: copyData.investmentType || 'SIP',
        averageSip: copyData.averageSip,
        copiedAllocation: sourcePortfolio.allocation,
        sourceReturnPercent: sourcePortfolio.xirr1Y || 0,
        duration: copyData.duration || '1Y',
        status: 'PENDING',
        executionDetails: {
          startDate: new Date(),
          expectedCompletion: new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours
        },
        metadata: {
          copyReason: copyData.reason,
          riskTolerance: copyData.riskTolerance,
          investmentGoals: copyData.goals
        }
      });

      await portfolioCopy.save();

      // Add to social feed
      this.addToSocialFeed({
        type: 'PORTFOLIO_COPIED',
        userId: targetUserId,
        sourceUserId,
        copyId: portfolioCopy._id,
        timestamp: new Date()
      });

      logger.info(`Portfolio copy initiated: ${portfolioCopy._id} from ${sourceUserId} to ${targetUserId}`);
      return portfolioCopy;
    } catch (error) {
      logger.error('Error copying portfolio:', error);
      return null;
    }
  }

  /**
   * Execute portfolio copy
   */
  async executePortfolioCopy(copyId) {
    try {
      const portfolioCopy = await PortfolioCopy.findById(copyId);
      if (!portfolioCopy) {
        throw new Error('Portfolio copy not found');
      }

      // Get source portfolio allocation
      const sourcePortfolio = await UserPortfolio.findOne({ userId: portfolioCopy.sourceUserId });
      if (!sourcePortfolio) {
        throw new Error('Source portfolio not found');
      }

      // Create target portfolio or update existing
      let targetPortfolio = await UserPortfolio.findOne({ userId: portfolioCopy.targetUserId });
      
      if (!targetPortfolio) {
        targetPortfolio = new UserPortfolio({
          userId: portfolioCopy.targetUserId,
          funds: [],
          allocation: portfolioCopy.copiedAllocation,
          totalInvested: 0,
          totalCurrentValue: 0
        });
      } else {
        // Update allocation
        targetPortfolio.allocation = portfolioCopy.copiedAllocation;
      }

      await targetPortfolio.save();

      // Update copy status
      portfolioCopy.status = 'COMPLETED';
      portfolioCopy.executionDetails.completedAt = new Date();
      await portfolioCopy.save();

      logger.info(`Portfolio copy executed: ${copyId}`);
      return portfolioCopy;
    } catch (error) {
      logger.error('Error executing portfolio copy:', error);
      return null;
    }
  }

  /**
   * Get copy trading recommendations
   */
  async getCopyTradingRecommendations(userId, preferences) {
    try {
      // Find top performers based on preferences
      const topPerformers = await UserPortfolio.find({
        'xirr1Y': { $gt: 15 }, // 15%+ returns
        'totalInvested': { $gt: 100000 } // Minimum investment
      })
      .sort({ 'xirr1Y': -1 })
      .limit(10);

      const recommendations = topPerformers.map(portfolio => ({
        userId: portfolio.userId,
        secretCode: portfolio.secretCode,
        performance: {
          xirr1Y: portfolio.xirr1Y,
          xirr3Y: portfolio.xirr3Y,
          totalReturn: portfolio.totalCurrentValue - portfolio.totalInvested
        },
        allocation: portfolio.allocation,
        riskScore: this.calculateRiskScore(portfolio),
        suitability: this.calculateSuitability(portfolio, preferences)
      }));

      // Sort by suitability
      recommendations.sort((a, b) => b.suitability - a.suitability);

      return recommendations;
    } catch (error) {
      logger.error('Error getting copy trading recommendations:', error);
      return [];
    }
  }

  /**
   * Calculate risk score for portfolio
   */
  calculateRiskScore(portfolio) {
    // Calculate risk based on allocation and volatility
    let riskScore = 0;
    
    for (const [fund, allocation] of Object.entries(portfolio.allocation)) {
      if (fund.includes('Small Cap')) riskScore += allocation * 0.8;
      else if (fund.includes('Mid Cap')) riskScore += allocation * 0.6;
      else if (fund.includes('Large Cap')) riskScore += allocation * 0.4;
      else if (fund.includes('Debt')) riskScore += allocation * 0.2;
    }

    return Math.min(100, riskScore);
  }

  /**
   * Calculate suitability score
   */
  calculateSuitability(portfolio, preferences) {
    let suitability = 0;
    
    // Risk tolerance match
    const riskScore = this.calculateRiskScore(portfolio);
    const riskMatch = 1 - Math.abs(riskScore - preferences.riskTolerance) / 100;
    suitability += riskMatch * 0.4;

    // Performance match
    const performanceMatch = Math.min(portfolio.xirr1Y / 20, 1); // Normalize to 20%
    suitability += performanceMatch * 0.4;

    // Investment amount match
    const amountMatch = portfolio.totalInvested >= preferences.minInvestment ? 1 : 0;
    suitability += amountMatch * 0.2;

    return suitability;
  }

  /**
   * Get social feed
   */
  async getSocialFeed(userId, limit = 20) {
    try {
      // Get personalized feed based on user's interests and connections
      const user = await User.findOne({ supabaseId: userId });
      if (!user) return [];

      // Filter feed based on user preferences
      const filteredFeed = this.communityFeatures.socialFeed
        .filter(item => this.isRelevantToUser(item, user))
        .sort((a, b) => b.timestamp - a.timestamp)
        .slice(0, limit);

      return filteredFeed;
    } catch (error) {
      logger.error('Error getting social feed:', error);
      return [];
    }
  }

  /**
   * Check if feed item is relevant to user
   */
  isRelevantToUser(item, user) {
    // Check if user follows the content creator
    // Check if content matches user interests
    // Check if content is from user's connections
    return true; // Simplified for now
  }

  /**
   * Add item to social feed
   */
  addToSocialFeed(item) {
    this.communityFeatures.socialFeed.push(item);
    
    // Keep feed size manageable
    if (this.communityFeatures.socialFeed.length > 1000) {
      this.communityFeatures.socialFeed = this.communityFeatures.socialFeed.slice(-500);
    }
  }

  /**
   * Get community statistics
   */
  async getCommunityStats() {
    try {
      const stats = {
        totalUsers: await User.countDocuments(),
        activeDiscussions: Array.from(this.communityFeatures.discussions.values()).filter(d => d.isActive).length,
        activeChallenges: Array.from(this.communityFeatures.challenges.values()).filter(c => c.isActive).length,
        expertAdvisors: Array.from(this.communityFeatures.expertAdvisors.values()).filter(e => e.isActive).length,
        totalPortfolioCopies: await PortfolioCopy.countDocuments(),
        socialFeedItems: this.communityFeatures.socialFeed.length
      };

      return stats;
    } catch (error) {
      logger.error('Error getting community stats:', error);
      return {};
    }
  }

  /**
   * Generate unique ID
   */
  generateId() {
    return `social_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

module.exports = SocialTradingService; 