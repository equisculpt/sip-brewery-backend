const express = require('express');
const router = express.Router();
const { body, validationResult } = require('express-validator');
const AddictiveCommunityEngine = require('../services/AddictiveCommunityEngine');
const auth = require('../middleware/auth');
const logger = require('../utils/logger');
const response = require('../utils/response');

/**
 * ADDICTIVE COMMUNITY API ROUTES
 * The Most Engaging Financial Community Backend
 * 
 * These routes power the most addictive financial community platform
 * designed to keep users engaged 24/7 with their office page always open.
 * 
 * Features:
 * - Real-time live discussions with WebSocket support
 * - Gamified point system and achievement tracking
 * - Exclusive alpha distribution with FOMO triggers
 * - Live market events and competitions
 * - Social trading groups and portfolio battles
 * - Fund manager direct interactions
 * - Addiction psychology implementation
 * - Professional networking and knowledge sharing
 */

// Initialize the addictive community engine
const communityEngine = new AddictiveCommunityEngine();

/**
 * @route GET /api/community/status
 * @desc Get community health and engagement metrics
 * @access Public
 */
router.get('/status', async (req, res) => {
  try {
    logger.info('📊 Fetching community health metrics...');
    
    const healthMetrics = await communityEngine.monitorCommunityHealth();
    
    const status = {
      community_health: 'EXCELLENT',
      addiction_level: healthMetrics.addiction_level,
      active_users: healthMetrics.active_users,
      live_discussions: healthMetrics.live_discussions,
      market_events: healthMetrics.market_events,
      engagement_rate: healthMetrics.engagement_rate,
      office_always_open_rate: healthMetrics.office_always_open_rate,
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      features: {
        real_time_discussions: true,
        gamification_system: true,
        exclusive_alpha: true,
        live_events: true,
        social_trading: true,
        fund_manager_access: true,
        addiction_psychology: true,
        professional_networking: true
      }
    };
    
    return response.success(res, 'Community status retrieved successfully', status);
    
  } catch (error) {
    logger.error('❌ Failed to get community status:', error);
    return response.error(res, 'Failed to get community status', error.message, 500);
  }
});

/**
 * @route GET /api/community/discussions/live
 * @desc Get all live discussions with real-time updates
 * @access Public
 */
router.get('/discussions/live', async (req, res) => {
  try {
    logger.info('🔥 Fetching live discussions...');
    
    const { urgency, exclusivity, limit = 20 } = req.query;
    
    // Mock live discussions data (in production, this would come from database)
    const liveDiscussions = [
      {
        id: 'disc_001',
        title: '🚨 BREAKING: Adani Group Stocks Surge 15% - Live Reactions',
        type: 'BREAKING_NEWS',
        participants: 1247,
        urgency: 'CRITICAL',
        exclusivity: 'PUBLIC',
        created_at: new Date(Date.now() - 300000), // 5 minutes ago
        last_activity: new Date(Date.now() - 30000), // 30 seconds ago
        message_count: 156,
        trending_score: 0.98,
        addiction_factor: 0.95,
        tags: ['adani', 'breaking', 'surge', 'reactions']
      },
      {
        id: 'disc_002',
        title: '📊 Live Market Pulse: Nifty Testing 22,000 Resistance',
        type: 'MARKET_PULSE',
        participants: 892,
        urgency: 'HIGH',
        exclusivity: 'PREMIUM',
        created_at: new Date(Date.now() - 600000), // 10 minutes ago
        last_activity: new Date(Date.now() - 15000), // 15 seconds ago
        message_count: 89,
        trending_score: 0.87,
        addiction_factor: 0.92,
        tags: ['nifty', 'resistance', 'technical', 'live']
      },
      {
        id: 'disc_003',
        title: '💰 Fund Manager Exclusive: Portfolio Rebalancing Strategies',
        type: 'FUND_MANAGER_INSIGHTS',
        participants: 456,
        urgency: 'MEDIUM',
        exclusivity: 'VIP',
        created_at: new Date(Date.now() - 900000), // 15 minutes ago
        last_activity: new Date(Date.now() - 60000), // 1 minute ago
        message_count: 34,
        trending_score: 0.79,
        addiction_factor: 0.89,
        tags: ['fund-manager', 'portfolio', 'rebalancing', 'exclusive']
      },
      {
        id: 'disc_004',
        title: '🚀 IPO Analysis: Tata Technologies Valuation Deep Dive',
        type: 'IPO_ANALYSIS',
        participants: 678,
        urgency: 'HIGH',
        exclusivity: 'PREMIUM',
        created_at: new Date(Date.now() - 1200000), // 20 minutes ago
        last_activity: new Date(Date.now() - 45000), // 45 seconds ago
        message_count: 67,
        trending_score: 0.84,
        addiction_factor: 0.91,
        tags: ['ipo', 'tata-technologies', 'valuation', 'analysis']
      }
    ];
    
    // Filter based on query parameters
    let filteredDiscussions = liveDiscussions;
    
    if (urgency) {
      filteredDiscussions = filteredDiscussions.filter(d => d.urgency === urgency.toUpperCase());
    }
    
    if (exclusivity) {
      filteredDiscussions = filteredDiscussions.filter(d => d.exclusivity === exclusivity.toUpperCase());
    }
    
    // Sort by trending score and addiction factor
    filteredDiscussions.sort((a, b) => (b.trending_score + b.addiction_factor) - (a.trending_score + a.addiction_factor));
    
    // Limit results
    filteredDiscussions = filteredDiscussions.slice(0, parseInt(limit));
    
    return response.success(res, 'Live discussions retrieved successfully', {
      discussions: filteredDiscussions,
      total_count: filteredDiscussions.length,
      engagement_metrics: {
        total_participants: filteredDiscussions.reduce((sum, d) => sum + d.participants, 0),
        average_addiction_factor: filteredDiscussions.reduce((sum, d) => sum + d.addiction_factor, 0) / filteredDiscussions.length,
        trending_discussions: filteredDiscussions.filter(d => d.trending_score > 0.8).length
      }
    });
    
  } catch (error) {
    logger.error('❌ Failed to get live discussions:', error);
    return response.error(res, 'Failed to get live discussions', error.message, 500);
  }
});

/**
 * @route GET /api/community/discussions/:discussionId/messages
 * @desc Get messages for a specific discussion
 * @access Public
 */
router.get('/discussions/:discussionId/messages', async (req, res) => {
  try {
    const { discussionId } = req.params;
    const { limit = 50, offset = 0 } = req.query;
    
    logger.info(`💬 Fetching messages for discussion: ${discussionId}`);
    
    // Mock messages data (in production, this would come from database)
    const messages = [
      {
        id: 'msg_001',
        discussion_id: discussionId,
        user_id: 'fm_001',
        user_name: 'Prashant Jain',
        user_role: 'FUND_MANAGER',
        user_level: 25,
        content: '🔥 This is exactly what I predicted last week! Check my analysis from Tuesday. The fundamentals were screaming buy signal.',
        timestamp: new Date(Date.now() - 120000),
        likes: 234,
        replies: 45,
        exclusive: false,
        alpha: true,
        verified: true,
        addiction_trigger: 'EXPERT_VALIDATION'
      },
      {
        id: 'msg_002',
        discussion_id: discussionId,
        user_id: 'analyst_001',
        user_name: 'Motilal Oswal',
        user_role: 'ANALYST',
        user_level: 22,
        content: '💎 EXCLUSIVE ALPHA: More positive news coming in next 48 hours. My sources confirm regulatory approval for the new project. This could be a game-changer!',
        timestamp: new Date(Date.now() - 60000),
        likes: 567,
        replies: 89,
        exclusive: true,
        alpha: true,
        verified: true,
        addiction_trigger: 'EXCLUSIVE_INFORMATION'
      },
      {
        id: 'msg_003',
        discussion_id: discussionId,
        user_id: 'investor_001',
        user_name: 'Rakesh Jhunjhunwala Jr.',
        user_role: 'INVESTOR',
        user_level: 18,
        content: 'Completely agree with @PrashantJain! I\'ve been accumulating since last month. The risk-reward ratio is fantastic at these levels.',
        timestamp: new Date(Date.now() - 30000),
        likes: 123,
        replies: 23,
        exclusive: false,
        alpha: false,
        verified: true,
        addiction_trigger: 'SOCIAL_VALIDATION'
      }
    ];
    
    return response.success(res, 'Discussion messages retrieved successfully', {
      messages,
      discussion_id: discussionId,
      total_count: messages.length,
      pagination: {
        limit: parseInt(limit),
        offset: parseInt(offset),
        has_more: false
      }
    });
    
  } catch (error) {
    logger.error('❌ Failed to get discussion messages:', error);
    return response.error(res, 'Failed to get discussion messages', error.message, 500);
  }
});

/**
 * @route POST /api/community/discussions/:discussionId/messages
 * @desc Post a new message to a discussion
 * @access Private
 */
router.post('/discussions/:discussionId/messages', 
  auth,
  [
    body('content').notEmpty().withMessage('Message content is required'),
    body('content').isLength({ min: 1, max: 2000 }).withMessage('Message must be between 1 and 2000 characters')
  ],
  async (req, res) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return response.error(res, 'Validation failed', errors.array(), 400);
      }
      
      const { discussionId } = req.params;
      const { content, alpha = false, exclusive = false } = req.body;
      const userId = req.user.id;
      
      logger.info(`📝 User ${userId} posting message to discussion ${discussionId}`);
      
      // Create new message
      const newMessage = {
        id: `msg_${Date.now()}`,
        discussion_id: discussionId,
        user_id: userId,
        user_name: req.user.name || 'Anonymous',
        user_role: req.user.role || 'INVESTOR',
        user_level: req.user.level || 1,
        content,
        timestamp: new Date(),
        likes: 0,
        replies: 0,
        exclusive,
        alpha,
        verified: req.user.verified || false,
        addiction_trigger: 'USER_PARTICIPATION'
      };
      
      // Track user engagement
      await communityEngine.trackUserEngagement(userId, 'POST_MESSAGE', {
        discussion_id: discussionId,
        content_length: content.length,
        alpha,
        exclusive
      });
      
      // Award points for participation
      const pointsAwarded = alpha ? 100 : exclusive ? 75 : 25;
      
      return response.success(res, 'Message posted successfully', {
        message: newMessage,
        points_awarded: pointsAwarded,
        engagement_bonus: true
      });
      
    } catch (error) {
      logger.error('❌ Failed to post message:', error);
      return response.error(res, 'Failed to post message', error.message, 500);
    }
  }
);

/**
 * @route GET /api/community/events/live
 * @desc Get live market events and competitions
 * @access Public
 */
router.get('/events/live', async (req, res) => {
  try {
    logger.info('🎪 Fetching live market events...');
    
    const liveEvents = [
      {
        id: 'event_001',
        title: '🎪 TCS Earnings Live Reaction Party',
        type: 'EARNINGS_LIVE',
        description: 'Join thousands of investors for live reactions to TCS Q3 earnings',
        participants: 2341,
        max_participants: 5000,
        reward: 500,
        time_left: 1800, // 30 minutes
        status: 'LIVE',
        urgency: 'HIGH',
        addiction_factor: 0.96,
        created_at: new Date(Date.now() - 3600000), // 1 hour ago
        features: ['Live chat', 'Expert analysis', 'Prediction contests', 'Instant rewards']
      },
      {
        id: 'event_002',
        title: '🚀 IPO Prediction Tournament: Tata Technologies',
        type: 'IPO_LISTING',
        description: 'Predict the listing price and win exclusive rewards',
        participants: 1876,
        max_participants: 3000,
        reward: 750,
        time_left: 7200, // 2 hours
        status: 'UPCOMING',
        urgency: 'MEDIUM',
        addiction_factor: 0.94,
        created_at: new Date(Date.now() - 1800000), // 30 minutes ago
        features: ['Price prediction', 'Expert insights', 'Leaderboard', 'Exclusive alpha']
      },
      {
        id: 'event_003',
        title: '📊 Market Crash Simulation Challenge',
        type: 'MARKET_CRASH',
        description: 'Test your portfolio management skills in a simulated crash',
        participants: 892,
        max_participants: 2000,
        reward: 1000,
        time_left: 10800, // 3 hours
        status: 'UPCOMING',
        urgency: 'LOW',
        addiction_factor: 0.91,
        created_at: new Date(Date.now() - 900000), // 15 minutes ago
        features: ['Portfolio simulation', 'Risk management', 'Strategy testing', 'Learning rewards']
      }
    ];
    
    return response.success(res, 'Live events retrieved successfully', {
      events: liveEvents,
      total_count: liveEvents.length,
      engagement_metrics: {
        total_participants: liveEvents.reduce((sum, e) => sum + e.participants, 0),
        live_events: liveEvents.filter(e => e.status === 'LIVE').length,
        upcoming_events: liveEvents.filter(e => e.status === 'UPCOMING').length,
        average_addiction_factor: liveEvents.reduce((sum, e) => sum + e.addiction_factor, 0) / liveEvents.length
      }
    });
    
  } catch (error) {
    logger.error('❌ Failed to get live events:', error);
    return response.error(res, 'Failed to get live events', error.message, 500);
  }
});

/**
 * @route POST /api/community/events/:eventId/join
 * @desc Join a live market event
 * @access Private
 */
router.post('/events/:eventId/join', auth, async (req, res) => {
  try {
    const { eventId } = req.params;
    const userId = req.user.id;
    
    logger.info(`🎪 User ${userId} joining event ${eventId}`);
    
    // Track user engagement
    await communityEngine.trackUserEngagement(userId, 'JOIN_EVENT', {
      event_id: eventId,
      event_type: 'LIVE_MARKET_EVENT'
    });
    
    // Award participation points
    const participationReward = 50;
    
    return response.success(res, 'Successfully joined event', {
      event_id: eventId,
      user_id: userId,
      participation_reward: participationReward,
      status: 'JOINED',
      next_steps: [
        'Participate in live discussions',
        'Make predictions to earn more points',
        'Share insights with the community',
        'Compete for top positions on leaderboard'
      ]
    });
    
  } catch (error) {
    logger.error('❌ Failed to join event:', error);
    return response.error(res, 'Failed to join event', error.message, 500);
  }
});

/**
 * @route GET /api/community/leaderboard
 * @desc Get community leaderboard with top performers
 * @access Public
 */
router.get('/leaderboard', async (req, res) => {
  try {
    const { period = 'daily', limit = 20 } = req.query;
    
    logger.info(`🏆 Fetching ${period} leaderboard...`);
    
    // Mock leaderboard data (in production, this would come from database)
    const leaderboard = [
      {
        rank: 1,
        user_id: 'fm_001',
        user_name: 'Prashant Jain',
        user_role: 'FUND_MANAGER',
        user_level: 25,
        points: 2450,
        change: '+15%',
        badges: ['Market Prophet', 'Alpha Hunter', 'Community Leader'],
        achievements: 47,
        accuracy_rate: 0.89,
        contributions: 156,
        addiction_score: 0.97
      },
      {
        rank: 2,
        user_id: 'investor_001',
        user_name: 'Rakesh Jhunjhunwala Jr.',
        user_role: 'INVESTOR',
        user_level: 18,
        points: 2340,
        change: '+12%',
        badges: ['Trend Spotter', 'Risk Master'],
        achievements: 34,
        accuracy_rate: 0.84,
        contributions: 123,
        addiction_score: 0.94
      },
      {
        rank: 3,
        user_id: 'analyst_001',
        user_name: 'Motilal Oswal',
        user_role: 'ANALYST',
        user_level: 22,
        points: 2180,
        change: '+8%',
        badges: ['IPO Oracle', 'Technical Master'],
        achievements: 41,
        accuracy_rate: 0.91,
        contributions: 98,
        addiction_score: 0.92
      }
    ];
    
    return response.success(res, 'Leaderboard retrieved successfully', {
      leaderboard,
      period,
      total_count: leaderboard.length,
      metrics: {
        average_points: leaderboard.reduce((sum, u) => sum + u.points, 0) / leaderboard.length,
        average_accuracy: leaderboard.reduce((sum, u) => sum + u.accuracy_rate, 0) / leaderboard.length,
        total_contributions: leaderboard.reduce((sum, u) => sum + u.contributions, 0),
        average_addiction_score: leaderboard.reduce((sum, u) => sum + u.addiction_score, 0) / leaderboard.length
      }
    });
    
  } catch (error) {
    logger.error('❌ Failed to get leaderboard:', error);
    return response.error(res, 'Failed to get leaderboard', error.message, 500);
  }
});

/**
 * @route GET /api/community/alpha/exclusive
 * @desc Get exclusive alpha information for premium users
 * @access Private (Premium)
 */
router.get('/alpha/exclusive', auth, async (req, res) => {
  try {
    const userId = req.user.id;
    const userTier = req.user.tier || 'BASIC';
    
    logger.info(`💎 User ${userId} requesting exclusive alpha (tier: ${userTier})`);
    
    // Check user tier for access
    if (!['PREMIUM', 'VIP', 'ELITE'].includes(userTier)) {
      return response.error(res, 'Access denied', 'Premium subscription required for exclusive alpha', 403);
    }
    
    // Mock exclusive alpha data
    const exclusiveAlpha = [
      {
        id: 'alpha_001',
        category: 'BREAKING_IPO_INTEL',
        title: '🚨 Major IPO Pipeline Leak',
        content: 'Exclusive: 3 unicorns planning Q2 2024 listings. Sources confirm Zerodha, Razorpay, and Byju\'s in advanced stages. Expected combined valuation: $15B+',
        exclusivity: 'VIP',
        confidence: 0.92,
        source_reliability: 0.89,
        timestamp: new Date(Date.now() - 120000),
        expiry: new Date(Date.now() + 3600000), // 1 hour
        addiction_factor: 0.99,
        fomo_level: 'MAXIMUM'
      },
      {
        id: 'alpha_002',
        category: 'FUND_MANAGER_WHISPERS',
        title: '💰 Institutional Flow Intelligence',
        content: 'HDFC AMC planning major rebalancing. 500+ Cr moving from banking to IT sector this week. Confirmed by 3 independent sources.',
        exclusivity: 'PREMIUM',
        confidence: 0.87,
        source_reliability: 0.94,
        timestamp: new Date(Date.now() - 300000),
        expiry: new Date(Date.now() + 7200000), // 2 hours
        addiction_factor: 0.96,
        fomo_level: 'HIGH'
      }
    ];
    
    // Filter based on user tier
    const filteredAlpha = exclusiveAlpha.filter(alpha => {
      if (userTier === 'ELITE') return true;
      if (userTier === 'VIP') return ['VIP', 'PREMIUM'].includes(alpha.exclusivity);
      if (userTier === 'PREMIUM') return alpha.exclusivity === 'PREMIUM';
      return false;
    });
    
    // Track alpha access
    await communityEngine.trackUserEngagement(userId, 'ACCESS_EXCLUSIVE_ALPHA', {
      alpha_count: filteredAlpha.length,
      user_tier: userTier
    });
    
    return response.success(res, 'Exclusive alpha retrieved successfully', {
      alpha: filteredAlpha,
      user_tier: userTier,
      access_level: 'GRANTED',
      total_count: filteredAlpha.length,
      next_drop: new Date(Date.now() + 900000), // 15 minutes
      addiction_metrics: {
        average_fomo_level: filteredAlpha.reduce((sum, a) => sum + (a.fomo_level === 'MAXIMUM' ? 1 : a.fomo_level === 'HIGH' ? 0.7 : 0.4), 0) / filteredAlpha.length,
        average_addiction_factor: filteredAlpha.reduce((sum, a) => sum + a.addiction_factor, 0) / filteredAlpha.length
      }
    });
    
  } catch (error) {
    logger.error('❌ Failed to get exclusive alpha:', error);
    return response.error(res, 'Failed to get exclusive alpha', error.message, 500);
  }
});

/**
 * @route GET /api/community/user/:userId/engagement
 * @desc Get user engagement metrics and addiction tracking
 * @access Private
 */
router.get('/user/:userId/engagement', auth, async (req, res) => {
  try {
    const { userId } = req.params;
    const requestingUserId = req.user.id;
    
    // Users can only view their own engagement metrics
    if (userId !== requestingUserId && req.user.role !== 'ADMIN') {
      return response.error(res, 'Access denied', 'Can only view own engagement metrics', 403);
    }
    
    logger.info(`📊 Fetching engagement metrics for user ${userId}`);
    
    // Mock engagement data
    const engagementMetrics = {
      user_id: userId,
      session_stats: {
        current_session_duration: 14400, // 4 hours
        daily_average_duration: 12600, // 3.5 hours
        weekly_sessions: 35,
        monthly_sessions: 142
      },
      addiction_metrics: {
        addiction_level: 0.94,
        office_always_open_streak: 23, // days
        return_frequency: 'Every 12 minutes',
        engagement_score: 0.91,
        compulsive_behavior_indicators: [
          'Frequent page refreshes',
          'Extended session durations',
          'High notification interaction rate',
          'Rapid response to exclusive content'
        ]
      },
      participation_stats: {
        total_messages: 456,
        discussions_started: 23,
        events_joined: 67,
        alpha_accessed: 89,
        points_earned: 12450,
        badges_unlocked: 7,
        achievements: 34
      },
      social_metrics: {
        followers: 234,
        following: 156,
        likes_received: 2340,
        replies_received: 567,
        mentions: 123,
        influence_score: 0.78
      },
      behavioral_patterns: {
        most_active_hours: ['09:00-11:00', '14:00-16:00', '20:00-22:00'],
        favorite_discussion_types: ['BREAKING_NEWS', 'FUND_MANAGER_INSIGHTS'],
        engagement_triggers: ['Exclusive content', 'Breaking news', 'Social validation'],
        addiction_triggers_activated: 47
      }
    };
    
    return response.success(res, 'User engagement metrics retrieved successfully', engagementMetrics);
    
  } catch (error) {
    logger.error('❌ Failed to get user engagement metrics:', error);
    return response.error(res, 'Failed to get user engagement metrics', error.message, 500);
  }
});

/**
 * @route POST /api/community/feedback
 * @desc Submit community feedback and suggestions
 * @access Private
 */
router.post('/feedback',
  auth,
  [
    body('type').isIn(['BUG', 'FEATURE', 'IMPROVEMENT', 'GENERAL']).withMessage('Invalid feedback type'),
    body('content').notEmpty().withMessage('Feedback content is required'),
    body('rating').optional().isInt({ min: 1, max: 5 }).withMessage('Rating must be between 1 and 5')
  ],
  async (req, res) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return response.error(res, 'Validation failed', errors.array(), 400);
      }
      
      const { type, content, rating, category } = req.body;
      const userId = req.user.id;
      
      logger.info(`📝 User ${userId} submitting ${type} feedback`);
      
      const feedback = {
        id: `feedback_${Date.now()}`,
        user_id: userId,
        type,
        content,
        rating: rating || null,
        category: category || 'GENERAL',
        timestamp: new Date(),
        status: 'SUBMITTED',
        priority: type === 'BUG' ? 'HIGH' : 'MEDIUM'
      };
      
      // Award points for feedback
      const feedbackReward = 25;
      
      return response.success(res, 'Feedback submitted successfully', {
        feedback_id: feedback.id,
        points_awarded: feedbackReward,
        status: 'SUBMITTED',
        estimated_response_time: '24-48 hours'
      });
      
    } catch (error) {
      logger.error('❌ Failed to submit feedback:', error);
      return response.error(res, 'Failed to submit feedback', error.message, 500);
    }
  }
);

module.exports = router;
