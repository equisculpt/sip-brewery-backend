const WebSocket = require('ws');
const EventEmitter = require('events');
const logger = require('../utils/logger');

/**
 * REAL-TIME COMMUNITY WEBSOCKET ENGINE
 * Powers the Most Addictive Financial Community with Real-Time Features
 * 
 * This WebSocket engine creates real-time interactions that are so engaging
 * that users will keep the community page open in their office 24/7.
 * 
 * Features that create maximum addiction:
 * - Instant message delivery and notifications
 * - Real-time market event broadcasts
 * - Live exclusive alpha drops with FOMO triggers
 * - Instant social validation and reactions
 * - Real-time leaderboard updates
 * - Live fund manager interactions
 * - Breaking news instant alerts
 * - Gamification real-time rewards
 * - Social trading live updates
 * - Addiction psychology triggers
 */
class RealTimeCommunityWebSocket extends EventEmitter {
  constructor(server) {
    super();
    this.server = server;
    this.wss = null;
    this.clients = new Map(); // userId -> WebSocket connection
    this.rooms = new Map(); // roomId -> Set of userIds
    this.userSessions = new Map(); // userId -> session data
    this.addictionTriggers = new Map(); // userId -> addiction trigger data
    
    // Real-time engagement configuration
    this.engagementConfig = {
      message_broadcast_delay: 0, // Instant delivery
      notification_priority: {
        BREAKING_NEWS: 'CRITICAL',
        EXCLUSIVE_ALPHA: 'HIGH',
        FUND_MANAGER_INSIGHTS: 'HIGH',
        SOCIAL_VALIDATION: 'MEDIUM',
        GAMIFICATION_REWARDS: 'MEDIUM'
      },
      addiction_triggers: {
        variable_rewards: true,
        fomo_notifications: true,
        social_pressure: true,
        exclusive_access: true,
        real_time_feedback: true
      },
      engagement_goals: {
        instant_response_rate: '95%+',
        notification_click_rate: '80%+',
        real_time_participation: '90%+',
        addiction_maintenance: 'MAXIMUM'
      }
    };

    this.initializeWebSocketServer();
  }

  initializeWebSocketServer() {
    try {
      logger.info('🌐 Initializing Real-Time Community WebSocket Server...');
      
      this.wss = new WebSocket.Server({ 
        server: this.server,
        path: '/community-ws',
        clientTracking: true
      });

      this.wss.on('connection', (ws, req) => {
        this.handleNewConnection(ws, req);
      });

      this.wss.on('error', (error) => {
        logger.error('❌ WebSocket Server Error:', error);
      });

      // Start real-time engagement loops
      this.startRealTimeEngagementLoops();
      
      logger.info('✅ Real-Time Community WebSocket Server initialized successfully');
      logger.info('🎯 Goal: Keep users engaged with instant real-time interactions');
      logger.info('🧠 Addiction triggers: Variable rewards, FOMO, social validation');
      
    } catch (error) {
      logger.error('❌ Failed to initialize WebSocket server:', error);
      throw error;
    }
  }

  handleNewConnection(ws, req) {
    try {
      const userId = this.extractUserIdFromRequest(req);
      const userRole = this.extractUserRoleFromRequest(req);
      
      logger.info(`🔗 New WebSocket connection: User ${userId} (${userRole})`);
      
      // Store client connection
      this.clients.set(userId, ws);
      
      // Initialize user session
      this.userSessions.set(userId, {
        userId,
        userRole,
        connectedAt: new Date(),
        lastActivity: new Date(),
        messageCount: 0,
        engagementScore: 0,
        addictionLevel: 0,
        rooms: new Set(),
        notifications: [],
        achievements: []
      });

      // Set up connection handlers
      ws.on('message', (data) => {
        this.handleMessage(userId, data);
      });

      ws.on('close', () => {
        this.handleDisconnection(userId);
      });

      ws.on('error', (error) => {
        logger.error(`❌ WebSocket error for user ${userId}:`, error);
      });

      // Send welcome message with addiction hooks
      this.sendWelcomeMessage(userId);
      
      // Start addiction psychology for this user
      this.initializeAddictionPsychology(userId);
      
    } catch (error) {
      logger.error('❌ Failed to handle new connection:', error);
    }
  }

  handleMessage(userId, data) {
    try {
      const message = JSON.parse(data.toString());
      const session = this.userSessions.get(userId);
      
      if (!session) return;
      
      // Update user activity
      session.lastActivity = new Date();
      session.messageCount++;
      
      logger.info(`📨 Message from user ${userId}:`, message.type);
      
      switch (message.type) {
        case 'JOIN_DISCUSSION':
          this.handleJoinDiscussion(userId, message.data);
          break;
          
        case 'SEND_MESSAGE':
          this.handleSendMessage(userId, message.data);
          break;
          
        case 'JOIN_EVENT':
          this.handleJoinEvent(userId, message.data);
          break;
          
        case 'REQUEST_ALPHA':
          this.handleAlphaRequest(userId, message.data);
          break;
          
        case 'LIKE_MESSAGE':
          this.handleLikeMessage(userId, message.data);
          break;
          
        case 'HEARTBEAT':
          this.handleHeartbeat(userId);
          break;
          
        default:
          logger.warn(`Unknown message type: ${message.type}`);
      }
      
      // Trigger addiction psychology
      this.triggerAddictionPsychology(userId, message.type);
      
    } catch (error) {
      logger.error(`❌ Failed to handle message from user ${userId}:`, error);
    }
  }

  handleJoinDiscussion(userId, data) {
    const { discussionId } = data;
    const roomId = `discussion_${discussionId}`;
    
    // Add user to discussion room
    if (!this.rooms.has(roomId)) {
      this.rooms.set(roomId, new Set());
    }
    this.rooms.get(roomId).add(userId);
    
    // Update user session
    const session = this.userSessions.get(userId);
    if (session) {
      session.rooms.add(roomId);
    }
    
    // Broadcast user joined
    this.broadcastToRoom(roomId, {
      type: 'USER_JOINED_DISCUSSION',
      data: {
        userId,
        discussionId,
        timestamp: new Date()
      }
    }, userId);
    
    // Send discussion history to user
    this.sendDiscussionHistory(userId, discussionId);
    
    logger.info(`👥 User ${userId} joined discussion ${discussionId}`);
  }

  handleSendMessage(userId, data) {
    const { discussionId, content, alpha, exclusive } = data;
    const roomId = `discussion_${discussionId}`;
    
    const messageData = {
      id: `msg_${Date.now()}_${userId}`,
      userId,
      discussionId,
      content,
      alpha: alpha || false,
      exclusive: exclusive || false,
      timestamp: new Date(),
      likes: 0,
      replies: 0
    };
    
    // Broadcast message to discussion room instantly
    this.broadcastToRoom(roomId, {
      type: 'NEW_MESSAGE',
      data: messageData
    });
    
    // Award points for participation
    const points = alpha ? 100 : exclusive ? 75 : 25;
    this.awardPoints(userId, points, 'MESSAGE_SENT');
    
    // Trigger social validation
    this.triggerSocialValidation(userId, messageData);
    
    logger.info(`💬 User ${userId} sent message to discussion ${discussionId}`);
  }

  handleJoinEvent(userId, data) {
    const { eventId } = data;
    const roomId = `event_${eventId}`;
    
    // Add user to event room
    if (!this.rooms.has(roomId)) {
      this.rooms.set(roomId, new Set());
    }
    this.rooms.get(roomId).add(userId);
    
    // Award participation points
    this.awardPoints(userId, 50, 'EVENT_JOINED');
    
    // Send event welcome
    this.sendToUser(userId, {
      type: 'EVENT_JOINED',
      data: {
        eventId,
        welcome_message: 'Welcome to the live event! Participate actively to earn more points!',
        participation_reward: 50
      }
    });
    
    logger.info(`🎪 User ${userId} joined event ${eventId}`);
  }

  handleAlphaRequest(userId, data) {
    const session = this.userSessions.get(userId);
    if (!session) return;
    
    // Check user tier for alpha access
    const userTier = session.userTier || 'BASIC';
    
    if (['PREMIUM', 'VIP', 'ELITE'].includes(userTier)) {
      // Send exclusive alpha
      this.sendExclusiveAlpha(userId, userTier);
      
      // Track alpha access for addiction
      this.trackAddictionTrigger(userId, 'EXCLUSIVE_ACCESS');
    } else {
      // Send upgrade prompt (FOMO trigger)
      this.sendUpgradePrompt(userId);
    }
  }

  handleLikeMessage(userId, data) {
    const { messageId, discussionId } = data;
    
    // Broadcast like update
    const roomId = `discussion_${discussionId}`;
    this.broadcastToRoom(roomId, {
      type: 'MESSAGE_LIKED',
      data: {
        messageId,
        likedBy: userId,
        timestamp: new Date()
      }
    });
    
    // Award points for engagement
    this.awardPoints(userId, 5, 'MESSAGE_LIKED');
    
    logger.info(`❤️ User ${userId} liked message ${messageId}`);
  }

  handleHeartbeat(userId) {
    const session = this.userSessions.get(userId);
    if (session) {
      session.lastActivity = new Date();
      
      // Update addiction level based on session duration
      const sessionDuration = Date.now() - session.connectedAt.getTime();
      session.addictionLevel = Math.min(sessionDuration / (4 * 60 * 60 * 1000), 1); // Max after 4 hours
      
      // Send heartbeat response
      this.sendToUser(userId, {
        type: 'HEARTBEAT_ACK',
        data: {
          timestamp: new Date(),
          session_duration: sessionDuration,
          addiction_level: session.addictionLevel
        }
      });
    }
  }

  handleDisconnection(userId) {
    logger.info(`🔌 User ${userId} disconnected`);
    
    // Remove from all rooms
    const session = this.userSessions.get(userId);
    if (session) {
      session.rooms.forEach(roomId => {
        const room = this.rooms.get(roomId);
        if (room) {
          room.delete(userId);
          
          // Broadcast user left
          this.broadcastToRoom(roomId, {
            type: 'USER_LEFT',
            data: {
              userId,
              timestamp: new Date()
            }
          }, userId);
        }
      });
    }
    
    // Clean up
    this.clients.delete(userId);
    this.userSessions.delete(userId);
    this.addictionTriggers.delete(userId);
  }

  // Real-time engagement and addiction methods
  startRealTimeEngagementLoops() {
    // Breaking news alerts every 2-5 minutes
    setInterval(() => {
      this.broadcastBreakingNews();
    }, Math.random() * 180000 + 120000); // 2-5 minutes
    
    // Exclusive alpha drops every 10-15 minutes
    setInterval(() => {
      this.broadcastExclusiveAlpha();
    }, Math.random() * 300000 + 600000); // 10-15 minutes
    
    // Gamification rewards every 5-10 minutes
    setInterval(() => {
      this.distributeRandomRewards();
    }, Math.random() * 300000 + 300000); // 5-10 minutes
    
    // FOMO triggers every 3-7 minutes
    setInterval(() => {
      this.triggerFOMONotifications();
    }, Math.random() * 240000 + 180000); // 3-7 minutes
    
    // Social validation every 1-3 minutes
    setInterval(() => {
      this.broadcastSocialValidation();
    }, Math.random() * 120000 + 60000); // 1-3 minutes
  }

  broadcastBreakingNews() {
    const breakingNews = [
      {
        title: '🚨 BREAKING: Sensex crosses 75,000 for first time!',
        content: 'Historic milestone reached as markets surge on positive economic data',
        urgency: 'CRITICAL',
        addiction_factor: 0.98
      },
      {
        title: '💥 ALERT: Major IPO announcement expected today',
        content: 'Sources confirm unicorn startup planning IPO announcement within hours',
        urgency: 'HIGH',
        addiction_factor: 0.95
      },
      {
        title: '⚡ FLASH: RBI policy decision leaked?',
        content: 'Unconfirmed reports suggest rate cut decision ahead of official announcement',
        urgency: 'HIGH',
        addiction_factor: 0.93
      }
    ];
    
    const news = breakingNews[Math.floor(Math.random() * breakingNews.length)];
    
    this.broadcastToAll({
      type: 'BREAKING_NEWS',
      data: {
        ...news,
        timestamp: new Date(),
        id: `news_${Date.now()}`
      }
    });
    
    logger.info('📢 Broadcasted breaking news to all users');
  }

  broadcastExclusiveAlpha() {
    const alphaContent = [
      {
        title: '💎 EXCLUSIVE: Major fund rebalancing detected',
        content: 'Large cap fund moving 200+ Cr from banking to IT sector',
        exclusivity: 'PREMIUM',
        confidence: 0.89
      },
      {
        title: '🔥 ALPHA: Insider buying activity surge',
        content: 'Promoter buying detected in 5 mid-cap stocks this week',
        exclusivity: 'VIP',
        confidence: 0.92
      }
    ];
    
    const alpha = alphaContent[Math.floor(Math.random() * alphaContent.length)];
    
    // Send to premium users only
    this.clients.forEach((ws, userId) => {
      const session = this.userSessions.get(userId);
      if (session && ['PREMIUM', 'VIP', 'ELITE'].includes(session.userTier)) {
        this.sendToUser(userId, {
          type: 'EXCLUSIVE_ALPHA',
          data: {
            ...alpha,
            timestamp: new Date(),
            id: `alpha_${Date.now()}`
          }
        });
      }
    });
    
    logger.info('💎 Broadcasted exclusive alpha to premium users');
  }

  distributeRandomRewards() {
    const rewards = [
      { type: 'POINTS', amount: 50, reason: 'Active participation bonus' },
      { type: 'BADGE', name: 'Night Owl', reason: 'Late night engagement' },
      { type: 'EXCLUSIVE_ACCESS', feature: 'VIP Alpha Room', duration: '1 hour' }
    ];
    
    // Give rewards to random active users
    const activeUsers = Array.from(this.userSessions.keys()).filter(userId => {
      const session = this.userSessions.get(userId);
      return session && (Date.now() - session.lastActivity.getTime()) < 300000; // Active in last 5 minutes
    });
    
    const luckyUsers = activeUsers.slice(0, Math.floor(Math.random() * 5) + 1);
    
    luckyUsers.forEach(userId => {
      const reward = rewards[Math.floor(Math.random() * rewards.length)];
      this.sendToUser(userId, {
        type: 'RANDOM_REWARD',
        data: {
          ...reward,
          timestamp: new Date(),
          id: `reward_${Date.now()}_${userId}`
        }
      });
    });
    
    if (luckyUsers.length > 0) {
      logger.info(`🎁 Distributed random rewards to ${luckyUsers.length} users`);
    }
  }

  triggerFOMONotifications() {
    const fomoTriggers = [
      'Only 2 spots left in exclusive fund manager AMA!',
      'Limited time: Premium alpha access 50% off!',
      'Last chance to join IPO prediction tournament!',
      'VIP room filling up fast - 5 minutes left to join!'
    ];
    
    const trigger = fomoTriggers[Math.floor(Math.random() * fomoTriggers.length)];
    
    // Send to random subset of users
    const allUsers = Array.from(this.clients.keys());
    const targetUsers = allUsers.slice(0, Math.floor(allUsers.length * 0.3)); // 30% of users
    
    targetUsers.forEach(userId => {
      this.sendToUser(userId, {
        type: 'FOMO_NOTIFICATION',
        data: {
          message: trigger,
          urgency: 'HIGH',
          timestamp: new Date(),
          expires_in: 300000 // 5 minutes
        }
      });
    });
    
    logger.info(`🚨 Triggered FOMO notifications for ${targetUsers.length} users`);
  }

  broadcastSocialValidation() {
    const validationMessages = [
      'Prashant Jain just liked your analysis!',
      'Your prediction is trending in the community!',
      '5 fund managers are following your insights!',
      'You\'re in the top 10 contributors today!'
    ];
    
    // Send validation to active users
    this.clients.forEach((ws, userId) => {
      const session = this.userSessions.get(userId);
      if (session && session.messageCount > 0) {
        const message = validationMessages[Math.floor(Math.random() * validationMessages.length)];
        this.sendToUser(userId, {
          type: 'SOCIAL_VALIDATION',
          data: {
            message,
            timestamp: new Date(),
            validation_type: 'PEER_RECOGNITION'
          }
        });
      }
    });
  }

  // Utility methods
  sendToUser(userId, message) {
    const ws = this.clients.get(userId);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }

  broadcastToRoom(roomId, message, excludeUserId = null) {
    const room = this.rooms.get(roomId);
    if (room) {
      room.forEach(userId => {
        if (userId !== excludeUserId) {
          this.sendToUser(userId, message);
        }
      });
    }
  }

  broadcastToAll(message, excludeUserId = null) {
    this.clients.forEach((ws, userId) => {
      if (userId !== excludeUserId) {
        this.sendToUser(userId, message);
      }
    });
  }

  awardPoints(userId, points, reason) {
    const session = this.userSessions.get(userId);
    if (session) {
      session.engagementScore += points;
      
      this.sendToUser(userId, {
        type: 'POINTS_AWARDED',
        data: {
          points,
          reason,
          total_points: session.engagementScore,
          timestamp: new Date()
        }
      });
    }
  }

  triggerAddictionPsychology(userId, action) {
    // Variable reward schedule
    if (Math.random() > 0.7) {
      this.giveVariableReward(userId, action);
    }
    
    // Track addiction triggers
    this.trackAddictionTrigger(userId, action);
    
    // Update addiction level
    this.updateAddictionLevel(userId);
  }

  trackAddictionTrigger(userId, trigger) {
    if (!this.addictionTriggers.has(userId)) {
      this.addictionTriggers.set(userId, {
        triggers: [],
        count: 0,
        last_trigger: null
      });
    }
    
    const userTriggers = this.addictionTriggers.get(userId);
    userTriggers.triggers.push({
      trigger,
      timestamp: new Date()
    });
    userTriggers.count++;
    userTriggers.last_trigger = new Date();
  }

  extractUserIdFromRequest(req) {
    // Extract user ID from query params or headers
    return req.url.split('userId=')[1]?.split('&')[0] || `user_${Date.now()}`;
  }

  extractUserRoleFromRequest(req) {
    // Extract user role from query params or headers
    return req.url.split('role=')[1]?.split('&')[0] || 'INVESTOR';
  }

  sendWelcomeMessage(userId) {
    this.sendToUser(userId, {
      type: 'WELCOME',
      data: {
        message: 'Welcome to the most addictive financial community! 🚀',
        features: [
          'Real-time market discussions',
          'Exclusive alpha from fund managers',
          'Live events and competitions',
          'Instant notifications and rewards'
        ],
        addiction_warning: 'Warning: This platform is designed to be highly engaging!',
        timestamp: new Date()
      }
    });
  }

  getConnectionStats() {
    return {
      total_connections: this.clients.size,
      active_rooms: this.rooms.size,
      total_sessions: this.userSessions.size,
      addiction_triggers_active: this.addictionTriggers.size
    };
  }
}

module.exports = RealTimeCommunityWebSocket;
