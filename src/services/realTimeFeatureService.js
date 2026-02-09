const Redis = require('ioredis');
const logger = require('../utils/logger');
const kafkaConsumer = require('./kafkaConsumerService');

class RealTimeFeatureService {
  constructor() {
    this.redis = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: process.env.REDIS_PORT || 6379,
      db: 2 // Separate DB for features
    });

    this.featureCache = new Map();
    this.computationQueue = [];
    this.isProcessing = false;
  }

  async initialize() {
    try {
      // Start consuming user events for real-time feature updates
      await kafkaConsumer.createConsumer(
        'feature-computation-group',
        ['user-events', 'portfolio-changes', 'market-data'],
        this.handleEvent.bind(this)
      );

      logger.info('Real-time feature service initialized');
    } catch (error) {
      logger.error('Failed to initialize real-time feature service', { 
        error: error.message 
      });
      throw error;
    }
  }

  async handleEvent(message) {
    const { topic, value } = message;

    try {
      switch (topic) {
        case 'user-events':
          await this.updateUserFeatures(value);
          break;
        case 'portfolio-changes':
          await this.updatePortfolioFeatures(value);
          break;
        case 'market-data':
          await this.updateMarketFeatures(value);
          break;
      }
    } catch (error) {
      logger.error('Failed to handle event for feature computation', {
        topic,
        error: error.message
      });
    }
  }

  async updateUserFeatures(event) {
    const { userId, eventType, eventData } = event;
    const startTime = Date.now();

    try {
      // Compute features based on event type
      const features = await this.computeUserFeatures(userId, eventType, eventData);

      // Store in Redis with TTL
      const key = `features:user:${userId}`;
      await this.redis.setex(
        key,
        3600, // 1 hour TTL
        JSON.stringify({
          ...features,
          computed_at: new Date().toISOString(),
          latency_ms: Date.now() - startTime
        })
      );

      // Update cache
      this.featureCache.set(userId, features);

      logger.debug('User features updated', { 
        userId, 
        latency: Date.now() - startTime 
      });
    } catch (error) {
      logger.error('Failed to update user features', { 
        userId, 
        error: error.message 
      });
    }
  }

  async computeUserFeatures(userId, eventType, eventData) {
    // Fetch recent activity from Redis
    const recentActions = await this.getRecentActions(userId, 100);

    // Compute features
    const features = {
      // Action-based features
      last_action: eventType,
      last_action_timestamp: Date.now(),
      action_count_1h: this.countActions(recentActions, 3600000),
      action_count_24h: this.countActions(recentActions, 86400000),
      unique_actions_1h: this.countUniqueActions(recentActions, 3600000),

      // Session features
      session_duration: this.calculateSessionDuration(recentActions),
      actions_per_session: this.calculateActionsPerSession(recentActions),

      // Engagement features
      engagement_score: this.calculateEngagementScore(recentActions),
      recency_score: this.calculateRecencyScore(recentActions),

      // Behavioral patterns
      is_active_user: recentActions.length > 10,
      preferred_action: this.getMostFrequentAction(recentActions),
      action_diversity: this.calculateActionDiversity(recentActions)
    };

    return features;
  }

  async updatePortfolioFeatures(event) {
    const { userId, changeType, changeData } = event;
    const startTime = Date.now();

    try {
      const features = await this.computePortfolioFeatures(userId, changeData);

      const key = `features:portfolio:${userId}`;
      await this.redis.setex(
        key,
        1800, // 30 min TTL
        JSON.stringify({
          ...features,
          computed_at: new Date().toISOString(),
          latency_ms: Date.now() - startTime
        })
      );

      logger.debug('Portfolio features updated', { 
        userId, 
        latency: Date.now() - startTime 
      });
    } catch (error) {
      logger.error('Failed to update portfolio features', { 
        userId, 
        error: error.message 
      });
    }
  }

  async computePortfolioFeatures(userId, portfolioData) {
    return {
      total_value: portfolioData.totalValue || 0,
      total_invested: portfolioData.totalInvested || 0,
      total_returns: portfolioData.totalReturns || 0,
      returns_percentage: portfolioData.returnsPercentage || 0,
      num_holdings: portfolioData.holdings?.length || 0,
      
      // Concentration metrics
      concentration_hhi: this.calculateHHI(portfolioData.holdings),
      max_holding_weight: this.getMaxHoldingWeight(portfolioData.holdings),
      
      // Allocation features
      equity_allocation: this.calculateAllocation(portfolioData.holdings, 'EQUITY'),
      debt_allocation: this.calculateAllocation(portfolioData.holdings, 'DEBT'),
      gold_allocation: this.calculateAllocation(portfolioData.holdings, 'GOLD'),
      
      // Risk features
      portfolio_volatility: portfolioData.volatility || 0,
      portfolio_beta: portfolioData.beta || 1.0,
      
      // Time-based features
      days_since_last_transaction: this.getDaysSinceLastTransaction(userId),
      avg_transaction_frequency: this.getAvgTransactionFrequency(userId)
    };
  }

  async updateMarketFeatures(event) {
    const { data } = event;
    const startTime = Date.now();

    try {
      const features = {
        nifty_50_value: data.nifty50 || 0,
        nifty_50_change: data.nifty50Change || 0,
        sensex_value: data.sensex || 0,
        sensex_change: data.sensexChange || 0,
        vix_value: data.vix || 0,
        market_sentiment: this.calculateMarketSentiment(data),
        market_regime: this.detectMarketRegime(data),
        volatility_index: data.vix || 0
      };

      const key = 'features:market:global';
      await this.redis.setex(
        key,
        300, // 5 min TTL
        JSON.stringify({
          ...features,
          computed_at: new Date().toISOString(),
          latency_ms: Date.now() - startTime
        })
      );

      logger.debug('Market features updated', { 
        latency: Date.now() - startTime 
      });
    } catch (error) {
      logger.error('Failed to update market features', { 
        error: error.message 
      });
    }
  }

  // Feature retrieval methods

  async getUserFeatures(userId) {
    const key = `features:user:${userId}`;
    const cached = await this.redis.get(key);

    if (cached) {
      return JSON.parse(cached);
    }

    // Compute if not cached
    const features = await this.computeUserFeatures(userId, null, {});
    await this.redis.setex(key, 3600, JSON.stringify(features));
    return features;
  }

  async getPortfolioFeatures(userId) {
    const key = `features:portfolio:${userId}`;
    const cached = await this.redis.get(key);

    if (cached) {
      return JSON.parse(cached);
    }

    return null; // Return null if not cached, will be computed on next event
  }

  async getMarketFeatures() {
    const key = 'features:market:global';
    const cached = await this.redis.get(key);

    if (cached) {
      return JSON.parse(cached);
    }

    return null;
  }

  async getAllFeatures(userId) {
    const [userFeatures, portfolioFeatures, marketFeatures] = await Promise.all([
      this.getUserFeatures(userId),
      this.getPortfolioFeatures(userId),
      this.getMarketFeatures()
    ]);

    return {
      user: userFeatures,
      portfolio: portfolioFeatures,
      market: marketFeatures,
      retrieved_at: new Date().toISOString()
    };
  }

  // Helper methods

  async getRecentActions(userId, limit = 100) {
    const key = `actions:${userId}`;
    const actions = await this.redis.lrange(key, 0, limit - 1);
    return actions.map(a => JSON.parse(a));
  }

  countActions(actions, timeWindow) {
    const cutoff = Date.now() - timeWindow;
    return actions.filter(a => a.timestamp > cutoff).length;
  }

  countUniqueActions(actions, timeWindow) {
    const cutoff = Date.now() - timeWindow;
    const recentActions = actions.filter(a => a.timestamp > cutoff);
    return new Set(recentActions.map(a => a.type)).size;
  }

  calculateSessionDuration(actions) {
    if (actions.length < 2) return 0;
    const first = actions[actions.length - 1].timestamp;
    const last = actions[0].timestamp;
    return last - first;
  }

  calculateActionsPerSession(actions) {
    // Simplified: assume session gap of 30 minutes
    const sessionGap = 1800000; // 30 min
    let sessions = 1;
    
    for (let i = 1; i < actions.length; i++) {
      if (actions[i - 1].timestamp - actions[i].timestamp > sessionGap) {
        sessions++;
      }
    }
    
    return actions.length / sessions;
  }

  calculateEngagementScore(actions) {
    // Weighted score based on action types and recency
    let score = 0;
    const weights = {
      'BUY': 10,
      'SELL': 8,
      'VIEW_INSIGHTS': 5,
      'VIEW_PORTFOLIO': 3,
      'LOGIN': 1
    };

    actions.forEach((action, index) => {
      const weight = weights[action.type] || 1;
      const recencyFactor = 1 / (index + 1); // More recent = higher weight
      score += weight * recencyFactor;
    });

    return Math.min(score, 100); // Cap at 100
  }

  calculateRecencyScore(actions) {
    if (actions.length === 0) return 0;
    const lastActionTime = actions[0].timestamp;
    const hoursSinceLastAction = (Date.now() - lastActionTime) / 3600000;
    return Math.max(0, 100 - hoursSinceLastAction * 10);
  }

  getMostFrequentAction(actions) {
    const counts = {};
    actions.forEach(a => {
      counts[a.type] = (counts[a.type] || 0) + 1;
    });
    
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])[0]?.[0] || 'NONE';
  }

  calculateActionDiversity(actions) {
    if (actions.length === 0) return 0;
    const uniqueActions = new Set(actions.map(a => a.type)).size;
    return uniqueActions / Math.min(actions.length, 10); // Normalize
  }

  calculateHHI(holdings) {
    if (!holdings || holdings.length === 0) return 0;
    const totalValue = holdings.reduce((sum, h) => sum + h.value, 0);
    return holdings.reduce((hhi, h) => {
      const weight = h.value / totalValue;
      return hhi + (weight * weight);
    }, 0);
  }

  getMaxHoldingWeight(holdings) {
    if (!holdings || holdings.length === 0) return 0;
    const totalValue = holdings.reduce((sum, h) => sum + h.value, 0);
    const maxValue = Math.max(...holdings.map(h => h.value));
    return maxValue / totalValue;
  }

  calculateAllocation(holdings, category) {
    if (!holdings || holdings.length === 0) return 0;
    const totalValue = holdings.reduce((sum, h) => sum + h.value, 0);
    const categoryValue = holdings
      .filter(h => h.category === category)
      .reduce((sum, h) => sum + h.value, 0);
    return categoryValue / totalValue;
  }

  getDaysSinceLastTransaction(userId) {
    // Placeholder - would fetch from database
    return 0;
  }

  getAvgTransactionFrequency(userId) {
    // Placeholder - would calculate from transaction history
    return 0;
  }

  calculateMarketSentiment(data) {
    // Simplified sentiment calculation
    const niftyChange = data.nifty50Change || 0;
    const sensexChange = data.sensexChange || 0;
    const avgChange = (niftyChange + sensexChange) / 2;
    
    if (avgChange > 1) return 'BULLISH';
    if (avgChange < -1) return 'BEARISH';
    return 'NEUTRAL';
  }

  detectMarketRegime(data) {
    const vix = data.vix || 0;
    const change = data.nifty50Change || 0;
    
    if (vix > 25) return 'VOLATILE';
    if (change > 1) return 'BULL';
    if (change < -1) return 'BEAR';
    return 'SIDEWAYS';
  }

  async getMetrics() {
    const cacheSize = this.featureCache.size;
    const queueSize = this.computationQueue.length;
    
    return {
      cache_size: cacheSize,
      queue_size: queueSize,
      is_processing: this.isProcessing
    };
  }
}

module.exports = new RealTimeFeatureService();
