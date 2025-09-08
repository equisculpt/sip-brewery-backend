/**
 * Enterprise Caching Service
 * High-performance Redis-based caching for financial data
 * @module services/cacheService
 */

const { getRedisClient } = require('../config/redis');
const logger = require('../utils/logger');

/**
 * Cache TTL (Time To Live) configurations in seconds
 */
const CACHE_TTL = {
  // Financial Data Caching
  MARKET_DATA: 60, // 1 minute for real-time market data
  FUND_PRICES: 300, // 5 minutes for fund prices
  PORTFOLIO_DATA: 600, // 10 minutes for portfolio data
  USER_PROFILE: 1800, // 30 minutes for user profiles
  REWARDS_DATA: 900, // 15 minutes for rewards data
  
  // Static Data Caching
  FUND_METADATA: 3600, // 1 hour for fund metadata
  MARKET_HOLIDAYS: 86400, // 24 hours for market holidays
  EXCHANGE_RATES: 1800, // 30 minutes for exchange rates
  
  // Analytics Caching
  DASHBOARD_ANALYTICS: 300, // 5 minutes for dashboard data
  PERFORMANCE_METRICS: 600, // 10 minutes for performance data
  LEADERBOARD: 900, // 15 minutes for leaderboard
  
  // Long-term Caching
  HISTORICAL_DATA: 7200, // 2 hours for historical data
  REPORTS: 3600, // 1 hour for generated reports
  CONFIGURATIONS: 1800 // 30 minutes for app configurations
};

/**
 * Cache Key Prefixes for Organization
 */
const CACHE_PREFIXES = {
  USER: 'user:',
  FUND: 'fund:',
  MARKET: 'market:',
  PORTFOLIO: 'portfolio:',
  REWARDS: 'rewards:',
  ANALYTICS: 'analytics:',
  REPORT: 'report:',
  CONFIG: 'config:'
};

class CacheService {
  constructor() {
    this.redis = null;
    this.isEnabled = false;
  }

  /**
   * Initialize cache service
   */
  async initialize() {
    try {
      this.redis = getRedisClient();
      this.isEnabled = this.redis !== null;
      
      if (this.isEnabled) {
        logger.info('✅ Cache service initialized with Redis');
      } else {
        logger.warn('⚠️  Cache service running without Redis - performance may be impacted');
      }
    } catch (error) {
      logger.error('❌ Failed to initialize cache service:', error.message);
      this.isEnabled = false;
    }
  }

  /**
   * Generate cache key with prefix
   */
  generateKey(prefix, identifier) {
    return `${prefix}${identifier}`;
  }

  /**
   * Set cache with TTL
   */
  async set(key, value, ttl = CACHE_TTL.FUND_PRICES) {
    if (!this.isEnabled) return false;

    try {
      const serializedValue = JSON.stringify(value);
      await this.redis.setEx(key, ttl, serializedValue);
      
      logger.debug(`Cache SET: ${key} (TTL: ${ttl}s)`);
      return true;
    } catch (error) {
      logger.error(`Cache SET error for key ${key}:`, error.message);
      return false;
    }
  }

  /**
   * Get cached value
   */
  async get(key) {
    if (!this.isEnabled) return null;

    try {
      const cachedValue = await this.redis.get(key);
      
      if (cachedValue) {
        logger.debug(`Cache HIT: ${key}`);
        return JSON.parse(cachedValue);
      }
      
      logger.debug(`Cache MISS: ${key}`);
      return null;
    } catch (error) {
      logger.error(`Cache GET error for key ${key}:`, error.message);
      return null;
    }
  }

  /**
   * Delete cached value
   */
  async del(key) {
    if (!this.isEnabled) return false;

    try {
      const result = await this.redis.del(key);
      logger.debug(`Cache DEL: ${key}`);
      return result > 0;
    } catch (error) {
      logger.error(`Cache DEL error for key ${key}:`, error.message);
      return false;
    }
  }

  /**
   * Delete multiple keys by pattern
   */
  async delPattern(pattern) {
    if (!this.isEnabled) return 0;

    try {
      const keys = await this.redis.keys(pattern);
      if (keys.length === 0) return 0;
      
      const result = await this.redis.del(keys);
      logger.debug(`Cache DEL pattern ${pattern}: ${result} keys deleted`);
      return result;
    } catch (error) {
      logger.error(`Cache DEL pattern error for ${pattern}:`, error.message);
      return 0;
    }
  }

  /**
   * Check if key exists
   */
  async exists(key) {
    if (!this.isEnabled) return false;

    try {
      const result = await this.redis.exists(key);
      return result === 1;
    } catch (error) {
      logger.error(`Cache EXISTS error for key ${key}:`, error.message);
      return false;
    }
  }

  /**
   * Set TTL for existing key
   */
  async expire(key, ttl) {
    if (!this.isEnabled) return false;

    try {
      const result = await this.redis.expire(key, ttl);
      return result === 1;
    } catch (error) {
      logger.error(`Cache EXPIRE error for key ${key}:`, error.message);
      return false;
    }
  }

  /**
   * Get cache statistics
   */
  async getStats() {
    if (!this.isEnabled) {
      return {
        enabled: false,
        message: 'Redis not available'
      };
    }

    try {
      const info = await this.redis.info('memory');
      const keyspace = await this.redis.info('keyspace');
      
      return {
        enabled: true,
        memory: this.parseRedisInfo(info),
        keyspace: this.parseRedisInfo(keyspace),
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      logger.error('Cache stats error:', error.message);
      return {
        enabled: false,
        error: error.message
      };
    }
  }

  /**
   * Parse Redis INFO response
   */
  parseRedisInfo(info) {
    const lines = info.split('\r\n');
    const result = {};
    
    lines.forEach(line => {
      if (line && !line.startsWith('#')) {
        const [key, value] = line.split(':');
        if (key && value) {
          result[key] = value;
        }
      }
    });
    
    return result;
  }

  /**
   * Warm up cache with frequently accessed data
   */
  async warmUp() {
    if (!this.isEnabled) return;

    try {
      logger.info('🔥 Starting cache warm-up...');
      
      // Warm up with common configurations
      const commonKeys = [
        'app:config:general',
        'market:status:current',
        'exchange:rates:latest'
      ];
      
      for (const key of commonKeys) {
        // Check if key exists, if not, it will be populated by the application
        const exists = await this.exists(key);
        if (!exists) {
          logger.debug(`Cache warm-up: ${key} not found, will be populated on first access`);
        }
      }
      
      logger.info('✅ Cache warm-up completed');
    } catch (error) {
      logger.error('❌ Cache warm-up failed:', error.message);
    }
  }

  /**
   * Clear all cache
   */
  async clear() {
    if (!this.isEnabled) return false;

    try {
      await this.redis.flushDb();
      logger.info('🧹 Cache cleared');
      return true;
    } catch (error) {
      logger.error('❌ Cache clear error:', error.message);
      return false;
    }
  }

  // ============================================================================
  // HIGH-LEVEL CACHING METHODS FOR SPECIFIC DATA TYPES
  // ============================================================================

  /**
   * Cache user data
   */
  async cacheUserData(userId, userData) {
    const key = this.generateKey(CACHE_PREFIXES.USER, userId);
    return await this.set(key, userData, CACHE_TTL.USER_PROFILE);
  }

  /**
   * Get cached user data
   */
  async getUserData(userId) {
    const key = this.generateKey(CACHE_PREFIXES.USER, userId);
    return await this.get(key);
  }

  /**
   * Cache fund data
   */
  async cacheFundData(fundId, fundData) {
    const key = this.generateKey(CACHE_PREFIXES.FUND, fundId);
    return await this.set(key, fundData, CACHE_TTL.FUND_PRICES);
  }

  /**
   * Get cached fund data
   */
  async getFundData(fundId) {
    const key = this.generateKey(CACHE_PREFIXES.FUND, fundId);
    return await this.get(key);
  }

  /**
   * Cache market data
   */
  async cacheMarketData(symbol, marketData) {
    const key = this.generateKey(CACHE_PREFIXES.MARKET, symbol);
    return await this.set(key, marketData, CACHE_TTL.MARKET_DATA);
  }

  /**
   * Get cached market data
   */
  async getMarketData(symbol) {
    const key = this.generateKey(CACHE_PREFIXES.MARKET, symbol);
    return await this.get(key);
  }

  /**
   * Cache portfolio data
   */
  async cachePortfolioData(userId, portfolioData) {
    const key = this.generateKey(CACHE_PREFIXES.PORTFOLIO, userId);
    return await this.set(key, portfolioData, CACHE_TTL.PORTFOLIO_DATA);
  }

  /**
   * Get cached portfolio data
   */
  async getPortfolioData(userId) {
    const key = this.generateKey(CACHE_PREFIXES.PORTFOLIO, userId);
    return await this.get(key);
  }

  /**
   * Cache rewards data
   */
  async cacheRewardsData(userId, rewardsData) {
    const key = this.generateKey(CACHE_PREFIXES.REWARDS, userId);
    return await this.set(key, rewardsData, CACHE_TTL.REWARDS_DATA);
  }

  /**
   * Get cached rewards data
   */
  async getRewardsData(userId) {
    const key = this.generateKey(CACHE_PREFIXES.REWARDS, userId);
    return await this.get(key);
  }

  /**
   * Invalidate user-related cache
   */
  async invalidateUserCache(userId) {
    const patterns = [
      `${CACHE_PREFIXES.USER}${userId}*`,
      `${CACHE_PREFIXES.PORTFOLIO}${userId}*`,
      `${CACHE_PREFIXES.REWARDS}${userId}*`
    ];

    let totalDeleted = 0;
    for (const pattern of patterns) {
      totalDeleted += await this.delPattern(pattern);
    }

    logger.info(`Invalidated ${totalDeleted} cache entries for user ${userId}`);
    return totalDeleted;
  }
}

// Export singleton instance
const cacheService = new CacheService();

module.exports = {
  cacheService,
  CACHE_TTL,
  CACHE_PREFIXES
};
