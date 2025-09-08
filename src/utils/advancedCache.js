/**
 * 🚀 ADVANCED CACHING SYSTEM
 * 
 * Multi-layer caching for maximum performance
 */

const Redis = require('redis');
const NodeCache = require('node-cache');
const logger = require('./logger');

class AdvancedCache {
  constructor() {
    // Memory cache (L1) - fastest
    this.memoryCache = new NodeCache({
      stdTTL: 300, // 5 minutes default TTL
      checkperiod: 60, // Check for expired keys every minute
      useClones: false, // Don't clone objects for better performance
      maxKeys: 10000 // Limit memory usage
    });
    
    // Redis cache (L2) - persistent
    this.redisClient = null;
    this.initializeRedis();
    
    // Cache statistics
    this.stats = {
      memoryHits: 0,
      redisHits: 0,
      misses: 0,
      sets: 0
    };
  }

  async initializeRedis() {
    try {
      this.redisClient = Redis.createClient({
        url: process.env.REDIS_URL,
        retry_strategy: (options) => {
          if (options.error && options.error.code === 'ECONNREFUSED') {
            return new Error('Redis server connection refused');
          }
          if (options.total_retry_time > 1000 * 60 * 60) {
            return new Error('Redis retry time exhausted');
          }
          if (options.attempt > 10) {
            return undefined;
          }
          return Math.min(options.attempt * 100, 3000);
        }
      });
      
      await this.redisClient.connect();
      logger.info('✅ Redis cache connected');
    } catch (error) {
      logger.warn('Redis cache not available, using memory cache only:', error.message);
    }
  }

  // Multi-layer cache get
  async get(key) {
    try {
      // L1: Check memory cache first
      const memoryResult = this.memoryCache.get(key);
      if (memoryResult !== undefined) {
        this.stats.memoryHits++;
        return memoryResult;
      }
      
      // L2: Check Redis cache
      if (this.redisClient) {
        const redisResult = await this.redisClient.get(key);
        if (redisResult) {
          const parsed = JSON.parse(redisResult);
          // Store in memory cache for faster future access
          this.memoryCache.set(key, parsed, 300);
          this.stats.redisHits++;
          return parsed;
        }
      }
      
      this.stats.misses++;
      return null;
    } catch (error) {
      logger.error('Cache get error:', error);
      return null;
    }
  }

  // Multi-layer cache set
  async set(key, value, ttl = 300) {
    try {
      // Set in memory cache
      this.memoryCache.set(key, value, ttl);
      
      // Set in Redis cache with longer TTL
      if (this.redisClient) {
        await this.redisClient.setEx(key, ttl * 2, JSON.stringify(value));
      }
      
      this.stats.sets++;
      return true;
    } catch (error) {
      logger.error('Cache set error:', error);
      return false;
    }
  }

  // Cache with automatic refresh
  async getOrSet(key, fetchFunction, ttl = 300) {
    const cached = await this.get(key);
    if (cached !== null) {
      return cached;
    }
    
    try {
      const value = await fetchFunction();
      await this.set(key, value, ttl);
      return value;
    } catch (error) {
      logger.error('Cache getOrSet error:', error);
      throw error;
    }
  }

  // Batch operations for better performance
  async mget(keys) {
    const results = {};
    const missingKeys = [];
    
    // Check memory cache first
    for (const key of keys) {
      const memoryResult = this.memoryCache.get(key);
      if (memoryResult !== undefined) {
        results[key] = memoryResult;
        this.stats.memoryHits++;
      } else {
        missingKeys.push(key);
      }
    }
    
    // Check Redis for missing keys
    if (missingKeys.length > 0 && this.redisClient) {
      try {
        const redisResults = await this.redisClient.mGet(missingKeys);
        missingKeys.forEach((key, index) => {
          if (redisResults[index]) {
            const parsed = JSON.parse(redisResults[index]);
            results[key] = parsed;
            this.memoryCache.set(key, parsed, 300);
            this.stats.redisHits++;
          } else {
            this.stats.misses++;
          }
        });
      } catch (error) {
        logger.error('Cache mget error:', error);
      }
    }
    
    return results;
  }

  // Cache invalidation
  async del(key) {
    this.memoryCache.del(key);
    if (this.redisClient) {
      await this.redisClient.del(key);
    }
  }

  // Pattern-based cache invalidation
  async delPattern(pattern) {
    // Clear memory cache
    const keys = this.memoryCache.keys();
    keys.forEach(key => {
      if (key.includes(pattern)) {
        this.memoryCache.del(key);
      }
    });
    
    // Clear Redis cache
    if (this.redisClient) {
      try {
        const keys = await this.redisClient.keys(`*${pattern}*`);
        if (keys.length > 0) {
          await this.redisClient.del(keys);
        }
      } catch (error) {
        logger.error('Cache delPattern error:', error);
      }
    }
  }

  // Cache statistics
  getStats() {
    const total = this.stats.memoryHits + this.stats.redisHits + this.stats.misses;
    const hitRate = total > 0 ? ((this.stats.memoryHits + this.stats.redisHits) / total * 100).toFixed(2) : 0;
    
    return {
      ...this.stats,
      hitRate: `${hitRate}%`,
      memoryKeys: this.memoryCache.keys().length,
      memorySize: this.memoryCache.getStats()
    };
  }

  // Warm up cache with frequently accessed data
  async warmUp(warmUpData) {
    logger.info('Warming up cache...');
    
    for (const [key, value] of Object.entries(warmUpData)) {
      await this.set(key, value, 3600); // 1 hour TTL for warm-up data
    }
    
    logger.info(`Cache warmed up with ${Object.keys(warmUpData).length} items`);
  }
}

// Singleton instance
const advancedCache = new AdvancedCache();

module.exports = {
  AdvancedCache,
  advancedCache
};