/**
 * Enterprise Redis Configuration
 * High-performance caching layer for financial data
 * @module config/redis
 */

const redis = require('redis');
const logger = require('../utils/logger');

/**
 * Redis Configuration for Different Environments
 */
const redisConfig = {
  development: {
    host: process.env.REDIS_HOST || 'localhost',
    port: process.env.REDIS_PORT || 6379,
    password: process.env.REDIS_PASSWORD,
    db: 0,
    // Connection Pool Settings
    maxRetriesPerRequest: 3,
    retryDelayOnFailover: 100,
    enableReadyCheck: true,
    maxLoadingTimeout: 5000,
    // Performance Settings
    lazyConnect: true,
    keepAlive: 30000,
    connectTimeout: 10000,
    commandTimeout: 5000
  },
  test: {
    host: process.env.REDIS_TEST_HOST || 'localhost',
    port: process.env.REDIS_TEST_PORT || 6379,
    password: process.env.REDIS_TEST_PASSWORD,
    db: 1, // Separate database for testing
    maxRetriesPerRequest: 1,
    connectTimeout: 3000,
    commandTimeout: 2000
  },
  production: {
    host: process.env.REDIS_HOST,
    port: process.env.REDIS_PORT || 6379,
    password: process.env.REDIS_PASSWORD,
    db: 0,
    // Production Performance Optimization
    maxRetriesPerRequest: 5,
    retryDelayOnFailover: 200,
    enableReadyCheck: true,
    maxLoadingTimeout: 10000,
    lazyConnect: false, // Connect immediately in production
    keepAlive: 60000, // Longer keep-alive for production
    connectTimeout: 15000,
    commandTimeout: 10000,
    // Production Security
    tls: process.env.REDIS_TLS === 'true' ? {
      rejectUnauthorized: true
    } : undefined,
    // Connection Pool for Production
    family: 4, // Use IPv4
    enableOfflineQueue: false // Fail fast if Redis is down
  }
};

/**
 * Redis Client Instance
 */
let redisClient = null;

/**
 * Initialize Redis Connection
 */
const initializeRedis = async () => {
  try {
    const env = process.env.NODE_ENV || 'development';
    const config = redisConfig[env];
    
    if (!config) {
      throw new Error(`Redis configuration not found for environment: ${env}`);
    }

    // Create Redis client with configuration
    redisClient = redis.createClient({
      socket: {
        host: config.host,
        port: config.port,
        connectTimeout: config.connectTimeout,
        commandTimeout: config.commandTimeout,
        keepAlive: config.keepAlive,
        tls: config.tls
      },
      password: config.password,
      database: config.db,
      // Retry Strategy
      retry_strategy: (options) => {
        if (options.error && options.error.code === 'ECONNREFUSED') {
          logger.error('Redis server refused connection');
          return new Error('Redis server refused connection');
        }
        if (options.total_retry_time > 1000 * 60 * 60) {
          logger.error('Redis retry time exhausted');
          return new Error('Redis retry time exhausted');
        }
        if (options.attempt > config.maxRetriesPerRequest) {
          logger.error('Redis max retries exceeded');
          return undefined;
        }
        // Exponential backoff
        return Math.min(options.attempt * 100, 3000);
      }
    });

    // Event Handlers
    redisClient.on('connect', () => {
      logger.info('✅ Redis client connected');
    });

    redisClient.on('ready', () => {
      logger.info('✅ Redis client ready for commands');
    });

    redisClient.on('error', (err) => {
      logger.error('❌ Redis client error:', err);
    });

    redisClient.on('end', () => {
      logger.warn('⚠️  Redis client connection ended');
    });

    redisClient.on('reconnecting', () => {
      logger.info('🔄 Redis client reconnecting...');
    });

    // Connect to Redis
    await redisClient.connect();
    
    // Test connection
    await redisClient.ping();
    logger.info('✅ Redis connection established successfully');
    
    return redisClient;
    
  } catch (error) {
    logger.error('❌ Failed to initialize Redis:', error.message);
    
    // In development, Redis is optional
    if (process.env.NODE_ENV === 'development') {
      logger.warn('⚠️  Redis not available in development - caching disabled');
      return null;
    }
    
    // In production, Redis is required
    if (process.env.NODE_ENV === 'production') {
      throw error;
    }
    
    return null;
  }
};

/**
 * Get Redis Client Instance
 */
const getRedisClient = () => {
  return redisClient;
};

/**
 * Close Redis Connection
 */
const closeRedis = async () => {
  if (redisClient) {
    try {
      await redisClient.quit();
      logger.info('✅ Redis connection closed gracefully');
    } catch (error) {
      logger.error('❌ Error closing Redis connection:', error.message);
    }
  }
};

/**
 * Redis Health Check
 */
const redisHealthCheck = async () => {
  try {
    if (!redisClient) {
      return { status: 'disconnected', message: 'Redis client not initialized' };
    }
    
    const start = Date.now();
    await redisClient.ping();
    const latency = Date.now() - start;
    
    return {
      status: 'connected',
      latency: `${latency}ms`,
      message: 'Redis is healthy'
    };
  } catch (error) {
    return {
      status: 'error',
      message: error.message
    };
  }
};

module.exports = {
  initializeRedis,
  getRedisClient,
  closeRedis,
  redisHealthCheck,
  redisConfig
};
