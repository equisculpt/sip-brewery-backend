/**
 * Database Query Optimization Service
 * Enterprise-grade query optimization for financial data
 * @module services/queryOptimizationService
 */

const mongoose = require('mongoose');
const { cacheService } = require('./cacheService');
const logger = require('../utils/logger');

/**
 * Query Performance Monitoring
 */
class QueryPerformanceMonitor {
  constructor() {
    this.slowQueryThreshold = process.env.SLOW_QUERY_THRESHOLD || 1000; // 1 second
    this.queryStats = new Map();
  }

  /**
   * Monitor query execution time
   */
  async monitorQuery(queryName, queryFunction, cacheKey = null, cacheTTL = 300) {
    const startTime = Date.now();
    
    try {
      // Check cache first if cacheKey provided
      if (cacheKey && cacheService.isEnabled) {
        const cachedResult = await cacheService.get(cacheKey);
        if (cachedResult) {
          const endTime = Date.now();
          this.logQueryPerformance(queryName, endTime - startTime, true);
          return cachedResult;
        }
      }

      // Execute query
      const result = await queryFunction();
      const endTime = Date.now();
      const executionTime = endTime - startTime;

      // Cache result if cacheKey provided
      if (cacheKey && cacheService.isEnabled && result) {
        await cacheService.set(cacheKey, result, cacheTTL);
      }

      // Log performance
      this.logQueryPerformance(queryName, executionTime, false);
      
      // Update query statistics
      this.updateQueryStats(queryName, executionTime);

      return result;
    } catch (error) {
      const endTime = Date.now();
      const executionTime = endTime - startTime;
      
      logger.error(`Query error: ${queryName}`, {
        error: error.message,
        executionTime: `${executionTime}ms`,
        stack: error.stack
      });
      
      throw error;
    }
  }

  /**
   * Log query performance
   */
  logQueryPerformance(queryName, executionTime, fromCache) {
    const logData = {
      query: queryName,
      executionTime: `${executionTime}ms`,
      fromCache,
      timestamp: new Date().toISOString()
    };

    if (fromCache) {
      logger.debug('Query cache hit', logData);
    } else if (executionTime > this.slowQueryThreshold) {
      logger.warn('Slow query detected', logData);
    } else {
      logger.debug('Query executed', logData);
    }
  }

  /**
   * Update query statistics
   */
  updateQueryStats(queryName, executionTime) {
    if (!this.queryStats.has(queryName)) {
      this.queryStats.set(queryName, {
        count: 0,
        totalTime: 0,
        minTime: Infinity,
        maxTime: 0,
        avgTime: 0
      });
    }

    const stats = this.queryStats.get(queryName);
    stats.count++;
    stats.totalTime += executionTime;
    stats.minTime = Math.min(stats.minTime, executionTime);
    stats.maxTime = Math.max(stats.maxTime, executionTime);
    stats.avgTime = stats.totalTime / stats.count;
  }

  /**
   * Get query statistics
   */
  getQueryStats() {
    const stats = {};
    this.queryStats.forEach((value, key) => {
      stats[key] = {
        ...value,
        minTime: `${value.minTime}ms`,
        maxTime: `${value.maxTime}ms`,
        avgTime: `${Math.round(value.avgTime)}ms`,
        totalTime: `${value.totalTime}ms`
      };
    });
    return stats;
  }
}

/**
 * MongoDB Query Optimization Utilities
 */
class MongoQueryOptimizer {
  constructor() {
    this.performanceMonitor = new QueryPerformanceMonitor();
  }

  /**
   * Optimized pagination with cursor-based approach
   */
  async paginateQuery(model, query = {}, options = {}) {
    const {
      limit = 20,
      cursor = null,
      sortField = '_id',
      sortOrder = 1,
      select = null,
      populate = null
    } = options;

    const queryName = `paginate_${model.modelName}`;
    const cacheKey = `pagination:${model.modelName}:${JSON.stringify(query)}:${cursor}:${limit}`;

    return await this.performanceMonitor.monitorQuery(
      queryName,
      async () => {
        let mongoQuery = model.find(query);

        // Apply cursor-based pagination
        if (cursor) {
          const cursorCondition = sortOrder === 1 
            ? { [sortField]: { $gt: cursor } }
            : { [sortField]: { $lt: cursor } };
          mongoQuery = mongoQuery.find(cursorCondition);
        }

        // Apply sorting
        mongoQuery = mongoQuery.sort({ [sortField]: sortOrder });

        // Apply limit
        mongoQuery = mongoQuery.limit(limit + 1); // +1 to check if there's a next page

        // Apply field selection
        if (select) {
          mongoQuery = mongoQuery.select(select);
        }

        // Apply population
        if (populate) {
          mongoQuery = mongoQuery.populate(populate);
        }

        const results = await mongoQuery.lean().exec();
        
        const hasNextPage = results.length > limit;
        if (hasNextPage) {
          results.pop(); // Remove the extra item
        }

        const nextCursor = hasNextPage && results.length > 0 
          ? results[results.length - 1][sortField]
          : null;

        return {
          data: results,
          pagination: {
            hasNextPage,
            nextCursor,
            limit,
            count: results.length
          }
        };
      },
      cacheKey,
      300 // 5 minutes cache
    );
  }

  /**
   * Optimized aggregation with caching
   */
  async aggregateWithCache(model, pipeline, cacheKey, cacheTTL = 600) {
    const queryName = `aggregate_${model.modelName}`;

    return await this.performanceMonitor.monitorQuery(
      queryName,
      async () => {
        return await model.aggregate(pipeline).exec();
      },
      cacheKey,
      cacheTTL
    );
  }

  /**
   * Optimized find with intelligent indexing hints
   */
  async optimizedFind(model, query, options = {}) {
    const queryName = `find_${model.modelName}`;
    const cacheKey = options.cacheKey;
    const cacheTTL = options.cacheTTL || 300;

    return await this.performanceMonitor.monitorQuery(
      queryName,
      async () => {
        let mongoQuery = model.find(query);

        // Apply optimizations
        if (options.select) {
          mongoQuery = mongoQuery.select(options.select);
        }

        if (options.populate) {
          mongoQuery = mongoQuery.populate(options.populate);
        }

        if (options.sort) {
          mongoQuery = mongoQuery.sort(options.sort);
        }

        if (options.limit) {
          mongoQuery = mongoQuery.limit(options.limit);
        }

        if (options.skip) {
          mongoQuery = mongoQuery.skip(options.skip);
        }

        // Use lean() for better performance when you don't need Mongoose documents
        if (options.lean !== false) {
          mongoQuery = mongoQuery.lean();
        }

        // Add query hints for better index usage
        if (options.hint) {
          mongoQuery = mongoQuery.hint(options.hint);
        }

        return await mongoQuery.exec();
      },
      cacheKey,
      cacheTTL
    );
  }

  /**
   * Bulk operations optimization
   */
  async optimizedBulkWrite(model, operations, options = {}) {
    const queryName = `bulk_${model.modelName}`;
    
    return await this.performanceMonitor.monitorQuery(
      queryName,
      async () => {
        const bulkOptions = {
          ordered: false, // Allow parallel execution
          ...options
        };

        return await model.bulkWrite(operations, bulkOptions);
      }
    );
  }

  /**
   * Get database performance statistics
   */
  async getDatabaseStats() {
    try {
      const db = mongoose.connection.db;
      const stats = await db.stats();
      
      return {
        database: {
          collections: stats.collections,
          dataSize: `${Math.round(stats.dataSize / 1024 / 1024)}MB`,
          storageSize: `${Math.round(stats.storageSize / 1024 / 1024)}MB`,
          indexSize: `${Math.round(stats.indexSize / 1024 / 1024)}MB`,
          objects: stats.objects
        },
        queries: this.performanceMonitor.getQueryStats(),
        connection: {
          readyState: mongoose.connection.readyState,
          host: mongoose.connection.host,
          port: mongoose.connection.port,
          name: mongoose.connection.name
        }
      };
    } catch (error) {
      logger.error('Error getting database stats:', error.message);
      return {
        error: error.message,
        queries: this.performanceMonitor.getQueryStats()
      };
    }
  }
}

/**
 * Financial Data Specific Optimizations
 */
class FinancialDataOptimizer extends MongoQueryOptimizer {
  /**
   * Optimized portfolio data retrieval
   */
  async getPortfolioData(userId, options = {}) {
    const cacheKey = `portfolio:${userId}:${JSON.stringify(options)}`;
    
    return await this.performanceMonitor.monitorQuery(
      'get_portfolio_data',
      async () => {
        // Implementation would depend on your portfolio model
        // This is a template for optimization patterns
        const query = { userId };
        
        if (options.dateRange) {
          query.createdAt = {
            $gte: options.dateRange.start,
            $lte: options.dateRange.end
          };
        }

        return await this.optimizedFind(
          require('../models/Portfolio'), // Assuming you have this model
          query,
          {
            select: 'fundId quantity currentValue lastUpdated',
            populate: {
              path: 'fundId',
              select: 'name symbol currentPrice'
            },
            sort: { lastUpdated: -1 },
            lean: true
          }
        );
      },
      cacheKey,
      600 // 10 minutes cache for portfolio data
    );
  }

  /**
   * Optimized market data aggregation
   */
  async getMarketSummary(symbols = [], options = {}) {
    const cacheKey = `market_summary:${symbols.join(',')}:${JSON.stringify(options)}`;
    
    return await this.performanceMonitor.monitorQuery(
      'get_market_summary',
      async () => {
        const pipeline = [
          { $match: { symbol: { $in: symbols } } },
          {
            $group: {
              _id: '$symbol',
              currentPrice: { $last: '$price' },
              dayHigh: { $max: '$price' },
              dayLow: { $min: '$price' },
              volume: { $sum: '$volume' },
              lastUpdated: { $max: '$timestamp' }
            }
          },
          { $sort: { _id: 1 } }
        ];

        return await this.aggregateWithCache(
          require('../models/MarketData'), // Assuming you have this model
          pipeline,
          cacheKey,
          60 // 1 minute cache for market data
        );
      },
      cacheKey,
      60
    );
  }

  /**
   * Optimized rewards calculation
   */
  async calculateUserRewards(userId, options = {}) {
    const cacheKey = `rewards:${userId}:${JSON.stringify(options)}`;
    
    return await this.performanceMonitor.monitorQuery(
      'calculate_user_rewards',
      async () => {
        const pipeline = [
          { $match: { userId: new mongoose.Types.ObjectId(userId) } },
          {
            $group: {
              _id: '$type',
              totalRewards: { $sum: '$amount' },
              count: { $sum: 1 },
              lastEarned: { $max: '$createdAt' }
            }
          },
          {
            $group: {
              _id: null,
              totalAmount: { $sum: '$totalRewards' },
              rewardTypes: {
                $push: {
                  type: '$_id',
                  amount: '$totalRewards',
                  count: '$count',
                  lastEarned: '$lastEarned'
                }
              }
            }
          }
        ];

        return await this.aggregateWithCache(
          require('../models/Reward'), // Assuming you have this model
          pipeline,
          cacheKey,
          900 // 15 minutes cache for rewards
        );
      },
      cacheKey,
      900
    );
  }
}

// Export singleton instance
const queryOptimizer = new FinancialDataOptimizer();

module.exports = {
  queryOptimizer,
  QueryPerformanceMonitor,
  MongoQueryOptimizer,
  FinancialDataOptimizer
};
