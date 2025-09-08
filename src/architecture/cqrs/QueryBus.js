/**
 * 🔍 ENTERPRISE CQRS QUERY BUS
 * 
 * High-performance query processing with caching, pagination, and optimization
 * Handles read operations separately from write operations (CQRS pattern)
 * 
 * @author Senior AI Backend Developer (35+ years)
 * @version 3.0.0
 */

const { v4: uuidv4 } = require('uuid');
const logger = require('../../utils/logger');
const { cacheService } = require('../../services/cacheService');

class QueryBus {
  constructor() {
    this.queryHandlers = new Map();
    this.middleware = [];
    this.cache = cacheService;
    this.metrics = {
      queriesProcessed: 0,
      cacheHits: 0,
      cacheMisses: 0,
      averageProcessingTime: 0
    };
  }

  /**
   * Register query handler
   */
  registerHandler(queryType, handler, options = {}) {
    if (this.queryHandlers.has(queryType)) {
      throw new Error(`Query handler already registered for: ${queryType}`);
    }

    const handlerConfig = {
      handler,
      options: {
        timeout: options.timeout || 15000,
        cacheable: options.cacheable !== false,
        cacheTTL: options.cacheTTL || 300, // 5 minutes default
        cacheKey: options.cacheKey || null,
        authorization: options.authorization || null,
        validation: options.validation || null,
        pagination: options.pagination || false,
        sorting: options.sorting || false,
        filtering: options.filtering || false
      }
    };

    this.queryHandlers.set(queryType, handlerConfig);
    logger.info('🔍 Query handler registered', { queryType, options: handlerConfig.options });
  }

  /**
   * Add middleware to query processing pipeline
   */
  use(middleware) {
    this.middleware.push(middleware);
    logger.info('🔧 Query middleware added', { middlewareCount: this.middleware.length });
  }

  /**
   * Execute query
   */
  async execute(query) {
    const startTime = Date.now();
    const queryId = uuidv4();
    const context = {
      queryId,
      query,
      timestamp: new Date().toISOString(),
      user: query.user || null,
      correlationId: query.correlationId || uuidv4()
    };

    try {
      logger.debug('🔍 Executing query', {
        queryType: query.type,
        queryId,
        correlationId: context.correlationId
      });

      // Validate query structure
      this.validateQuery(query);

      // Get handler
      const handlerConfig = this.getHandler(query.type);

      // Apply middleware pipeline
      await this.applyMiddleware(context, handlerConfig);

      // Check cache first
      if (handlerConfig.options.cacheable) {
        const cachedResult = await this.getCachedResult(query, handlerConfig);
        if (cachedResult) {
          this.updateMetrics(Date.now() - startTime, true, true);
          logger.debug('✅ Query served from cache', {
            queryType: query.type,
            queryId,
            processingTime: Date.now() - startTime
          });
          return cachedResult;
        }
      }

      // Execute query with timeout
      const result = await this.executeWithTimeout(context, handlerConfig);

      // Process result (pagination, sorting, filtering)
      const processedResult = await this.processResult(result, query, handlerConfig);

      // Cache result if cacheable
      if (handlerConfig.options.cacheable && processedResult) {
        await this.cacheResult(query, processedResult, handlerConfig);
      }

      // Update metrics
      this.updateMetrics(Date.now() - startTime, true, false);

      logger.debug('✅ Query executed successfully', {
        queryType: query.type,
        queryId,
        processingTime: Date.now() - startTime,
        resultSize: this.getResultSize(processedResult)
      });

      return processedResult;

    } catch (error) {
      // Update metrics
      this.updateMetrics(Date.now() - startTime, false, false);

      logger.error('❌ Query execution failed', {
        queryType: query.type,
        queryId,
        error: error.message,
        processingTime: Date.now() - startTime
      });

      throw error;
    }
  }

  /**
   * Validate query structure
   */
  validateQuery(query) {
    if (!query || typeof query !== 'object') {
      throw new Error('Query must be an object');
    }

    if (!query.type || typeof query.type !== 'string') {
      throw new Error('Query must have a type property');
    }

    // Payload is optional for queries
  }

  /**
   * Get query handler
   */
  getHandler(queryType) {
    const handlerConfig = this.queryHandlers.get(queryType);
    if (!handlerConfig) {
      throw new Error(`No handler registered for query type: ${queryType}`);
    }
    return handlerConfig;
  }

  /**
   * Apply middleware pipeline
   */
  async applyMiddleware(context, handlerConfig) {
    for (const middleware of this.middleware) {
      await middleware(context, handlerConfig);
    }

    // Apply handler-specific validation
    if (handlerConfig.options.validation) {
      await handlerConfig.options.validation(context.query);
    }

    // Apply handler-specific authorization
    if (handlerConfig.options.authorization) {
      await handlerConfig.options.authorization(context.query, context.user);
    }
  }

  /**
   * Get cached result
   */
  async getCachedResult(query, handlerConfig) {
    try {
      const cacheKey = this.generateCacheKey(query, handlerConfig);
      const cachedResult = await this.cache.get(cacheKey);
      
      if (cachedResult) {
        this.metrics.cacheHits++;
        return JSON.parse(cachedResult);
      }
      
      this.metrics.cacheMisses++;
      return null;
    } catch (error) {
      logger.warn('⚠️ Cache retrieval failed', { error: error.message });
      this.metrics.cacheMisses++;
      return null;
    }
  }

  /**
   * Cache result
   */
  async cacheResult(query, result, handlerConfig) {
    try {
      const cacheKey = this.generateCacheKey(query, handlerConfig);
      const ttl = handlerConfig.options.cacheTTL;
      
      await this.cache.set(cacheKey, JSON.stringify(result), ttl);
      
      logger.debug('💾 Query result cached', {
        cacheKey,
        ttl,
        resultSize: this.getResultSize(result)
      });
    } catch (error) {
      logger.warn('⚠️ Cache storage failed', { error: error.message });
    }
  }

  /**
   * Generate cache key
   */
  generateCacheKey(query, handlerConfig) {
    if (handlerConfig.options.cacheKey) {
      return handlerConfig.options.cacheKey(query);
    }

    // Default cache key generation
    const keyParts = [
      'query',
      query.type,
      JSON.stringify(query.payload || {}),
      query.user?.id || 'anonymous'
    ];

    return keyParts.join(':');
  }

  /**
   * Execute query with timeout
   */
  async executeWithTimeout(context, handlerConfig) {
    return Promise.race([
      handlerConfig.handler(context.query, context),
      this.createTimeout(handlerConfig.options.timeout)
    ]);
  }

  /**
   * Process result (pagination, sorting, filtering)
   */
  async processResult(result, query, handlerConfig) {
    let processedResult = result;

    // Apply filtering
    if (handlerConfig.options.filtering && query.filters) {
      processedResult = this.applyFilters(processedResult, query.filters);
    }

    // Apply sorting
    if (handlerConfig.options.sorting && query.sort) {
      processedResult = this.applySorting(processedResult, query.sort);
    }

    // Apply pagination
    if (handlerConfig.options.pagination && query.pagination) {
      processedResult = this.applyPagination(processedResult, query.pagination);
    }

    return processedResult;
  }

  /**
   * Apply filters to result
   */
  applyFilters(result, filters) {
    if (!Array.isArray(result)) {
      return result;
    }

    return result.filter(item => {
      return Object.entries(filters).every(([key, value]) => {
        if (typeof value === 'object' && value.operator) {
          return this.applyFilterOperator(item[key], value.operator, value.value);
        }
        return item[key] === value;
      });
    });
  }

  /**
   * Apply filter operator
   */
  applyFilterOperator(itemValue, operator, filterValue) {
    switch (operator) {
      case 'eq': return itemValue === filterValue;
      case 'ne': return itemValue !== filterValue;
      case 'gt': return itemValue > filterValue;
      case 'gte': return itemValue >= filterValue;
      case 'lt': return itemValue < filterValue;
      case 'lte': return itemValue <= filterValue;
      case 'in': return Array.isArray(filterValue) && filterValue.includes(itemValue);
      case 'nin': return Array.isArray(filterValue) && !filterValue.includes(itemValue);
      case 'contains': return String(itemValue).toLowerCase().includes(String(filterValue).toLowerCase());
      case 'startsWith': return String(itemValue).toLowerCase().startsWith(String(filterValue).toLowerCase());
      case 'endsWith': return String(itemValue).toLowerCase().endsWith(String(filterValue).toLowerCase());
      default: return true;
    }
  }

  /**
   * Apply sorting to result
   */
  applySorting(result, sort) {
    if (!Array.isArray(result)) {
      return result;
    }

    return result.sort((a, b) => {
      for (const sortRule of sort) {
        const { field, direction = 'asc' } = sortRule;
        const aValue = a[field];
        const bValue = b[field];

        if (aValue < bValue) {
          return direction === 'asc' ? -1 : 1;
        }
        if (aValue > bValue) {
          return direction === 'asc' ? 1 : -1;
        }
      }
      return 0;
    });
  }

  /**
   * Apply pagination to result
   */
  applyPagination(result, pagination) {
    if (!Array.isArray(result)) {
      return {
        data: result,
        pagination: {
          page: 1,
          limit: 1,
          total: 1,
          pages: 1
        }
      };
    }

    const { page = 1, limit = 10 } = pagination;
    const offset = (page - 1) * limit;
    const total = result.length;
    const pages = Math.ceil(total / limit);
    const data = result.slice(offset, offset + limit);

    return {
      data,
      pagination: {
        page,
        limit,
        total,
        pages,
        hasNext: page < pages,
        hasPrev: page > 1
      }
    };
  }

  /**
   * Create timeout promise
   */
  createTimeout(ms) {
    return new Promise((_, reject) => {
      setTimeout(() => reject(new Error(`Query timeout after ${ms}ms`)), ms);
    });
  }

  /**
   * Get result size for logging
   */
  getResultSize(result) {
    if (Array.isArray(result)) {
      return result.length;
    }
    if (result && typeof result === 'object') {
      return Object.keys(result).length;
    }
    return 1;
  }

  /**
   * Update metrics
   */
  updateMetrics(processingTime, success, fromCache) {
    this.metrics.queriesProcessed++;

    // Update average processing time
    this.metrics.averageProcessingTime = 
      (this.metrics.averageProcessingTime * (this.metrics.queriesProcessed - 1) + processingTime) / 
      this.metrics.queriesProcessed;
  }

  /**
   * Get query bus metrics
   */
  getMetrics() {
    const totalQueries = this.metrics.queriesProcessed;
    const cacheTotal = this.metrics.cacheHits + this.metrics.cacheMisses;
    
    return {
      ...this.metrics,
      cacheHitRate: cacheTotal > 0 ? (this.metrics.cacheHits / cacheTotal) * 100 : 0,
      registeredHandlers: this.queryHandlers.size,
      middlewareCount: this.middleware.length
    };
  }

  /**
   * Invalidate cache for query type
   */
  async invalidateCache(queryType, pattern = null) {
    try {
      const cachePattern = pattern || `query:${queryType}:*`;
      await this.cache.deletePattern(cachePattern);
      logger.info('🗑️ Cache invalidated', { queryType, pattern: cachePattern });
    } catch (error) {
      logger.warn('⚠️ Cache invalidation failed', { error: error.message });
    }
  }
}

/**
 * Common query middleware functions
 */
class QueryMiddleware {
  /**
   * Logging middleware
   */
  static logging() {
    return async (context, handlerConfig) => {
      logger.debug('📝 Query middleware: Logging', {
        queryType: context.query.type,
        queryId: context.queryId,
        user: context.user?.id
      });
    };
  }

  /**
   * Performance monitoring middleware
   */
  static performanceMonitoring() {
    return async (context, handlerConfig) => {
      context.startTime = Date.now();
      
      // Log slow queries after execution
      const originalHandler = handlerConfig.handler;
      handlerConfig.handler = async (...args) => {
        const result = await originalHandler(...args);
        const duration = Date.now() - context.startTime;
        
        if (duration > 5000) { // Log queries taking more than 5 seconds
          logger.warn('🐌 Slow query detected', {
            queryType: context.query.type,
            duration,
            queryId: context.queryId
          });
        }
        
        return result;
      };
    };
  }

  /**
   * Result size limiting middleware
   */
  static resultSizeLimit(maxSize = 1000) {
    return async (context, handlerConfig) => {
      const originalHandler = handlerConfig.handler;
      handlerConfig.handler = async (...args) => {
        const result = await originalHandler(...args);
        
        if (Array.isArray(result) && result.length > maxSize) {
          logger.warn('📏 Large result set detected', {
            queryType: context.query.type,
            resultSize: result.length,
            maxSize
          });
          
          // Optionally truncate or throw error
          return result.slice(0, maxSize);
        }
        
        return result;
      };
    };
  }
}

module.exports = { QueryBus, QueryMiddleware };
