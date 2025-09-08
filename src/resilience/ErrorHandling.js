/**
 * 🛡️ ENTERPRISE ERROR HANDLING & RESILIENCE
 * 
 * Advanced error handling with circuit breakers, retries, bulkheads,
 * and graceful degradation patterns
 * 
 * @author Senior AI Backend Developer (35+ years)
 * @version 3.0.0
 */

const { v4: uuidv4 } = require('uuid');
const logger = require('../utils/logger');

/**
 * Custom Error Classes
 */
class BusinessError extends Error {
  constructor(message, code, details = {}) {
    super(message);
    this.name = 'BusinessError';
    this.code = code;
    this.details = details;
    this.isOperational = true;
  }
}

class ValidationError extends BusinessError {
  constructor(message, field, value) {
    super(message, 'VALIDATION_ERROR', { field, value });
    this.name = 'ValidationError';
  }
}

class AuthenticationError extends BusinessError {
  constructor(message = 'Authentication failed') {
    super(message, 'AUTH_ERROR');
    this.name = 'AuthenticationError';
  }
}

class AuthorizationError extends BusinessError {
  constructor(message = 'Insufficient permissions') {
    super(message, 'AUTHZ_ERROR');
    this.name = 'AuthorizationError';
  }
}

class ExternalServiceError extends Error {
  constructor(service, message, statusCode = 500) {
    super(`External service error: ${service} - ${message}`);
    this.name = 'ExternalServiceError';
    this.service = service;
    this.statusCode = statusCode;
    this.isOperational = true;
  }
}

class RateLimitError extends BusinessError {
  constructor(limit, window) {
    super(`Rate limit exceeded: ${limit} requests per ${window}ms`, 'RATE_LIMIT_ERROR');
    this.name = 'RateLimitError';
  }
}

/**
 * Advanced Circuit Breaker Implementation
 */
class AdvancedCircuitBreaker {
  constructor(options = {}) {
    this.name = options.name || 'default';
    this.failureThreshold = options.failureThreshold || 5;
    this.recoveryTimeout = options.recoveryTimeout || 60000;
    this.monitoringPeriod = options.monitoringPeriod || 10000;
    this.volumeThreshold = options.volumeThreshold || 10;
    this.errorThresholdPercentage = options.errorThresholdPercentage || 50;
    
    this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
    this.failureCount = 0;
    this.successCount = 0;
    this.requestCount = 0;
    this.lastFailureTime = null;
    this.lastSuccessTime = null;
    this.nextAttempt = null;
    
    this.metrics = {
      totalRequests: 0,
      totalFailures: 0,
      totalSuccesses: 0,
      averageResponseTime: 0,
      stateChanges: 0
    };

    // Start monitoring
    this.startMonitoring();
  }

  async execute(operation, fallback = null) {
    this.requestCount++;
    this.metrics.totalRequests++;

    if (this.state === 'OPEN') {
      if (Date.now() < this.nextAttempt) {
        logger.debug('🚫 Circuit breaker OPEN, executing fallback', { 
          name: this.name,
          nextAttempt: this.nextAttempt 
        });
        
        if (fallback) {
          return await fallback();
        }
        throw new Error(`Circuit breaker ${this.name} is OPEN`);
      } else {
        this.state = 'HALF_OPEN';
        this.metrics.stateChanges++;
        logger.info('🔄 Circuit breaker transitioning to HALF_OPEN', { name: this.name });
      }
    }

    const startTime = Date.now();
    
    try {
      const result = await operation();
      const responseTime = Date.now() - startTime;
      
      this.onSuccess(responseTime);
      return result;
      
    } catch (error) {
      const responseTime = Date.now() - startTime;
      this.onFailure(error, responseTime);
      
      if (fallback && this.state === 'OPEN') {
        logger.warn('⚠️ Executing fallback due to circuit breaker', { 
          name: this.name,
          error: error.message 
        });
        return await fallback();
      }
      
      throw error;
    }
  }

  onSuccess(responseTime) {
    this.successCount++;
    this.metrics.totalSuccesses++;
    this.lastSuccessTime = Date.now();
    
    // Update average response time
    this.updateAverageResponseTime(responseTime);

    if (this.state === 'HALF_OPEN') {
      if (this.successCount >= 3) {
        this.state = 'CLOSED';
        this.metrics.stateChanges++;
        this.resetCounts();
        logger.info('✅ Circuit breaker CLOSED after successful recovery', { name: this.name });
      }
    } else if (this.state === 'CLOSED') {
      this.failureCount = 0; // Reset failure count on success
    }
  }

  onFailure(error, responseTime) {
    this.failureCount++;
    this.metrics.totalFailures++;
    this.lastFailureTime = Date.now();
    
    // Update average response time
    this.updateAverageResponseTime(responseTime);

    logger.warn('⚠️ Circuit breaker recorded failure', {
      name: this.name,
      failureCount: this.failureCount,
      error: error.message
    });

    if (this.shouldTrip()) {
      this.trip();
    }
  }

  shouldTrip() {
    // Must have minimum volume
    if (this.requestCount < this.volumeThreshold) {
      return false;
    }

    // Check failure percentage
    const failurePercentage = (this.failureCount / this.requestCount) * 100;
    return failurePercentage >= this.errorThresholdPercentage;
  }

  trip() {
    this.state = 'OPEN';
    this.metrics.stateChanges++;
    this.nextAttempt = Date.now() + this.recoveryTimeout;
    
    logger.error('🔴 Circuit breaker TRIPPED', {
      name: this.name,
      failureCount: this.failureCount,
      requestCount: this.requestCount,
      nextAttempt: this.nextAttempt
    });
  }

  startMonitoring() {
    setInterval(() => {
      this.resetCounts();
    }, this.monitoringPeriod);
  }

  resetCounts() {
    this.failureCount = 0;
    this.successCount = 0;
    this.requestCount = 0;
  }

  updateAverageResponseTime(responseTime) {
    const totalRequests = this.metrics.totalRequests;
    this.metrics.averageResponseTime = 
      (this.metrics.averageResponseTime * (totalRequests - 1) + responseTime) / totalRequests;
  }

  getState() {
    return {
      name: this.name,
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount,
      requestCount: this.requestCount,
      lastFailureTime: this.lastFailureTime,
      lastSuccessTime: this.lastSuccessTime,
      nextAttempt: this.nextAttempt,
      metrics: { ...this.metrics }
    };
  }

  getHealthScore() {
    const totalRequests = this.metrics.totalRequests;
    if (totalRequests === 0) return 100;

    const successRate = (this.metrics.totalSuccesses / totalRequests) * 100;
    const stateScore = this.state === 'CLOSED' ? 100 : this.state === 'HALF_OPEN' ? 50 : 0;
    
    return Math.round((successRate * 0.7) + (stateScore * 0.3));
  }
}

/**
 * Retry Strategy Implementation
 */
class RetryStrategy {
  constructor(options = {}) {
    this.maxAttempts = options.maxAttempts || 3;
    this.baseDelay = options.baseDelay || 1000;
    this.maxDelay = options.maxDelay || 30000;
    this.backoffMultiplier = options.backoffMultiplier || 2;
    this.jitterMax = options.jitterMax || 1000;
    this.retryableErrors = options.retryableErrors || [
      'ECONNRESET',
      'ENOTFOUND',
      'ECONNREFUSED',
      'ETIMEDOUT'
    ];
  }

  async execute(operation, context = {}) {
    let lastError;
    
    for (let attempt = 1; attempt <= this.maxAttempts; attempt++) {
      try {
        const result = await operation();
        
        if (attempt > 1) {
          logger.info('✅ Operation succeeded after retry', {
            attempt,
            context
          });
        }
        
        return result;
        
      } catch (error) {
        lastError = error;
        
        if (attempt === this.maxAttempts || !this.isRetryableError(error)) {
          logger.error('❌ Operation failed after all retries', {
            attempt,
            maxAttempts: this.maxAttempts,
            error: error.message,
            context
          });
          break;
        }

        const delay = this.calculateDelay(attempt);
        
        logger.warn('⚠️ Operation failed, retrying', {
          attempt,
          maxAttempts: this.maxAttempts,
          delay,
          error: error.message,
          context
        });

        await this.sleep(delay);
      }
    }

    throw lastError;
  }

  isRetryableError(error) {
    if (error.isOperational === false) return false;
    if (error.name === 'ValidationError') return false;
    if (error.name === 'AuthenticationError') return false;
    if (error.name === 'AuthorizationError') return false;
    
    return this.retryableErrors.some(retryableError => 
      error.code === retryableError || 
      error.message.includes(retryableError)
    );
  }

  calculateDelay(attempt) {
    const exponentialDelay = this.baseDelay * Math.pow(this.backoffMultiplier, attempt - 1);
    const jitter = Math.random() * this.jitterMax;
    return Math.min(exponentialDelay + jitter, this.maxDelay);
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Bulkhead Pattern Implementation
 */
class Bulkhead {
  constructor(options = {}) {
    this.name = options.name || 'default';
    this.maxConcurrent = options.maxConcurrent || 10;
    this.maxQueue = options.maxQueue || 100;
    this.timeout = options.timeout || 30000;
    
    this.running = 0;
    this.queue = [];
    this.metrics = {
      totalRequests: 0,
      completedRequests: 0,
      rejectedRequests: 0,
      timeoutRequests: 0,
      averageWaitTime: 0
    };
  }

  async execute(operation, context = {}) {
    return new Promise((resolve, reject) => {
      const request = {
        id: uuidv4(),
        operation,
        context,
        resolve,
        reject,
        startTime: Date.now(),
        timeout: null
      };

      this.metrics.totalRequests++;

      if (this.running < this.maxConcurrent) {
        this.executeRequest(request);
      } else if (this.queue.length < this.maxQueue) {
        this.queue.push(request);
        
        // Set timeout for queued request
        request.timeout = setTimeout(() => {
          this.removeFromQueue(request.id);
          this.metrics.timeoutRequests++;
          reject(new Error(`Bulkhead ${this.name}: Request timeout in queue`));
        }, this.timeout);
        
      } else {
        this.metrics.rejectedRequests++;
        reject(new Error(`Bulkhead ${this.name}: Queue full`));
      }
    });
  }

  async executeRequest(request) {
    this.running++;
    
    // Clear timeout if request was queued
    if (request.timeout) {
      clearTimeout(request.timeout);
    }

    const waitTime = Date.now() - request.startTime;
    this.updateAverageWaitTime(waitTime);

    try {
      const result = await request.operation();
      this.metrics.completedRequests++;
      request.resolve(result);
    } catch (error) {
      request.reject(error);
    } finally {
      this.running--;
      this.processQueue();
    }
  }

  processQueue() {
    if (this.queue.length > 0 && this.running < this.maxConcurrent) {
      const request = this.queue.shift();
      this.executeRequest(request);
    }
  }

  removeFromQueue(requestId) {
    const index = this.queue.findIndex(req => req.id === requestId);
    if (index !== -1) {
      this.queue.splice(index, 1);
    }
  }

  updateAverageWaitTime(waitTime) {
    const completedRequests = this.metrics.completedRequests + 1;
    this.metrics.averageWaitTime = 
      (this.metrics.averageWaitTime * (completedRequests - 1) + waitTime) / completedRequests;
  }

  getState() {
    return {
      name: this.name,
      running: this.running,
      queued: this.queue.length,
      maxConcurrent: this.maxConcurrent,
      maxQueue: this.maxQueue,
      metrics: { ...this.metrics }
    };
  }

  getHealthScore() {
    const utilizationScore = Math.max(0, 100 - (this.running / this.maxConcurrent) * 100);
    const queueScore = Math.max(0, 100 - (this.queue.length / this.maxQueue) * 100);
    const rejectionRate = this.metrics.totalRequests > 0 
      ? (this.metrics.rejectedRequests / this.metrics.totalRequests) * 100 
      : 0;
    const rejectionScore = Math.max(0, 100 - rejectionRate);
    
    return Math.round((utilizationScore * 0.4) + (queueScore * 0.3) + (rejectionScore * 0.3));
  }
}

/**
 * Enterprise Error Handler
 */
class EnterpriseErrorHandler {
  constructor() {
    this.circuitBreakers = new Map();
    this.retryStrategies = new Map();
    this.bulkheads = new Map();
    this.errorStats = {
      totalErrors: 0,
      errorsByType: new Map(),
      errorsByService: new Map()
    };
  }

  /**
   * Register circuit breaker
   */
  registerCircuitBreaker(name, options) {
    const circuitBreaker = new AdvancedCircuitBreaker({ name, ...options });
    this.circuitBreakers.set(name, circuitBreaker);
    return circuitBreaker;
  }

  /**
   * Register retry strategy
   */
  registerRetryStrategy(name, options) {
    const retryStrategy = new RetryStrategy(options);
    this.retryStrategies.set(name, retryStrategy);
    return retryStrategy;
  }

  /**
   * Register bulkhead
   */
  registerBulkhead(name, options) {
    const bulkhead = new Bulkhead({ name, ...options });
    this.bulkheads.set(name, bulkhead);
    return bulkhead;
  }

  /**
   * Execute operation with resilience patterns
   */
  async executeWithResilience(operation, options = {}) {
    const {
      circuitBreaker,
      retryStrategy,
      bulkhead,
      fallback,
      context = {}
    } = options;

    let wrappedOperation = operation;

    // Apply bulkhead if specified
    if (bulkhead && this.bulkheads.has(bulkhead)) {
      const bulkheadInstance = this.bulkheads.get(bulkhead);
      wrappedOperation = () => bulkheadInstance.execute(operation, context);
    }

    // Apply retry strategy if specified
    if (retryStrategy && this.retryStrategies.has(retryStrategy)) {
      const retryInstance = this.retryStrategies.get(retryStrategy);
      const originalOperation = wrappedOperation;
      wrappedOperation = () => retryInstance.execute(originalOperation, context);
    }

    // Apply circuit breaker if specified
    if (circuitBreaker && this.circuitBreakers.has(circuitBreaker)) {
      const cbInstance = this.circuitBreakers.get(circuitBreaker);
      return cbInstance.execute(wrappedOperation, fallback);
    }

    return wrappedOperation();
  }

  /**
   * Handle error with proper classification and logging
   */
  handleError(error, context = {}) {
    this.errorStats.totalErrors++;
    
    // Classify error
    const errorType = this.classifyError(error);
    const errorCount = this.errorStats.errorsByType.get(errorType) || 0;
    this.errorStats.errorsByType.set(errorType, errorCount + 1);

    // Track by service if available
    if (context.service) {
      const serviceErrorCount = this.errorStats.errorsByService.get(context.service) || 0;
      this.errorStats.errorsByService.set(context.service, serviceErrorCount + 1);
    }

    // Log error with appropriate level
    const logLevel = this.getLogLevel(error);
    const errorInfo = {
      errorType,
      message: error.message,
      stack: error.stack,
      context,
      timestamp: new Date().toISOString()
    };

    logger[logLevel](`${errorType}: ${error.message}`, errorInfo);

    return {
      errorType,
      isOperational: error.isOperational || false,
      shouldRetry: this.shouldRetry(error),
      logLevel
    };
  }

  /**
   * Classify error type
   */
  classifyError(error) {
    if (error.name === 'ValidationError') return 'VALIDATION_ERROR';
    if (error.name === 'AuthenticationError') return 'AUTH_ERROR';
    if (error.name === 'AuthorizationError') return 'AUTHZ_ERROR';
    if (error.name === 'BusinessError') return 'BUSINESS_ERROR';
    if (error.name === 'ExternalServiceError') return 'EXTERNAL_SERVICE_ERROR';
    if (error.name === 'RateLimitError') return 'RATE_LIMIT_ERROR';
    if (error.code === 'ECONNRESET') return 'CONNECTION_ERROR';
    if (error.code === 'ETIMEDOUT') return 'TIMEOUT_ERROR';
    if (error.name === 'MongoError') return 'DATABASE_ERROR';
    return 'UNKNOWN_ERROR';
  }

  /**
   * Determine log level based on error type
   */
  getLogLevel(error) {
    if (error.name === 'ValidationError') return 'warn';
    if (error.name === 'AuthenticationError') return 'warn';
    if (error.name === 'AuthorizationError') return 'warn';
    if (error.name === 'BusinessError') return 'info';
    if (error.name === 'RateLimitError') return 'warn';
    return 'error';
  }

  /**
   * Determine if error should be retried
   */
  shouldRetry(error) {
    if (error.isOperational === false) return false;
    if (error.name === 'ValidationError') return false;
    if (error.name === 'AuthenticationError') return false;
    if (error.name === 'AuthorizationError') return false;
    if (error.name === 'BusinessError') return false;
    return true;
  }

  /**
   * Get system health based on error patterns
   */
  getSystemHealth() {
    const totalErrors = this.errorStats.totalErrors;
    const criticalErrors = (this.errorStats.errorsByType.get('DATABASE_ERROR') || 0) +
                          (this.errorStats.errorsByType.get('EXTERNAL_SERVICE_ERROR') || 0);
    
    let healthScore = 100;
    
    if (totalErrors > 0) {
      const criticalErrorRate = (criticalErrors / totalErrors) * 100;
      healthScore -= criticalErrorRate;
    }

    // Factor in circuit breaker states
    for (const cb of this.circuitBreakers.values()) {
      const cbHealth = cb.getHealthScore();
      healthScore = (healthScore + cbHealth) / 2;
    }

    // Factor in bulkhead states
    for (const bulkhead of this.bulkheads.values()) {
      const bulkheadHealth = bulkhead.getHealthScore();
      healthScore = (healthScore + bulkheadHealth) / 2;
    }

    return {
      score: Math.max(0, Math.round(healthScore)),
      status: healthScore > 80 ? 'HEALTHY' : healthScore > 60 ? 'DEGRADED' : 'UNHEALTHY',
      errorStats: this.errorStats,
      circuitBreakers: Array.from(this.circuitBreakers.values()).map(cb => cb.getState()),
      bulkheads: Array.from(this.bulkheads.values()).map(b => b.getState())
    };
  }
}

module.exports = {
  EnterpriseErrorHandler,
  AdvancedCircuitBreaker,
  RetryStrategy,
  Bulkhead,
  BusinessError,
  ValidationError,
  AuthenticationError,
  AuthorizationError,
  ExternalServiceError,
  RateLimitError
};
