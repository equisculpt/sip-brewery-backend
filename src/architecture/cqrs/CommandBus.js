/**
 * 🏗️ ENTERPRISE CQRS COMMAND BUS
 * 
 * Command Query Responsibility Segregation pattern implementation
 * Separates write operations (commands) from read operations (queries)
 * 
 * @author Senior AI Backend Developer (35+ years)
 * @version 3.0.0
 */

const { v4: uuidv4 } = require('uuid');
const logger = require('../../utils/logger');
const { EnterpriseEventBus } = require('../eventBus');

class CommandBus {
  constructor(eventBus) {
    this.eventBus = eventBus || new EnterpriseEventBus();
    this.commandHandlers = new Map();
    this.middleware = [];
    this.metrics = {
      commandsProcessed: 0,
      commandsFailed: 0,
      averageProcessingTime: 0
    };
  }

  /**
   * Register command handler
   */
  registerHandler(commandType, handler, options = {}) {
    if (this.commandHandlers.has(commandType)) {
      throw new Error(`Command handler already registered for: ${commandType}`);
    }

    const handlerConfig = {
      handler,
      options: {
        timeout: options.timeout || 30000,
        retries: options.retries || 3,
        validation: options.validation || null,
        authorization: options.authorization || null,
        idempotent: options.idempotent || false
      }
    };

    this.commandHandlers.set(commandType, handlerConfig);
    logger.info('📝 Command handler registered', { commandType, options: handlerConfig.options });
  }

  /**
   * Add middleware to command processing pipeline
   */
  use(middleware) {
    this.middleware.push(middleware);
    logger.info('🔧 Command middleware added', { middlewareCount: this.middleware.length });
  }

  /**
   * Execute command
   */
  async execute(command) {
    const startTime = Date.now();
    const commandId = uuidv4();
    const context = {
      commandId,
      command,
      timestamp: new Date().toISOString(),
      user: command.user || null,
      correlationId: command.correlationId || uuidv4()
    };

    try {
      logger.info('🚀 Executing command', {
        commandType: command.type,
        commandId,
        correlationId: context.correlationId
      });

      // Validate command structure
      this.validateCommand(command);

      // Get handler
      const handlerConfig = this.getHandler(command.type);

      // Apply middleware pipeline
      await this.applyMiddleware(context, handlerConfig);

      // Check idempotency
      if (handlerConfig.options.idempotent) {
        const existingResult = await this.checkIdempotency(command);
        if (existingResult) {
          logger.info('🔄 Command already processed (idempotent)', { commandId });
          return existingResult;
        }
      }

      // Execute command with timeout and retries
      const result = await this.executeWithRetries(context, handlerConfig);

      // Store idempotency result
      if (handlerConfig.options.idempotent) {
        await this.storeIdempotencyResult(command, result);
      }

      // Publish command executed event
      await this.eventBus.publish('command.executed', {
        commandId,
        commandType: command.type,
        result,
        processingTime: Date.now() - startTime
      }, { correlationId: context.correlationId });

      // Update metrics
      this.updateMetrics(Date.now() - startTime, true);

      logger.info('✅ Command executed successfully', {
        commandType: command.type,
        commandId,
        processingTime: Date.now() - startTime
      });

      return result;

    } catch (error) {
      // Update metrics
      this.updateMetrics(Date.now() - startTime, false);

      // Publish command failed event
      await this.eventBus.publish('command.failed', {
        commandId,
        commandType: command.type,
        error: error.message,
        processingTime: Date.now() - startTime
      }, { correlationId: context.correlationId });

      logger.error('❌ Command execution failed', {
        commandType: command.type,
        commandId,
        error: error.message,
        processingTime: Date.now() - startTime
      });

      throw error;
    }
  }

  /**
   * Validate command structure
   */
  validateCommand(command) {
    if (!command || typeof command !== 'object') {
      throw new Error('Command must be an object');
    }

    if (!command.type || typeof command.type !== 'string') {
      throw new Error('Command must have a type property');
    }

    if (!command.payload) {
      throw new Error('Command must have a payload property');
    }
  }

  /**
   * Get command handler
   */
  getHandler(commandType) {
    const handlerConfig = this.commandHandlers.get(commandType);
    if (!handlerConfig) {
      throw new Error(`No handler registered for command type: ${commandType}`);
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
      await handlerConfig.options.validation(context.command);
    }

    // Apply handler-specific authorization
    if (handlerConfig.options.authorization) {
      await handlerConfig.options.authorization(context.command, context.user);
    }
  }

  /**
   * Execute command with retries
   */
  async executeWithRetries(context, handlerConfig) {
    const maxRetries = handlerConfig.options.retries;
    let lastError;

    for (let attempt = 1; attempt <= maxRetries + 1; attempt++) {
      try {
        // Execute with timeout
        const result = await Promise.race([
          handlerConfig.handler(context.command, context),
          this.createTimeout(handlerConfig.options.timeout)
        ]);

        return result;

      } catch (error) {
        lastError = error;
        
        if (attempt <= maxRetries && this.isRetryableError(error)) {
          const delay = this.calculateRetryDelay(attempt);
          logger.warn('⚠️ Command execution failed, retrying', {
            commandType: context.command.type,
            attempt,
            maxRetries,
            delay,
            error: error.message
          });
          
          await this.sleep(delay);
        } else {
          break;
        }
      }
    }

    throw lastError;
  }

  /**
   * Check if error is retryable
   */
  isRetryableError(error) {
    // Don't retry validation errors or authorization errors
    if (error.name === 'ValidationError' || error.name === 'AuthorizationError') {
      return false;
    }

    // Retry network errors, database connection errors, etc.
    return true;
  }

  /**
   * Calculate retry delay with exponential backoff
   */
  calculateRetryDelay(attempt) {
    const baseDelay = 1000; // 1 second
    const maxDelay = 30000; // 30 seconds
    const delay = Math.min(baseDelay * Math.pow(2, attempt - 1), maxDelay);
    
    // Add jitter to prevent thundering herd
    return delay + Math.random() * 1000;
  }

  /**
   * Check idempotency
   */
  async checkIdempotency(command) {
    // In a real implementation, this would check a database or cache
    // For now, return null (no existing result)
    return null;
  }

  /**
   * Store idempotency result
   */
  async storeIdempotencyResult(command, result) {
    // In a real implementation, this would store the result
    // with the command's idempotency key
  }

  /**
   * Create timeout promise
   */
  createTimeout(ms) {
    return new Promise((_, reject) => {
      setTimeout(() => reject(new Error(`Command timeout after ${ms}ms`)), ms);
    });
  }

  /**
   * Sleep utility
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Update metrics
   */
  updateMetrics(processingTime, success) {
    if (success) {
      this.metrics.commandsProcessed++;
    } else {
      this.metrics.commandsFailed++;
    }

    // Update average processing time
    const totalCommands = this.metrics.commandsProcessed + this.metrics.commandsFailed;
    this.metrics.averageProcessingTime = 
      (this.metrics.averageProcessingTime * (totalCommands - 1) + processingTime) / totalCommands;
  }

  /**
   * Get command bus metrics
   */
  getMetrics() {
    return {
      ...this.metrics,
      successRate: this.metrics.commandsProcessed / 
        (this.metrics.commandsProcessed + this.metrics.commandsFailed) * 100,
      registeredHandlers: this.commandHandlers.size,
      middlewareCount: this.middleware.length
    };
  }
}

/**
 * Common middleware functions
 */
class CommandMiddleware {
  /**
   * Logging middleware
   */
  static logging() {
    return async (context, handlerConfig) => {
      logger.debug('📝 Command middleware: Logging', {
        commandType: context.command.type,
        commandId: context.commandId,
        user: context.user?.id
      });
    };
  }

  /**
   * Authentication middleware
   */
  static authentication() {
    return async (context, handlerConfig) => {
      if (!context.user) {
        throw new Error('Authentication required');
      }
      
      logger.debug('🔐 Command middleware: Authentication passed', {
        userId: context.user.id,
        commandType: context.command.type
      });
    };
  }

  /**
   * Rate limiting middleware
   */
  static rateLimit(maxCommands = 100, windowMs = 60000) {
    const userCommandCounts = new Map();
    
    return async (context, handlerConfig) => {
      const userId = context.user?.id || 'anonymous';
      const now = Date.now();
      const windowStart = now - windowMs;
      
      // Clean old entries
      const userCommands = userCommandCounts.get(userId) || [];
      const recentCommands = userCommands.filter(timestamp => timestamp > windowStart);
      
      if (recentCommands.length >= maxCommands) {
        throw new Error(`Rate limit exceeded: ${maxCommands} commands per ${windowMs}ms`);
      }
      
      recentCommands.push(now);
      userCommandCounts.set(userId, recentCommands);
      
      logger.debug('🚦 Command middleware: Rate limit check passed', {
        userId,
        commandCount: recentCommands.length,
        maxCommands
      });
    };
  }

  /**
   * Audit trail middleware
   */
  static auditTrail() {
    return async (context, handlerConfig) => {
      // In a real implementation, this would log to an audit database
      logger.info('📋 Command audit trail', {
        commandId: context.commandId,
        commandType: context.command.type,
        userId: context.user?.id,
        timestamp: context.timestamp,
        payload: context.command.payload
      });
    };
  }
}

module.exports = { CommandBus, CommandMiddleware };
