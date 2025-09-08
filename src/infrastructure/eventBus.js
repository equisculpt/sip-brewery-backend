/**
 * 🚀 ENTERPRISE EVENT-DRIVEN ARCHITECTURE
 * 
 * High-performance event bus with Redis Streams for microservices communication
 * Supports event sourcing, CQRS, and distributed system patterns
 * 
 * @author Senior AI Backend Developer (35+ years)
 * @version 3.0.0
 */

const Redis = require('ioredis');
const { v4: uuidv4 } = require('uuid');
const logger = require('../utils/logger');

class EnterpriseEventBus {
  constructor() {
    this.redis = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: process.env.REDIS_PORT || 6379,
      password: process.env.REDIS_PASSWORD,
      retryDelayOnFailover: 100,
      enableReadyCheck: false,
      maxRetriesPerRequest: null,
      lazyConnect: true
    });

    this.subscribers = new Map();
    this.eventHandlers = new Map();
    this.deadLetterQueue = 'events:dead-letter';
    this.retryQueue = 'events:retry';
    this.maxRetries = 3;
    this.isInitialized = false;
  }

  /**
   * Initialize the event bus
   */
  async initialize() {
    try {
      await this.redis.connect();
      await this.setupConsumerGroups();
      await this.startEventProcessing();
      this.isInitialized = true;
      logger.info('✅ Enterprise Event Bus initialized successfully');
    } catch (error) {
      logger.error('❌ Failed to initialize Event Bus:', error);
      throw error;
    }
  }

  /**
   * Setup consumer groups for different event types
   */
  async setupConsumerGroups() {
    const eventStreams = [
      'portfolio:events',
      'market:events', 
      'user:events',
      'transaction:events',
      'notification:events',
      'ai:events'
    ];

    for (const stream of eventStreams) {
      try {
        await this.redis.xgroup('CREATE', stream, 'processors', '$', 'MKSTREAM');
        logger.info(`✅ Created consumer group for ${stream}`);
      } catch (error) {
        if (!error.message.includes('BUSYGROUP')) {
          logger.warn(`⚠️ Consumer group setup warning for ${stream}:`, error.message);
        }
      }
    }
  }

  /**
   * Publish event to the event bus
   */
  async publish(eventType, eventData, options = {}) {
    try {
      const event = {
        id: uuidv4(),
        type: eventType,
        data: eventData,
        timestamp: new Date().toISOString(),
        source: options.source || 'sip-brewery-backend',
        version: options.version || '1.0',
        correlationId: options.correlationId || uuidv4(),
        metadata: options.metadata || {}
      };

      const streamKey = this.getStreamKey(eventType);
      const eventId = await this.redis.xadd(
        streamKey,
        '*',
        'event', JSON.stringify(event)
      );

      logger.info('📤 Event published', {
        eventType,
        eventId,
        streamKey,
        correlationId: event.correlationId
      });

      // Emit to local subscribers immediately
      this.emitToLocalSubscribers(eventType, event);

      return { eventId, correlationId: event.correlationId };
    } catch (error) {
      logger.error('❌ Failed to publish event:', error);
      throw error;
    }
  }

  /**
   * Subscribe to events
   */
  subscribe(eventPattern, handler, options = {}) {
    const subscriberId = uuidv4();
    const subscription = {
      id: subscriberId,
      pattern: eventPattern,
      handler,
      options: {
        retryCount: options.retryCount || this.maxRetries,
        timeout: options.timeout || 30000,
        batchSize: options.batchSize || 10
      }
    };

    if (!this.subscribers.has(eventPattern)) {
      this.subscribers.set(eventPattern, new Set());
    }
    this.subscribers.get(eventPattern).add(subscription);

    logger.info('📥 Event subscription created', {
      eventPattern,
      subscriberId,
      options: subscription.options
    });

    return subscriberId;
  }

  /**
   * Unsubscribe from events
   */
  unsubscribe(subscriberId) {
    for (const [pattern, subscribers] of this.subscribers.entries()) {
      for (const subscription of subscribers) {
        if (subscription.id === subscriberId) {
          subscribers.delete(subscription);
          logger.info('📤 Event subscription removed', { subscriberId, pattern });
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Start processing events from Redis Streams
   */
  async startEventProcessing() {
    const streams = Array.from(this.getEventStreams());
    
    // Process each stream concurrently
    const processors = streams.map(stream => this.processStream(stream));
    await Promise.all(processors);
  }

  /**
   * Process events from a specific stream
   */
  async processStream(streamKey) {
    const consumerName = `consumer-${process.pid}-${Date.now()}`;
    
    while (this.isInitialized) {
      try {
        const results = await this.redis.xreadgroup(
          'GROUP', 'processors', consumerName,
          'COUNT', 10,
          'BLOCK', 1000,
          'STREAMS', streamKey, '>'
        );

        if (results && results.length > 0) {
          for (const [stream, messages] of results) {
            await this.processMessages(stream, messages, consumerName);
          }
        }
      } catch (error) {
        logger.error(`❌ Stream processing error for ${streamKey}:`, error);
        await this.sleep(5000); // Wait before retrying
      }
    }
  }

  /**
   * Process messages from stream
   */
  async processMessages(streamKey, messages, consumerName) {
    for (const [messageId, fields] of messages) {
      try {
        const eventData = JSON.parse(fields[1]); // fields[0] is 'event', fields[1] is data
        await this.handleEvent(eventData);
        
        // Acknowledge successful processing
        await this.redis.xack('processors', streamKey, messageId);
        
        logger.debug('✅ Event processed successfully', {
          messageId,
          eventType: eventData.type,
          streamKey
        });
      } catch (error) {
        logger.error('❌ Event processing failed', {
          messageId,
          streamKey,
          error: error.message
        });
        
        await this.handleFailedEvent(streamKey, messageId, fields, error);
      }
    }
  }

  /**
   * Handle individual events
   */
  async handleEvent(event) {
    const matchingSubscribers = this.findMatchingSubscribers(event.type);
    
    const processingPromises = matchingSubscribers.map(async (subscription) => {
      try {
        await Promise.race([
          subscription.handler(event),
          this.timeout(subscription.options.timeout)
        ]);
      } catch (error) {
        logger.error('❌ Event handler failed', {
          eventType: event.type,
          subscriberId: subscription.id,
          error: error.message
        });
        throw error;
      }
    });

    await Promise.allSettled(processingPromises);
  }

  /**
   * Handle failed events with retry logic
   */
  async handleFailedEvent(streamKey, messageId, fields, error) {
    const eventData = JSON.parse(fields[1]);
    const retryCount = (eventData.retryCount || 0) + 1;

    if (retryCount <= this.maxRetries) {
      // Add to retry queue
      eventData.retryCount = retryCount;
      eventData.lastError = error.message;
      eventData.retryAt = new Date(Date.now() + (retryCount * 5000)).toISOString();

      await this.redis.xadd(
        this.retryQueue,
        '*',
        'event', JSON.stringify(eventData),
        'originalStream', streamKey,
        'originalMessageId', messageId
      );

      logger.warn('⚠️ Event queued for retry', {
        eventType: eventData.type,
        retryCount,
        maxRetries: this.maxRetries
      });
    } else {
      // Send to dead letter queue
      await this.redis.xadd(
        this.deadLetterQueue,
        '*',
        'event', JSON.stringify(eventData),
        'originalStream', streamKey,
        'originalMessageId', messageId,
        'finalError', error.message
      );

      logger.error('💀 Event sent to dead letter queue', {
        eventType: eventData.type,
        retryCount,
        error: error.message
      });
    }

    // Acknowledge the failed message
    await this.redis.xack('processors', streamKey, messageId);
  }

  /**
   * Find subscribers matching event pattern
   */
  findMatchingSubscribers(eventType) {
    const matchingSubscribers = [];
    
    for (const [pattern, subscribers] of this.subscribers.entries()) {
      if (this.matchesPattern(eventType, pattern)) {
        matchingSubscribers.push(...Array.from(subscribers));
      }
    }
    
    return matchingSubscribers;
  }

  /**
   * Check if event type matches pattern
   */
  matchesPattern(eventType, pattern) {
    // Support wildcards: portfolio.* matches portfolio.created, portfolio.updated
    const regexPattern = pattern.replace(/\*/g, '.*');
    const regex = new RegExp(`^${regexPattern}$`);
    return regex.test(eventType);
  }

  /**
   * Emit to local subscribers (for immediate processing)
   */
  emitToLocalSubscribers(eventType, event) {
    const matchingSubscribers = this.findMatchingSubscribers(eventType);
    
    // Process locally without waiting
    setImmediate(async () => {
      for (const subscription of matchingSubscribers) {
        try {
          await subscription.handler(event);
        } catch (error) {
          logger.error('❌ Local event handler failed', {
            eventType,
            subscriberId: subscription.id,
            error: error.message
          });
        }
      }
    });
  }

  /**
   * Get stream key for event type
   */
  getStreamKey(eventType) {
    const [domain] = eventType.split('.');
    return `${domain}:events`;
  }

  /**
   * Get all event streams
   */
  getEventStreams() {
    return new Set([
      'portfolio:events',
      'market:events',
      'user:events', 
      'transaction:events',
      'notification:events',
      'ai:events'
    ]);
  }

  /**
   * Utility methods
   */
  async timeout(ms) {
    return new Promise((_, reject) => 
      setTimeout(() => reject(new Error('Event handler timeout')), ms)
    );
  }

  async sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get event bus statistics
   */
  async getStatistics() {
    const streams = Array.from(this.getEventStreams());
    const stats = {};

    for (const stream of streams) {
      try {
        const info = await this.redis.xinfo('STREAM', stream);
        stats[stream] = {
          length: info[1],
          groups: info[5],
          lastGeneratedId: info[3]
        };
      } catch (error) {
        stats[stream] = { error: error.message };
      }
    }

    return {
      streams: stats,
      subscribers: this.subscribers.size,
      deadLetterQueue: await this.redis.xlen(this.deadLetterQueue),
      retryQueue: await this.redis.xlen(this.retryQueue)
    };
  }

  /**
   * Graceful shutdown
   */
  async shutdown() {
    logger.info('🔄 Shutting down Event Bus...');
    this.isInitialized = false;
    await this.redis.quit();
    logger.info('✅ Event Bus shutdown complete');
  }
}

module.exports = { EnterpriseEventBus };
