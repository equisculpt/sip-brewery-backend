/**
 * 🚀 ENHANCED ASI SYSTEM
 * 
 * Picture-Perfect ASI with Advanced Monitoring, Caching, Security & Performance
 * Addresses all identified issues and adds next-generation capabilities
 * 
 * @author Universe-Class ASI Architect
 * @version 4.0.0 - Picture-Perfect ASI
 */

const EventEmitter = require('events');
const Redis = require('redis');
const logger = require('../utils/logger');
const { ASIMasterEngine } = require('./ASIMasterEngine');

class ASIMonitoringSystem extends EventEmitter {
  constructor() {
    super();
    this.metrics = {
      requestLatency: new Map(),
      errorRates: new Map(),
      modelAccuracy: new Map(),
      systemHealth: new Map(),
      resourceUsage: new Map()
    };
    
    this.alerts = {
      highLatency: 5000, // 5 seconds
      highErrorRate: 0.1, // 10%
      lowAccuracy: 0.7, // 70%
      highMemoryUsage: 0.8 // 80%
    };
    
    this.startMonitoring();
  }

  startMonitoring() {
    // Monitor system health every 30 seconds
    setInterval(() => {
      this.collectSystemMetrics();
      this.detectAnomalies();
    }, 30000);
    
    // Generate daily reports
    setInterval(() => {
      this.generateDailyReport();
    }, 24 * 60 * 60 * 1000);
  }

  trackRequest(requestId, startTime, endTime, success, accuracy = null) {
    const latency = endTime - startTime;
    
    // Store metrics
    this.metrics.requestLatency.set(requestId, {
      latency,
      timestamp: endTime,
      success,
      accuracy
    });
    
    // Emit events for real-time monitoring
    this.emit('requestCompleted', {
      requestId,
      latency,
      success,
      accuracy
    });
    
    // Check for alerts
    if (latency > this.alerts.highLatency) {
      this.emit('alert', {
        type: 'HIGH_LATENCY',
        requestId,
        latency,
        threshold: this.alerts.highLatency
      });
    }
    
    if (accuracy && accuracy < this.alerts.lowAccuracy) {
      this.emit('alert', {
        type: 'LOW_ACCURACY',
        requestId,
        accuracy,
        threshold: this.alerts.lowAccuracy
      });
    }
  }

  collectSystemMetrics() {
    const memoryUsage = process.memoryUsage();
    const cpuUsage = process.cpuUsage();
    
    this.metrics.resourceUsage.set(Date.now(), {
      memory: {
        heapUsed: memoryUsage.heapUsed,
        heapTotal: memoryUsage.heapTotal,
        external: memoryUsage.external,
        rss: memoryUsage.rss
      },
      cpu: cpuUsage
    });
    
    // Check memory usage
    const memoryUtilization = memoryUsage.heapUsed / memoryUsage.heapTotal;
    if (memoryUtilization > this.alerts.highMemoryUsage) {
      this.emit('alert', {
        type: 'HIGH_MEMORY_USAGE',
        utilization: memoryUtilization,
        threshold: this.alerts.highMemoryUsage
      });
    }
  }

  detectAnomalies() {
    // Analyze recent request patterns
    const recentRequests = Array.from(this.metrics.requestLatency.values())
      .filter(req => Date.now() - req.timestamp < 300000); // Last 5 minutes
    
    if (recentRequests.length === 0) return;
    
    // Calculate error rate
    const errorRate = recentRequests.filter(req => !req.success).length / recentRequests.length;
    if (errorRate > this.alerts.highErrorRate) {
      this.emit('alert', {
        type: 'HIGH_ERROR_RATE',
        errorRate,
        threshold: this.alerts.highErrorRate,
        sampleSize: recentRequests.length
      });
    }
    
    // Calculate average latency
    const avgLatency = recentRequests.reduce((sum, req) => sum + req.latency, 0) / recentRequests.length;
    if (avgLatency > this.alerts.highLatency) {
      this.emit('alert', {
        type: 'HIGH_AVERAGE_LATENCY',
        avgLatency,
        threshold: this.alerts.highLatency,
        sampleSize: recentRequests.length
      });
    }
  }

  generateDailyReport() {
    const report = {
      date: new Date().toISOString().split('T')[0],
      summary: this.generateSummaryMetrics(),
      alerts: this.getAlertsForPeriod(24 * 60 * 60 * 1000),
      recommendations: this.generateRecommendations()
    };
    
    logger.info('📊 Daily ASI Performance Report:', report);
    this.emit('dailyReport', report);
  }

  generateSummaryMetrics() {
    const last24h = Date.now() - (24 * 60 * 60 * 1000);
    const recentRequests = Array.from(this.metrics.requestLatency.values())
      .filter(req => req.timestamp > last24h);
    
    if (recentRequests.length === 0) {
      return { totalRequests: 0 };
    }
    
    const successfulRequests = recentRequests.filter(req => req.success);
    const avgLatency = recentRequests.reduce((sum, req) => sum + req.latency, 0) / recentRequests.length;
    const successRate = successfulRequests.length / recentRequests.length;
    
    const accuracyScores = recentRequests
      .filter(req => req.accuracy !== null)
      .map(req => req.accuracy);
    
    const avgAccuracy = accuracyScores.length > 0 
      ? accuracyScores.reduce((sum, acc) => sum + acc, 0) / accuracyScores.length 
      : null;
    
    return {
      totalRequests: recentRequests.length,
      successRate,
      avgLatency,
      avgAccuracy,
      p95Latency: this.calculatePercentile(recentRequests.map(r => r.latency), 0.95),
      p99Latency: this.calculatePercentile(recentRequests.map(r => r.latency), 0.99)
    };
  }

  calculatePercentile(values, percentile) {
    const sorted = values.sort((a, b) => a - b);
    const index = Math.ceil(sorted.length * percentile) - 1;
    return sorted[index] || 0;
  }

  getAlertsForPeriod(periodMs) {
    // This would return alerts from the last period
    // Implementation depends on how alerts are stored
    return [];
  }

  generateRecommendations() {
    const recommendations = [];
    const summary = this.generateSummaryMetrics();
    
    if (summary.avgLatency > 3000) {
      recommendations.push({
        type: 'PERFORMANCE',
        message: 'Consider enabling caching or optimizing model inference',
        priority: 'HIGH'
      });
    }
    
    if (summary.successRate < 0.95) {
      recommendations.push({
        type: 'RELIABILITY',
        message: 'Investigate error patterns and improve error handling',
        priority: 'HIGH'
      });
    }
    
    if (summary.avgAccuracy && summary.avgAccuracy < 0.8) {
      recommendations.push({
        type: 'ACCURACY',
        message: 'Consider retraining models or updating features',
        priority: 'MEDIUM'
      });
    }
    
    return recommendations;
  }

  getMetrics() {
    return {
      summary: this.generateSummaryMetrics(),
      systemHealth: this.getSystemHealth(),
      alerts: this.getRecentAlerts()
    };
  }

  getSystemHealth() {
    const memoryUsage = process.memoryUsage();
    const memoryUtilization = memoryUsage.heapUsed / memoryUsage.heapTotal;
    
    return {
      status: memoryUtilization < 0.8 ? 'healthy' : 'warning',
      memory: {
        utilization: memoryUtilization,
        heapUsed: memoryUsage.heapUsed,
        heapTotal: memoryUsage.heapTotal
      },
      uptime: process.uptime()
    };
  }

  getRecentAlerts() {
    // Return recent alerts - implementation depends on storage
    return [];
  }
}

class ASICachingSystem {
  constructor(options = {}) {
    this.config = {
      redisUrl: options.redisUrl || 'redis://localhost:6379',
      defaultTTL: options.defaultTTL || 300, // 5 minutes
      maxMemoryCache: options.maxMemoryCache || 1000,
      ...options
    };
    
    // Multi-layer cache
    this.l1Cache = new Map(); // In-memory cache
    this.l2Cache = null; // Redis cache
    this.cacheStats = {
      hits: 0,
      misses: 0,
      sets: 0,
      deletes: 0
    };
    
    this.initializeRedis();
  }

  async initializeRedis() {
    try {
      this.l2Cache = Redis.createClient({ url: this.config.redisUrl });
      await this.l2Cache.connect();
      logger.info('✅ Redis cache connected');
    } catch (error) {
      logger.warn('⚠️ Redis cache not available:', error.message);
    }
  }

  generateCacheKey(type, params) {
    const sortedParams = Object.keys(params)
      .sort()
      .map(key => `${key}:${JSON.stringify(params[key])}`)
      .join('|');
    
    return `asi:${type}:${Buffer.from(sortedParams).toString('base64')}`;
  }

  async get(type, params) {
    const key = this.generateCacheKey(type, params);
    
    // Try L1 cache first
    if (this.l1Cache.has(key)) {
      const cached = this.l1Cache.get(key);
      if (Date.now() < cached.expires) {
        this.cacheStats.hits++;
        return cached.data;
      } else {
        this.l1Cache.delete(key);
      }
    }
    
    // Try L2 cache (Redis)
    if (this.l2Cache) {
      try {
        const cached = await this.l2Cache.get(key);
        if (cached) {
          const data = JSON.parse(cached);
          // Store in L1 cache for faster access
          this.l1Cache.set(key, {
            data,
            expires: Date.now() + (this.config.defaultTTL * 1000)
          });
          this.cacheStats.hits++;
          return data;
        }
      } catch (error) {
        logger.warn('Redis cache get error:', error.message);
      }
    }
    
    this.cacheStats.misses++;
    return null;
  }

  async set(type, params, data, ttl = null) {
    const key = this.generateCacheKey(type, params);
    const actualTTL = ttl || this.config.defaultTTL;
    
    // Store in L1 cache
    if (this.l1Cache.size >= this.config.maxMemoryCache) {
      // Remove oldest entries
      const oldestKey = this.l1Cache.keys().next().value;
      this.l1Cache.delete(oldestKey);
    }
    
    this.l1Cache.set(key, {
      data,
      expires: Date.now() + (actualTTL * 1000)
    });
    
    // Store in L2 cache (Redis)
    if (this.l2Cache) {
      try {
        await this.l2Cache.setEx(key, actualTTL, JSON.stringify(data));
      } catch (error) {
        logger.warn('Redis cache set error:', error.message);
      }
    }
    
    this.cacheStats.sets++;
  }

  async invalidate(pattern) {
    // Invalidate L1 cache
    for (const key of this.l1Cache.keys()) {
      if (key.includes(pattern)) {
        this.l1Cache.delete(key);
        this.cacheStats.deletes++;
      }
    }
    
    // Invalidate L2 cache
    if (this.l2Cache) {
      try {
        const keys = await this.l2Cache.keys(`*${pattern}*`);
        if (keys.length > 0) {
          await this.l2Cache.del(keys);
          this.cacheStats.deletes += keys.length;
        }
      } catch (error) {
        logger.warn('Redis cache invalidation error:', error.message);
      }
    }
  }

  async warmCache(predictions) {
    // Pre-compute and cache likely requests
    for (const prediction of predictions) {
      await this.set('prediction', prediction.params, prediction.result, 600); // 10 minutes
    }
    
    logger.info(`🔥 Cache warmed with ${predictions.length} predictions`);
  }

  getStats() {
    const hitRate = this.cacheStats.hits / (this.cacheStats.hits + this.cacheStats.misses);
    
    return {
      ...this.cacheStats,
      hitRate: isNaN(hitRate) ? 0 : hitRate,
      l1Size: this.l1Cache.size,
      l2Connected: !!this.l2Cache
    };
  }
}

class ASISecuritySystem {
  constructor(options = {}) {
    this.config = {
      rateLimitWindow: options.rateLimitWindow || 60000, // 1 minute
      rateLimitMax: options.rateLimitMax || 100,
      enableAuditLog: options.enableAuditLog !== false,
      ...options
    };
    
    this.rateLimiter = new Map();
    this.auditLog = [];
    this.securityEvents = new Map();
  }

  checkRateLimit(userId, endpoint) {
    const key = `${userId}:${endpoint}`;
    const now = Date.now();
    const windowStart = now - this.config.rateLimitWindow;
    
    if (!this.rateLimiter.has(key)) {
      this.rateLimiter.set(key, []);
    }
    
    const requests = this.rateLimiter.get(key);
    
    // Remove old requests outside the window
    const validRequests = requests.filter(timestamp => timestamp > windowStart);
    
    if (validRequests.length >= this.config.rateLimitMax) {
      this.logSecurityEvent('RATE_LIMIT_EXCEEDED', {
        userId,
        endpoint,
        requestCount: validRequests.length,
        windowStart: new Date(windowStart).toISOString()
      });
      
      return false;
    }
    
    // Add current request
    validRequests.push(now);
    this.rateLimiter.set(key, validRequests);
    
    return true;
  }

  logSecurityEvent(eventType, details) {
    const event = {
      type: eventType,
      timestamp: new Date().toISOString(),
      details
    };
    
    if (this.config.enableAuditLog) {
      this.auditLog.push(event);
      
      // Keep only last 10000 events
      if (this.auditLog.length > 10000) {
        this.auditLog = this.auditLog.slice(-10000);
      }
    }
    
    // Track security event counts
    const count = this.securityEvents.get(eventType) || 0;
    this.securityEvents.set(eventType, count + 1);
    
    logger.warn(`🚨 Security Event: ${eventType}`, details);
  }

  encryptSensitiveData(data) {
    // Simple encryption - in production, use proper encryption
    return Buffer.from(JSON.stringify(data)).toString('base64');
  }

  decryptSensitiveData(encryptedData) {
    try {
      return JSON.parse(Buffer.from(encryptedData, 'base64').toString());
    } catch (error) {
      throw new Error('Failed to decrypt data');
    }
  }

  getSecurityMetrics() {
    return {
      totalAuditEvents: this.auditLog.length,
      securityEventCounts: Object.fromEntries(this.securityEvents),
      activeRateLimits: this.rateLimiter.size
    };
  }

  cleanupOldData() {
    const now = Date.now();
    const oldThreshold = now - (24 * 60 * 60 * 1000); // 24 hours
    
    // Cleanup rate limiter
    for (const [key, requests] of this.rateLimiter.entries()) {
      const validRequests = requests.filter(timestamp => timestamp > oldThreshold);
      if (validRequests.length === 0) {
        this.rateLimiter.delete(key);
      } else {
        this.rateLimiter.set(key, validRequests);
      }
    }
    
    // Cleanup audit log
    this.auditLog = this.auditLog.filter(event => 
      new Date(event.timestamp).getTime() > oldThreshold
    );
  }
}

class EnhancedASISystem extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      enableMonitoring: options.enableMonitoring !== false,
      enableCaching: options.enableCaching !== false,
      enableSecurity: options.enableSecurity !== false,
      ...options
    };
    
    // Core ASI engine
    this.asiEngine = null;
    
    // Enhanced systems
    this.monitoring = null;
    this.caching = null;
    this.security = null;
    
    // System state
    this.isInitialized = false;
    this.startTime = Date.now();
  }

  async initialize() {
    try {
      logger.info('🚀 Initializing Enhanced ASI System...');
      
      // Initialize core ASI engine
      this.asiEngine = new ASIMasterEngine(this.config);
      await this.asiEngine.initialize();
      
      // Initialize enhanced systems
      if (this.config.enableMonitoring) {
        this.monitoring = new ASIMonitoringSystem();
        this.setupMonitoringEvents();
      }
      
      if (this.config.enableCaching) {
        this.caching = new ASICachingSystem(this.config.caching);
      }
      
      if (this.config.enableSecurity) {
        this.security = new ASISecuritySystem(this.config.security);
        this.startSecurityCleanup();
      }
      
      this.isInitialized = true;
      logger.info('✅ Enhanced ASI System initialized successfully');
      
    } catch (error) {
      logger.error('❌ Enhanced ASI System initialization failed:', error);
      throw error;
    }
  }

  setupMonitoringEvents() {
    this.monitoring.on('alert', (alert) => {
      logger.warn(`🚨 ASI Alert: ${alert.type}`, alert);
      this.emit('systemAlert', alert);
    });
    
    this.monitoring.on('dailyReport', (report) => {
      logger.info('📊 Daily ASI Report generated');
      this.emit('dailyReport', report);
    });
  }

  startSecurityCleanup() {
    // Cleanup old security data every hour
    setInterval(() => {
      this.security.cleanupOldData();
    }, 60 * 60 * 1000);
  }

  async processRequest(request, userId = null) {
    const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const startTime = Date.now();
    
    try {
      // Security check
      if (this.security && userId) {
        const endpoint = request.type || 'unknown';
        if (!this.security.checkRateLimit(userId, endpoint)) {
          throw new Error('Rate limit exceeded');
        }
      }
      
      // Check cache first
      let result = null;
      if (this.caching) {
        result = await this.caching.get('request', request);
        if (result) {
          logger.info(`📦 Cache hit for request type: ${request.type}`);
        }
      }
      
      // Process request if not cached
      if (!result) {
        result = await this.asiEngine.processRequest(request);
        
        // Cache the result
        if (this.caching && result.success) {
          await this.caching.set('request', request, result, 300); // 5 minutes
        }
      }
      
      // Track metrics
      const endTime = Date.now();
      if (this.monitoring) {
        const accuracy = this.extractAccuracy(result);
        this.monitoring.trackRequest(requestId, startTime, endTime, result.success, accuracy);
      }
      
      return {
        ...result,
        requestId,
        processingTime: endTime - startTime,
        cached: !!result.cached
      };
      
    } catch (error) {
      const endTime = Date.now();
      
      // Track error metrics
      if (this.monitoring) {
        this.monitoring.trackRequest(requestId, startTime, endTime, false);
      }
      
      // Log security event if relevant
      if (this.security && error.message.includes('Rate limit')) {
        this.security.logSecurityEvent('REQUEST_BLOCKED', {
          requestId,
          userId,
          reason: error.message
        });
      }
      
      logger.error(`❌ Request ${requestId} failed:`, error);
      
      return {
        success: false,
        error: error.message,
        requestId,
        processingTime: endTime - startTime
      };
    }
  }

  extractAccuracy(result) {
    // Extract accuracy from result if available
    if (result.data && typeof result.data.accuracy === 'number') {
      return result.data.accuracy;
    }
    
    if (result.data && result.data.predictions) {
      // For prediction results, try to extract confidence as proxy for accuracy
      const predictions = Array.isArray(result.data.predictions) ? result.data.predictions : [result.data.predictions];
      const confidences = predictions
        .map(p => p.confidence || p.confidenceLevel)
        .filter(c => typeof c === 'number');
      
      if (confidences.length > 0) {
        return confidences.reduce((sum, c) => sum + c, 0) / confidences.length;
      }
    }
    
    return null;
  }

  async getSystemStatus() {
    const status = {
      initialized: this.isInitialized,
      uptime: Date.now() - this.startTime,
      timestamp: new Date().toISOString()
    };
    
    if (this.monitoring) {
      status.monitoring = this.monitoring.getMetrics();
    }
    
    if (this.caching) {
      status.caching = this.caching.getStats();
    }
    
    if (this.security) {
      status.security = this.security.getSecurityMetrics();
    }
    
    if (this.asiEngine && this.asiEngine.pythonBridge) {
      status.pythonBridge = this.asiEngine.pythonBridge.getStatus();
    }
    
    return status;
  }

  async warmCache(predictions) {
    if (this.caching) {
      await this.caching.warmCache(predictions);
    }
  }

  async invalidateCache(pattern) {
    if (this.caching) {
      await this.caching.invalidate(pattern);
    }
  }

  async shutdown() {
    logger.info('🛑 Shutting down Enhanced ASI System...');
    
    if (this.asiEngine && this.asiEngine.pythonBridge) {
      await this.asiEngine.pythonBridge.shutdown();
    }
    
    if (this.caching && this.caching.l2Cache) {
      await this.caching.l2Cache.quit();
    }
    
    logger.info('✅ Enhanced ASI System shutdown completed');
  }
}

module.exports = { 
  EnhancedASISystem,
  ASIMonitoringSystem,
  ASICachingSystem,
  ASISecuritySystem
};
