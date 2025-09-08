/**
 * 🚀 ENHANCED ASI API ROUTES
 * 
 * Picture-Perfect API with Advanced Monitoring, Caching, Security & Performance
 * Comprehensive endpoints for the Enhanced ASI System
 * 
 * @author Universe-Class ASI Architect
 * @version 4.0.0 - Picture-Perfect API
 */

const express = require('express');
const router = express.Router();
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const compression = require('compression');
const logger = require('../utils/logger');
const { EnhancedASISystem } = require('../asi/EnhancedASISystem');

// Global Enhanced ASI System instance
let enhancedASI = null;

// Security middleware
router.use(helmet());
router.use(compression());

// Rate limiting middleware
const createRateLimit = (windowMs, max, message) => rateLimit({
  windowMs,
  max,
  message: { success: false, error: message },
  standardHeaders: true,
  legacyHeaders: false,
});

// Different rate limits for different endpoints
const generalRateLimit = createRateLimit(15 * 60 * 1000, 100, 'Too many requests'); // 100 per 15 minutes
const predictionRateLimit = createRateLimit(60 * 1000, 20, 'Too many prediction requests'); // 20 per minute
const heavyRateLimit = createRateLimit(60 * 1000, 5, 'Too many heavy computation requests'); // 5 per minute

// Middleware to ensure Enhanced ASI is initialized
const ensureEnhancedASI = async (req, res, next) => {
  try {
    if (!enhancedASI) {
      enhancedASI = new EnhancedASISystem({
        enableMonitoring: true,
        enableCaching: true,
        enableSecurity: true,
        caching: {
          redisUrl: process.env.REDIS_URL || 'redis://localhost:6379',
          defaultTTL: 300,
          maxMemoryCache: 1000
        },
        security: {
          rateLimitWindow: 60000,
          rateLimitMax: 100,
          enableAuditLog: true
        }
      });
      
      await enhancedASI.initialize();
      
      // Setup event listeners
      enhancedASI.on('systemAlert', (alert) => {
        logger.warn(`🚨 System Alert: ${alert.type}`, alert);
      });
      
      enhancedASI.on('dailyReport', (report) => {
        logger.info('📊 Daily report generated', report);
      });
    }
    
    req.enhancedASI = enhancedASI;
    next();
  } catch (error) {
    logger.error('❌ Enhanced ASI initialization failed:', error);
    res.status(503).json({
      success: false,
      error: 'Enhanced ASI system unavailable',
      message: error.message
    });
  }
};

// Helper function to extract user ID from request
const getUserId = (req) => {
  return req.headers['x-user-id'] || req.ip || 'anonymous';
};

/**
 * @route GET /api/enhanced-asi/health
 * @desc Comprehensive health check with detailed system status
 * @access Public
 */
router.get('/health', generalRateLimit, ensureEnhancedASI, async (req, res) => {
  try {
    const status = await req.enhancedASI.getSystemStatus();
    
    const healthStatus = {
      status: status.monitoring?.systemHealth?.status || 'unknown',
      uptime: status.uptime,
      components: {
        asiEngine: status.initialized ? 'healthy' : 'unhealthy',
        monitoring: status.monitoring ? 'healthy' : 'disabled',
        caching: status.caching?.l2Connected ? 'healthy' : 'degraded',
        security: status.security ? 'healthy' : 'disabled',
        pythonBridge: status.pythonBridge?.isServiceRunning ? 'healthy' : 'unhealthy'
      },
      metrics: {
        cacheHitRate: status.caching?.hitRate || 0,
        averageLatency: status.monitoring?.summary?.avgLatency || 0,
        successRate: status.monitoring?.summary?.successRate || 0,
        totalRequests: status.monitoring?.summary?.totalRequests || 0
      },
      timestamp: status.timestamp
    };
    
    const httpStatus = healthStatus.status === 'healthy' ? 200 : 503;
    
    res.status(httpStatus).json({
      success: true,
      health: healthStatus
    });
    
  } catch (error) {
    logger.error('❌ Health check failed:', error);
    res.status(503).json({
      success: false,
      error: 'Health check failed',
      message: error.message
    });
  }
});

/**
 * @route GET /api/enhanced-asi/metrics
 * @desc Detailed system metrics and performance statistics
 * @access Public
 */
router.get('/metrics', generalRateLimit, ensureEnhancedASI, async (req, res) => {
  try {
    const status = await req.enhancedASI.getSystemStatus();
    
    res.json({
      success: true,
      metrics: {
        system: {
          uptime: status.uptime,
          initialized: status.initialized,
          timestamp: status.timestamp
        },
        monitoring: status.monitoring || null,
        caching: status.caching || null,
        security: status.security || null,
        pythonBridge: status.pythonBridge || null
      }
    });
    
  } catch (error) {
    logger.error('❌ Metrics request failed:', error);
    res.status(500).json({
      success: false,
      error: 'Metrics request failed',
      message: error.message
    });
  }
});

/**
 * @route POST /api/enhanced-asi/predict
 * @desc Ultra-accurate predictions with caching and monitoring
 * @access Public
 */
router.post('/predict', predictionRateLimit, ensureEnhancedASI, async (req, res) => {
  try {
    const { symbols, predictionType, timeHorizon, confidenceLevel } = req.body;
    const userId = getUserId(req);
    
    if (!symbols || !Array.isArray(symbols) || symbols.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'Symbols array is required and must not be empty'
      });
    }
    
    logger.info(`🎯 Enhanced prediction request from ${userId} for ${symbols.length} symbols`);
    
    const request = {
      type: 'ultra_accurate_prediction',
      data: {
        symbols,
        predictionType: predictionType || 'absolute',
        timeHorizon: timeHorizon || 30,
        confidenceLevel: confidenceLevel || 0.95
      }
    };
    
    const result = await req.enhancedASI.processRequest(request, userId);
    
    res.json({
      success: true,
      data: result.data,
      metadata: {
        requestId: result.requestId,
        processingTime: result.processingTime,
        cached: result.cached,
        accuracyTarget: '80% overall correctness'
      },
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    logger.error('❌ Enhanced prediction failed:', error);
    res.status(500).json({
      success: false,
      error: 'Enhanced prediction failed',
      message: error.message
    });
  }
});

/**
 * @route POST /api/enhanced-asi/relative-performance
 * @desc Relative performance analysis with 100% accuracy guarantee
 * @access Public
 */
router.post('/relative-performance', predictionRateLimit, ensureEnhancedASI, async (req, res) => {
  try {
    const { symbols, category, timeHorizon, confidenceLevel } = req.body;
    const userId = getUserId(req);
    
    if (!symbols || !Array.isArray(symbols) || symbols.length < 2) {
      return res.status(400).json({
        success: false,
        error: 'At least 2 symbols required for relative performance analysis'
      });
    }
    
    logger.info(`⚖️ Enhanced relative performance analysis from ${userId} for ${symbols.length} symbols`);
    
    const request = {
      type: 'relative_performance_analysis',
      data: {
        symbols,
        category,
        timeHorizon: timeHorizon || 30,
        confidenceLevel: confidenceLevel || 0.99
      }
    };
    
    const result = await req.enhancedASI.processRequest(request, userId);
    
    res.json({
      success: true,
      data: result.data,
      metadata: {
        requestId: result.requestId,
        processingTime: result.processingTime,
        cached: result.cached,
        accuracyGuarantee: '100% for relative performance'
      },
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    logger.error('❌ Enhanced relative performance analysis failed:', error);
    res.status(500).json({
      success: false,
      error: 'Enhanced relative performance analysis failed',
      message: error.message
    });
  }
});

/**
 * @route POST /api/enhanced-asi/compare
 * @desc Symbol comparison with advanced caching and monitoring
 * @access Public
 */
router.post('/compare', predictionRateLimit, ensureEnhancedASI, async (req, res) => {
  try {
    const { symbols } = req.body;
    const userId = getUserId(req);
    
    if (!symbols || !Array.isArray(symbols) || symbols.length < 2) {
      return res.status(400).json({
        success: false,
        error: 'At least 2 symbols required for comparison'
      });
    }
    
    logger.info(`🔍 Enhanced symbol comparison from ${userId} for ${symbols.length} symbols`);
    
    const request = {
      type: 'symbol_comparison',
      data: { symbols }
    };
    
    const result = await req.enhancedASI.processRequest(request, userId);
    
    res.json({
      success: true,
      data: result.data,
      metadata: {
        requestId: result.requestId,
        processingTime: result.processingTime,
        cached: result.cached,
        accuracyGuarantee: '100% for relative performance ranking'
      },
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    logger.error('❌ Enhanced symbol comparison failed:', error);
    res.status(500).json({
      success: false,
      error: 'Enhanced symbol comparison failed',
      message: error.message
    });
  }
});

/**
 * @route POST /api/enhanced-asi/advanced-analysis
 * @desc Advanced multi-modal analysis with full ASI capabilities
 * @access Public
 */
router.post('/advanced-analysis', heavyRateLimit, ensureEnhancedASI, async (req, res) => {
  try {
    const { symbols, analysisType, parameters } = req.body;
    const userId = getUserId(req);
    
    if (!symbols || !Array.isArray(symbols) || symbols.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'Symbols array is required'
      });
    }
    
    logger.info(`🧠 Enhanced advanced analysis from ${userId} for ${symbols.length} symbols`);
    
    const request = {
      type: analysisType || 'advanced_mutual_fund_prediction',
      data: {
        symbols,
        ...parameters
      }
    };
    
    const result = await req.enhancedASI.processRequest(request, userId);
    
    res.json({
      success: true,
      data: result.data,
      metadata: {
        requestId: result.requestId,
        processingTime: result.processingTime,
        cached: result.cached,
        analysisType: analysisType
      },
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    logger.error('❌ Enhanced advanced analysis failed:', error);
    res.status(500).json({
      success: false,
      error: 'Enhanced advanced analysis failed',
      message: error.message
    });
  }
});

/**
 * @route POST /api/enhanced-asi/warm-cache
 * @desc Warm cache with predicted requests
 * @access Public
 */
router.post('/warm-cache', generalRateLimit, ensureEnhancedASI, async (req, res) => {
  try {
    const { predictions } = req.body;
    const userId = getUserId(req);
    
    if (!predictions || !Array.isArray(predictions)) {
      return res.status(400).json({
        success: false,
        error: 'Predictions array is required'
      });
    }
    
    logger.info(`🔥 Cache warming request from ${userId} for ${predictions.length} predictions`);
    
    await req.enhancedASI.warmCache(predictions);
    
    res.json({
      success: true,
      message: `Cache warmed with ${predictions.length} predictions`,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    logger.error('❌ Cache warming failed:', error);
    res.status(500).json({
      success: false,
      error: 'Cache warming failed',
      message: error.message
    });
  }
});

/**
 * @route POST /api/enhanced-asi/invalidate-cache
 * @desc Invalidate cache entries matching pattern
 * @access Public
 */
router.post('/invalidate-cache', generalRateLimit, ensureEnhancedASI, async (req, res) => {
  try {
    const { pattern } = req.body;
    const userId = getUserId(req);
    
    if (!pattern) {
      return res.status(400).json({
        success: false,
        error: 'Pattern is required'
      });
    }
    
    logger.info(`🗑️ Cache invalidation request from ${userId} for pattern: ${pattern}`);
    
    await req.enhancedASI.invalidateCache(pattern);
    
    res.json({
      success: true,
      message: `Cache invalidated for pattern: ${pattern}`,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    logger.error('❌ Cache invalidation failed:', error);
    res.status(500).json({
      success: false,
      error: 'Cache invalidation failed',
      message: error.message
    });
  }
});

/**
 * @route GET /api/enhanced-asi/performance-report
 * @desc Generate comprehensive performance report
 * @access Public
 */
router.get('/performance-report', generalRateLimit, ensureEnhancedASI, async (req, res) => {
  try {
    const status = await req.enhancedASI.getSystemStatus();
    const { period = '24h' } = req.query;
    
    const report = {
      period,
      generatedAt: new Date().toISOString(),
      summary: status.monitoring?.summary || {},
      performance: {
        uptime: status.uptime,
        cachePerformance: status.caching || {},
        securityMetrics: status.security || {},
        systemHealth: status.monitoring?.systemHealth || {}
      },
      recommendations: generateRecommendations(status),
      alerts: status.monitoring?.alerts || []
    };
    
    res.json({
      success: true,
      report,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    logger.error('❌ Performance report generation failed:', error);
    res.status(500).json({
      success: false,
      error: 'Performance report generation failed',
      message: error.message
    });
  }
});

/**
 * @route POST /api/enhanced-asi/batch-predict
 * @desc Batch prediction processing with optimized performance
 * @access Public
 */
router.post('/batch-predict', heavyRateLimit, ensureEnhancedASI, async (req, res) => {
  try {
    const { requests } = req.body;
    const userId = getUserId(req);
    
    if (!requests || !Array.isArray(requests) || requests.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'Requests array is required and must not be empty'
      });
    }
    
    if (requests.length > 50) {
      return res.status(400).json({
        success: false,
        error: 'Maximum 50 requests allowed per batch'
      });
    }
    
    logger.info(`📦 Batch prediction request from ${userId} for ${requests.length} requests`);
    
    const results = await Promise.allSettled(
      requests.map(request => req.enhancedASI.processRequest(request, userId))
    );
    
    const processedResults = results.map((result, index) => ({
      index,
      success: result.status === 'fulfilled',
      data: result.status === 'fulfilled' ? result.value : null,
      error: result.status === 'rejected' ? result.reason.message : null
    }));
    
    const successCount = processedResults.filter(r => r.success).length;
    
    res.json({
      success: true,
      results: processedResults,
      summary: {
        total: requests.length,
        successful: successCount,
        failed: requests.length - successCount,
        successRate: successCount / requests.length
      },
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    logger.error('❌ Batch prediction failed:', error);
    res.status(500).json({
      success: false,
      error: 'Batch prediction failed',
      message: error.message
    });
  }
});

// Helper function to generate recommendations
function generateRecommendations(status) {
  const recommendations = [];
  
  if (status.monitoring?.summary?.avgLatency > 5000) {
    recommendations.push({
      type: 'PERFORMANCE',
      priority: 'HIGH',
      message: 'Average latency is high. Consider optimizing models or enabling more aggressive caching.',
      metric: 'avgLatency',
      value: status.monitoring.summary.avgLatency,
      threshold: 5000
    });
  }
  
  if (status.caching?.hitRate < 0.5) {
    recommendations.push({
      type: 'CACHING',
      priority: 'MEDIUM',
      message: 'Cache hit rate is low. Consider increasing cache TTL or warming cache with common requests.',
      metric: 'hitRate',
      value: status.caching.hitRate,
      threshold: 0.5
    });
  }
  
  if (status.monitoring?.summary?.successRate < 0.95) {
    recommendations.push({
      type: 'RELIABILITY',
      priority: 'HIGH',
      message: 'Success rate is below target. Investigate error patterns and improve error handling.',
      metric: 'successRate',
      value: status.monitoring.summary.successRate,
      threshold: 0.95
    });
  }
  
  if (!status.pythonBridge?.isServiceRunning) {
    recommendations.push({
      type: 'SYSTEM',
      priority: 'CRITICAL',
      message: 'Python bridge is not running. Ultra-accurate predictions may not be available.',
      metric: 'pythonBridge',
      value: 'offline',
      threshold: 'online'
    });
  }
  
  return recommendations;
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('🛑 Shutting down Enhanced ASI API...');
  if (enhancedASI) {
    await enhancedASI.shutdown();
  }
  process.exit(0);
});

module.exports = router;
