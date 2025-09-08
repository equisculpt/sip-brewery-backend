/**
 * 🏥 PRODUCTION HEALTH CHECK ENDPOINTS
 * 
 * Comprehensive health monitoring for production deployment
 * Includes readiness, liveness, and detailed system health checks
 */

const express = require('express');
const mongoose = require('mongoose');
const redis = require('redis');
const { unifiedASIService } = require('../services/UnifiedASIService');
const logger = require('../utils/logger');

const router = express.Router();

// Health check cache to avoid overwhelming services
let healthCache = {
  lastCheck: 0,
  result: null,
  ttl: 30000 // 30 seconds
};

/**
 * Basic liveness probe - responds if server is running
 */
router.get('/health', (req, res) => {
  res.status(200).json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    version: process.env.APP_VERSION || '3.0.0',
    environment: process.env.NODE_ENV || 'development'
  });
});

/**
 * Readiness probe - checks if all dependencies are ready
 */
router.get('/ready', async (req, res) => {
  try {
    const now = Date.now();
    
    // Use cached result if still valid
    if (healthCache.lastCheck && (now - healthCache.lastCheck) < healthCache.ttl) {
      return res.status(healthCache.result.status).json(healthCache.result.data);
    }

    const checks = await Promise.allSettled([
      checkDatabase(),
      checkRedis(),
      checkUnifiedASI(),
      checkPythonASI(),
      checkSystemResources()
    ]);

    const results = {
      database: checks[0],
      redis: checks[1],
      unifiedASI: checks[2],
      pythonASI: checks[3],
      system: checks[4]
    };

    const allHealthy = checks.every(check => check.status === 'fulfilled');
    const status = allHealthy ? 200 : 503;

    const response = {
      status: allHealthy ? 'ready' : 'not_ready',
      timestamp: new Date().toISOString(),
      checks: Object.keys(results).reduce((acc, key) => {
        acc[key] = results[key].status === 'fulfilled' 
          ? { status: 'healthy', ...results[key].value }
          : { status: 'unhealthy', error: results[key].reason?.message };
        return acc;
      }, {}),
      summary: {
        total: checks.length,
        healthy: checks.filter(c => c.status === 'fulfilled').length,
        unhealthy: checks.filter(c => c.status === 'rejected').length
      }
    };

    // Cache the result
    healthCache = {
      lastCheck: now,
      result: { status, data: response },
      ttl: 30000
    };

    res.status(status).json(response);

  } catch (error) {
    logger.error('Health check failed:', error);
    res.status(503).json({
      status: 'error',
      timestamp: new Date().toISOString(),
      error: error.message
    });
  }
});

/**
 * Detailed system health with metrics
 */
router.get('/health/detailed', async (req, res) => {
  try {
    const [
      dbHealth,
      redisHealth,
      asiHealth,
      pythonHealth,
      systemHealth,
      serviceMetrics
    ] = await Promise.allSettled([
      checkDatabase(),
      checkRedis(),
      checkUnifiedASI(),
      checkPythonASI(),
      checkSystemResources(),
      getServiceMetrics()
    ]);

    const response = {
      status: 'detailed_health',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      version: process.env.APP_VERSION || '3.0.0',
      environment: process.env.NODE_ENV,
      
      dependencies: {
        database: dbHealth.status === 'fulfilled' ? dbHealth.value : { status: 'error', error: dbHealth.reason?.message },
        redis: redisHealth.status === 'fulfilled' ? redisHealth.value : { status: 'error', error: redisHealth.reason?.message },
        unifiedASI: asiHealth.status === 'fulfilled' ? asiHealth.value : { status: 'error', error: asiHealth.reason?.message },
        pythonASI: pythonHealth.status === 'fulfilled' ? pythonHealth.value : { status: 'error', error: pythonHealth.reason?.message }
      },
      
      system: systemHealth.status === 'fulfilled' ? systemHealth.value : { status: 'error', error: systemHealth.reason?.message },
      
      metrics: serviceMetrics.status === 'fulfilled' ? serviceMetrics.value : { error: serviceMetrics.reason?.message }
    };

    res.status(200).json(response);

  } catch (error) {
    logger.error('Detailed health check failed:', error);
    res.status(500).json({
      status: 'error',
      timestamp: new Date().toISOString(),
      error: error.message
    });
  }
});

/**
 * ASI-specific health endpoint
 */
router.get('/health/asi', async (req, res) => {
  try {
    const asiStatus = await unifiedASIService.getSystemStatus();
    const pythonStatus = await checkPythonASI();

    res.status(200).json({
      status: 'asi_health',
      timestamp: new Date().toISOString(),
      unifiedASI: asiStatus,
      pythonASI: pythonStatus,
      integration: {
        status: asiStatus.initialized && pythonStatus.status === 'healthy' ? 'integrated' : 'partial',
        rating: asiStatus.rating || 'calculating',
        capabilities: asiStatus.capabilities?.financialCapabilities?.length || 0
      }
    });

  } catch (error) {
    logger.error('ASI health check failed:', error);
    res.status(503).json({
      status: 'asi_error',
      timestamp: new Date().toISOString(),
      error: error.message
    });
  }
});

/**
 * Check database connectivity
 */
async function checkDatabase() {
  const start = Date.now();
  
  if (mongoose.connection.readyState !== 1) {
    throw new Error('Database not connected');
  }

  // Test database operation
  await mongoose.connection.db.admin().ping();
  
  return {
    status: 'healthy',
    responseTime: Date.now() - start,
    state: mongoose.connection.readyState,
    host: mongoose.connection.host,
    database: mongoose.connection.name
  };
}

/**
 * Check Redis connectivity
 */
async function checkRedis() {
  const start = Date.now();
  
  try {
    const client = redis.createClient({
      url: process.env.REDIS_URL || 'redis://localhost:6379'
    });
    
    await client.connect();
    await client.ping();
    await client.disconnect();
    
    return {
      status: 'healthy',
      responseTime: Date.now() - start
    };
  } catch (error) {
    throw new Error(`Redis check failed: ${error.message}`);
  }
}

/**
 * Check Unified ASI System
 */
async function checkUnifiedASI() {
  const start = Date.now();
  
  try {
    const health = await unifiedASIService.getHealth();
    
    return {
      status: health.status === 'UNHEALTHY' ? 'unhealthy' : 'healthy',
      responseTime: Date.now() - start,
      initialized: health.service === 'UnifiedASIService',
      rating: health.metrics?.rating || 'calculating',
      requests: health.metrics?.requestCount || 0,
      successRate: health.metrics?.successRate || '0%'
    };
  } catch (error) {
    throw new Error(`Unified ASI check failed: ${error.message}`);
  }
}

/**
 * Check Python ASI Bridge
 */
async function checkPythonASI() {
  const start = Date.now();
  
  try {
    const axios = require('axios');
    const response = await axios.get('http://localhost:8001/health', {
      timeout: 5000
    });
    
    return {
      status: 'healthy',
      responseTime: Date.now() - start,
      pythonVersion: response.data.python_version,
      services: response.data.services || {},
      uptime: response.data.uptime
    };
  } catch (error) {
    throw new Error(`Python ASI check failed: ${error.message}`);
  }
}

/**
 * Check system resources
 */
async function checkSystemResources() {
  const memoryUsage = process.memoryUsage();
  const cpuUsage = process.cpuUsage();
  
  return {
    status: 'healthy',
    memory: {
      rss: Math.round(memoryUsage.rss / 1024 / 1024),
      heapTotal: Math.round(memoryUsage.heapTotal / 1024 / 1024),
      heapUsed: Math.round(memoryUsage.heapUsed / 1024 / 1024),
      external: Math.round(memoryUsage.external / 1024 / 1024),
      usage: Math.round((memoryUsage.heapUsed / memoryUsage.heapTotal) * 100)
    },
    cpu: {
      user: cpuUsage.user,
      system: cpuUsage.system
    },
    uptime: process.uptime(),
    pid: process.pid,
    platform: process.platform,
    nodeVersion: process.version
  };
}

/**
 * Get service metrics
 */
async function getServiceMetrics() {
  try {
    const asiMetrics = unifiedASIService.getMetrics();
    
    return {
      unifiedASI: asiMetrics,
      system: {
        uptime: process.uptime(),
        memoryUsage: process.memoryUsage(),
        cpuUsage: process.cpuUsage()
      }
    };
  } catch (error) {
    throw new Error(`Metrics collection failed: ${error.message}`);
  }
}

module.exports = router;
