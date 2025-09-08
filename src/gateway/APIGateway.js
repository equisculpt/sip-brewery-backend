/**
 * 🌐 ENTERPRISE API GATEWAY
 * 
 * Advanced API Gateway with routing, load balancing, circuit breakers,
 * rate limiting, authentication, and observability
 * 
 * @author Senior AI Backend Developer (35+ years)
 * @version 3.0.0
 */

const express = require('express');
const httpProxy = require('http-proxy-middleware');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const { v4: uuidv4 } = require('uuid');
const logger = require('../utils/logger');
const { AutonomousLearningSystem } = require('../asi/AutonomousLearningSystem');
// Singleton instance for audit (replace with DI if needed)
const autonomousLearningSystem = new AutonomousLearningSystem();
(async () => { try { await autonomousLearningSystem.initialize(); } catch(e) { logger.error('ASI init failed', e); } })();

class CircuitBreaker {
  constructor(options = {}) {
    this.failureThreshold = options.failureThreshold || 5;
    this.resetTimeout = options.resetTimeout || 60000;
    this.monitoringPeriod = options.monitoringPeriod || 10000;
    
    this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
    this.failureCount = 0;
    this.lastFailureTime = null;
    this.successCount = 0;
  }

  async execute(operation) {
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailureTime > this.resetTimeout) {
        this.state = 'HALF_OPEN';
        this.successCount = 0;
      } else {
        throw new Error('Circuit breaker is OPEN');
      }
    }

    try {
      const result = await operation();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  onSuccess() {
    this.failureCount = 0;
    if (this.state === 'HALF_OPEN') {
      this.successCount++;
      if (this.successCount >= 3) {
        this.state = 'CLOSED';
      }
    }
  }

  onFailure() {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    
    if (this.failureCount >= this.failureThreshold) {
      this.state = 'OPEN';
    }
  }

  getState() {
    return {
      state: this.state,
      failureCount: this.failureCount,
      lastFailureTime: this.lastFailureTime
    };
  }
}

class LoadBalancer {
  constructor(targets, strategy = 'round-robin') {
    this.targets = targets;
    this.strategy = strategy;
    this.currentIndex = 0;
    this.healthChecks = new Map();
    
    // Initialize health checks
    this.targets.forEach(target => {
      this.healthChecks.set(target, {
        healthy: true,
        lastCheck: Date.now(),
        consecutiveFailures: 0
      });
    });
    
    // Start health monitoring
    this.startHealthMonitoring();
  }

  getTarget() {
    const healthyTargets = this.targets.filter(target => 
      this.healthChecks.get(target).healthy
    );

    if (healthyTargets.length === 0) {
      throw new Error('No healthy targets available');
    }

    switch (this.strategy) {
      case 'round-robin':
        return this.roundRobin(healthyTargets);
      case 'least-connections':
        return this.leastConnections(healthyTargets);
      case 'weighted':
        return this.weighted(healthyTargets);
      default:
        return this.roundRobin(healthyTargets);
    }
  }

  roundRobin(targets) {
    const target = targets[this.currentIndex % targets.length];
    this.currentIndex++;
    return target;
  }

  leastConnections(targets) {
    // Simplified - would track actual connections in production
    return targets[0];
  }

  weighted(targets) {
    // Simplified - would implement weighted selection
    return targets[0];
  }

  async startHealthMonitoring() {
    setInterval(async () => {
      for (const target of this.targets) {
        try {
          const response = await fetch(`${target}/health`, { 
            timeout: 5000 
          });
          
          if (response.ok) {
            this.healthChecks.get(target).healthy = true;
            this.healthChecks.get(target).consecutiveFailures = 0;
          } else {
            this.markUnhealthy(target);
          }
        } catch (error) {
          this.markUnhealthy(target);
        }
        
        this.healthChecks.get(target).lastCheck = Date.now();
      }
    }, 30000); // Check every 30 seconds
  }

  markUnhealthy(target) {
    const health = this.healthChecks.get(target);
    health.consecutiveFailures++;
    
    if (health.consecutiveFailures >= 3) {
      health.healthy = false;
      logger.warn('Target marked as unhealthy', { target });
    }
  }
}

class APIGateway {
  constructor() {
    this.app = express();
    this.routes = new Map();
    this.circuitBreakers = new Map();
    this.loadBalancers = new Map();
    this.rateLimiters = new Map();
    this.middleware = [];
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      averageResponseTime: 0
    };
  }

  /**
   * Initialize the API Gateway
   */
  async initialize() {
    try {
      // Security middleware
      this.app.use(helmet({
        contentSecurityPolicy: {
          directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            scriptSrc: ["'self'"],
            imgSrc: ["'self'", "data:", "https:"]
          }
        }
      }));

      // CORS
      this.app.use(cors({
        origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
        credentials: true,
        methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
        allowedHeaders: ['Content-Type', 'Authorization', 'X-Correlation-ID']
      }));

      // Request parsing
      this.app.use(express.json({ limit: '10mb' }));
      this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

      // Request ID middleware
      this.app.use((req, res, next) => {
        req.correlationId = req.headers['x-correlation-id'] || uuidv4();
        res.setHeader('X-Correlation-ID', req.correlationId);
        next();
      });

      // Metrics middleware
      this.app.use((req, res, next) => {
        const startTime = Date.now();
        
        res.on('finish', () => {
          const responseTime = Date.now() - startTime;
          this.updateMetrics(responseTime, res.statusCode < 400);
        });
        
        next();
      });

      // Apply custom middleware
      this.middleware.forEach(middleware => {
        this.app.use(middleware);
      });

      // Setup routes
      this.setupRoutes();

      // ASI Audit endpoint (admin only)
      this.app.get('/asi/audit', this.authenticateRequest, async (req, res) => {
        if (!req.user || req.user.role !== 'admin') {
          return res.status(403).json({ error: 'Forbidden', message: 'Admin access required' });
        }
        try {
          const report = await autonomousLearningSystem.auditLearning();
          res.json({ status: 'success', report });
        } catch (err) {
          logger.error('ASI audit endpoint error', err);
          res.status(500).json({ error: 'Internal error', message: err.message });
        }
      });

      // Health check endpoint
      this.app.get('/gateway/health', (req, res) => {
        res.json({
          status: 'healthy',
          timestamp: new Date().toISOString(),
          metrics: this.getMetrics(),
          circuitBreakers: this.getCircuitBreakerStates()
        });
      });

      // Metrics endpoint
      this.app.get('/gateway/metrics', (req, res) => {
        res.json(this.getDetailedMetrics());
      });

      // 404 handler
      this.app.use((req, res) => {
        res.status(404).json({
          error: 'Route not found',
          path: req.path,
          method: req.method,
          correlationId: req.correlationId
        });
      });

      // Error handler
      this.app.use((error, req, res, next) => {
        logger.error('Gateway error', {
          error: error.message,
          path: req.path,
          method: req.method,
          correlationId: req.correlationId
        });

        res.status(500).json({
          error: 'Internal gateway error',
          correlationId: req.correlationId
        });
      });

      logger.info('✅ API Gateway initialized successfully');
    } catch (error) {
      logger.error('❌ Failed to initialize API Gateway:', error);
      throw error;
    }
  }

  /**
   * Register a route with the gateway
   */
  registerRoute(path, config) {
    const {
      targets,
      methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
      authentication = true,
      rateLimit: rateLimitConfig = null,
      circuitBreaker: cbConfig = {},
      loadBalancer: lbConfig = {},
      timeout = 30000,
      retries = 3
    } = config;

    // Setup load balancer
    const loadBalancer = new LoadBalancer(targets, lbConfig.strategy);
    this.loadBalancers.set(path, loadBalancer);

    // Setup circuit breaker
    const circuitBreaker = new CircuitBreaker(cbConfig);
    this.circuitBreakers.set(path, circuitBreaker);

    // Setup rate limiter
    if (rateLimitConfig) {
      const rateLimiter = rateLimit({
        windowMs: rateLimitConfig.windowMs || 60000,
        max: rateLimitConfig.max || 100,
        message: 'Rate limit exceeded',
        standardHeaders: true,
        legacyHeaders: false
      });
      this.rateLimiters.set(path, rateLimiter);
    }

    // Store route configuration
    this.routes.set(path, {
      targets,
      methods,
      authentication,
      timeout,
      retries,
      loadBalancer,
      circuitBreaker
    });

    logger.info('🔗 Route registered', {
      path,
      targets: targets.length,
      methods,
      authentication
    });
  }

  /**
   * Setup routes based on registered configurations
   */
  setupRoutes() {
    for (const [path, config] of this.routes.entries()) {
      const proxyMiddleware = this.createProxyMiddleware(path, config);
      
      // Apply rate limiting if configured
      if (this.rateLimiters.has(path)) {
        this.app.use(path, this.rateLimiters.get(path));
      }

      // Apply authentication if required
      if (config.authentication) {
        this.app.use(path, this.authenticateRequest);
      }

      // Apply proxy middleware
      this.app.use(path, proxyMiddleware);
    }
  }

  /**
   * Create proxy middleware for a route
   */
  createProxyMiddleware(path, config) {
    return httpProxy({
      target: 'http://placeholder', // Will be overridden by router
      changeOrigin: true,
      timeout: config.timeout,
      proxyTimeout: config.timeout,
      
      // Dynamic target selection
      router: async (req) => {
        try {
          const target = await config.circuitBreaker.execute(async () => {
            return config.loadBalancer.getTarget();
          });
          
          logger.debug('🎯 Request routed', {
            path: req.path,
            target,
            correlationId: req.correlationId
          });
          
          return target;
        } catch (error) {
          logger.error('❌ Route selection failed', {
            path: req.path,
            error: error.message,
            correlationId: req.correlationId
          });
          throw error;
        }
      },

      // Request transformation
      onProxyReq: (proxyReq, req, res) => {
        // Add correlation ID
        proxyReq.setHeader('X-Correlation-ID', req.correlationId);
        
        // Add gateway headers
        proxyReq.setHeader('X-Gateway-Version', '3.0.0');
        proxyReq.setHeader('X-Gateway-Timestamp', new Date().toISOString());
        
        // Forward user information
        if (req.user) {
          proxyReq.setHeader('X-User-ID', req.user.id);
          proxyReq.setHeader('X-User-Role', req.user.role);
        }
      },

      // Response transformation
      onProxyRes: (proxyRes, req, res) => {
        // Add gateway headers to response
        proxyRes.headers['X-Gateway-Version'] = '3.0.0';
        proxyRes.headers['X-Correlation-ID'] = req.correlationId;
      },

      // Error handling
      onError: (err, req, res) => {
        logger.error('🚨 Proxy error', {
          error: err.message,
          path: req.path,
          correlationId: req.correlationId
        });

        res.status(502).json({
          error: 'Bad Gateway',
          message: 'Service temporarily unavailable',
          correlationId: req.correlationId
        });
      }
    });
  }

  /**
   * Authentication middleware
   */
  authenticateRequest(req, res, next) {
    const token = req.headers.authorization?.replace('Bearer ', '');
    
    if (!token) {
      return res.status(401).json({
        error: 'Authentication required',
        correlationId: req.correlationId
      });
    }

    try {
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      req.user = decoded;
      next();
    } catch (error) {
      logger.warn('🔒 Authentication failed', {
        error: error.message,
        correlationId: req.correlationId
      });

      res.status(401).json({
        error: 'Invalid token',
        correlationId: req.correlationId
      });
    }
  }

  /**
   * Add custom middleware
   */
  use(middleware) {
    this.middleware.push(middleware);
  }

  /**
   * Update metrics
   */
  updateMetrics(responseTime, success) {
    this.metrics.totalRequests++;
    
    if (success) {
      this.metrics.successfulRequests++;
    } else {
      this.metrics.failedRequests++;
    }

    // Update average response time
    this.metrics.averageResponseTime = 
      (this.metrics.averageResponseTime * (this.metrics.totalRequests - 1) + responseTime) / 
      this.metrics.totalRequests;
  }

  /**
   * Get basic metrics
   */
  getMetrics() {
    return {
      ...this.metrics,
      successRate: (this.metrics.successfulRequests / this.metrics.totalRequests) * 100,
      registeredRoutes: this.routes.size
    };
  }

  /**
   * Get detailed metrics
   */
  getDetailedMetrics() {
    const routeMetrics = {};
    
    for (const [path] of this.routes.entries()) {
      const circuitBreaker = this.circuitBreakers.get(path);
      routeMetrics[path] = {
        circuitBreaker: circuitBreaker.getState(),
        // Add more route-specific metrics here
      };
    }

    return {
      gateway: this.getMetrics(),
      routes: routeMetrics,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Get circuit breaker states
   */
  getCircuitBreakerStates() {
    const states = {};
    
    for (const [path, circuitBreaker] of this.circuitBreakers.entries()) {
      states[path] = circuitBreaker.getState();
    }
    
    return states;
  }

  /**
   * Start the gateway server
   */
  listen(port = 3000) {
    return new Promise((resolve, reject) => {
      const server = this.app.listen(port, (error) => {
        if (error) {
          reject(error);
        } else {
          logger.info(`🌐 API Gateway listening on port ${port}`);
          resolve(server);
        }
      });
    });
  }
}

module.exports = { APIGateway, CircuitBreaker, LoadBalancer };
