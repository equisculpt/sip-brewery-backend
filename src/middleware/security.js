/**
 * Security Middleware
 * Comprehensive security measures for the application
 * @module middleware/security
 */

const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const cors = require('cors');
const logger = require('../utils/logger');

/**
 * Enhanced Rate Limiting Configuration
 */
const createRateLimit = (options = {}) => {
  const defaultOptions = {
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP to 100 requests per windowMs
    message: {
      success: false,
      message: 'Too many requests from this IP, please try again later.',
      error: 'RATE_LIMIT_EXCEEDED'
    },
    standardHeaders: true,
    legacyHeaders: false,
    handler: (req, res) => {
      logger.warn('Rate limit exceeded', {
        ip: req.ip,
        userAgent: req.get('User-Agent'),
        path: req.path,
        method: req.method
      });
      
      res.status(429).json({
        success: false,
        message: 'Too many requests from this IP, please try again later.',
        error: 'RATE_LIMIT_EXCEEDED',
        retryAfter: Math.round(options.windowMs / 1000) || 900
      });
    }
  };

  return rateLimit({ ...defaultOptions, ...options });
};

/**
 * Different rate limits for different endpoints
 */
const rateLimits = {
  // General API rate limit
  general: createRateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 1000 // 1000 requests per 15 minutes
  }),

  // Strict rate limit for authentication endpoints
  auth: createRateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 10, // 10 attempts per 15 minutes
    skipSuccessfulRequests: true
  }),

  // Rate limit for password reset
  passwordReset: createRateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 3 // 3 attempts per hour
  }),

  // Rate limit for file uploads
  upload: createRateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 20 // 20 uploads per 15 minutes
  }),

  // Rate limit for expensive operations
  expensive: createRateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 5 // 5 requests per minute
  })
};

/**
 * Enhanced CORS Configuration
 */
const corsOptions = {
  origin: (origin, callback) => {
    // Allow requests with no origin (mobile apps, Postman, etc.)
    if (!origin) return callback(null, true);

    const allowedOrigins = process.env.ALLOWED_ORIGINS 
      ? process.env.ALLOWED_ORIGINS.split(',').map(o => o.trim())
      : ['http://localhost:3000', 'http://localhost:3001'];

    if (allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      logger.warn('CORS blocked request from unauthorized origin', { origin });
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: [
    'Origin',
    'X-Requested-With',
    'Content-Type',
    'Accept',
    'Authorization',
    'X-API-Key',
    'X-Client-Version'
  ],
  exposedHeaders: ['X-Total-Count', 'X-Page-Count'],
  maxAge: 86400 // 24 hours
};

/**
 * Enhanced Helmet Configuration
 */
const helmetOptions = {
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'", 'https://fonts.googleapis.com'],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", 'data:', 'https:'],
      fontSrc: ["'self'", 'https://fonts.gstatic.com'],
      connectSrc: ["'self'", 'https://api.exchangerate-api.com'],
      frameSrc: ["'none'"],
      objectSrc: ["'none'"],
      mediaSrc: ["'self'"],
      manifestSrc: ["'self'"]
    }
  },
  crossOriginEmbedderPolicy: false, // Disable if causing issues with external APIs
  hsts: {
    maxAge: 31536000, // 1 year
    includeSubDomains: true,
    preload: true
  },
  noSniff: true,
  frameguard: { action: 'deny' },
  xssFilter: true,
  referrerPolicy: { policy: 'same-origin' }
};

/**
 * Request logging middleware with security focus
 */
const securityLogger = (req, res, next) => {
  const startTime = Date.now();
  
  // Log suspicious patterns
  const suspiciousPatterns = [
    /\.\./,  // Directory traversal
    /<script/i,  // XSS attempts
    /union.*select/i,  // SQL injection
    /javascript:/i,  // JavaScript injection
    /%3C/i,  // Encoded script tags
    /\$ne|\$gt|\$lt/i  // NoSQL injection
  ];

  const userAgent = req.get('User-Agent') || '';
  const isSuspicious = suspiciousPatterns.some(pattern => 
    pattern.test(req.url) || 
    pattern.test(JSON.stringify(req.body)) || 
    pattern.test(JSON.stringify(req.query)) ||
    pattern.test(userAgent)
  );

  if (isSuspicious) {
    logger.warn('Suspicious request detected', {
      ip: req.ip,
      method: req.method,
      url: req.url,
      userAgent,
      body: req.body,
      query: req.query,
      headers: req.headers
    });
  }

  // Log request completion
  res.on('finish', () => {
    const duration = Date.now() - startTime;
    const logLevel = res.statusCode >= 400 ? 'warn' : 'info';
    
    logger[logLevel]('Request completed', {
      method: req.method,
      url: req.url,
      statusCode: res.statusCode,
      duration: `${duration}ms`,
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      suspicious: isSuspicious
    });
  });

  next();
};

/**
 * API Key validation middleware
 */
const validateApiKey = (req, res, next) => {
  const apiKey = req.headers['x-api-key'];
  const validApiKeys = process.env.VALID_API_KEYS 
    ? process.env.VALID_API_KEYS.split(',').map(key => key.trim())
    : [];

  // Skip API key validation if none are configured
  if (validApiKeys.length === 0) {
    return next();
  }

  if (!apiKey || !validApiKeys.includes(apiKey)) {
    logger.warn('Invalid API key attempt', {
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      providedKey: apiKey ? `${apiKey.substring(0, 8)}...` : 'none'
    });

    return res.status(401).json({
      success: false,
      message: 'Valid API key required',
      error: 'INVALID_API_KEY'
    });
  }

  next();
};

/**
 * Request size limiter
 */
const requestSizeLimiter = (maxSize = '10mb') => {
  return (req, res, next) => {
    const contentLength = parseInt(req.headers['content-length'] || '0');
    const maxBytes = parseSize(maxSize);

    if (contentLength > maxBytes) {
      logger.warn('Request size exceeded limit', {
        ip: req.ip,
        contentLength,
        maxBytes,
        path: req.path
      });

      return res.status(413).json({
        success: false,
        message: `Request size exceeds limit of ${maxSize}`,
        error: 'REQUEST_TOO_LARGE'
      });
    }

    next();
  };
};

/**
 * Helper function to parse size strings like '10mb', '1gb'
 */
const parseSize = (size) => {
  const units = {
    b: 1,
    kb: 1024,
    mb: 1024 * 1024,
    gb: 1024 * 1024 * 1024
  };

  const match = size.toString().toLowerCase().match(/^(\d+(?:\.\d+)?)(b|kb|mb|gb)?$/);
  if (!match) return 0;

  const value = parseFloat(match[1]);
  const unit = match[2] || 'b';
  
  return Math.floor(value * units[unit]);
};

/**
 * IP whitelist middleware
 */
const ipWhitelist = (allowedIPs = []) => {
  return (req, res, next) => {
    if (allowedIPs.length === 0) return next();

    const clientIP = req.ip || req.connection.remoteAddress;
    
    if (!allowedIPs.includes(clientIP)) {
      logger.warn('IP not in whitelist', {
        ip: clientIP,
        path: req.path,
        method: req.method
      });

      return res.status(403).json({
        success: false,
        message: 'Access denied from this IP address',
        error: 'IP_NOT_WHITELISTED'
      });
    }

    next();
  };
};

/**
 * Security headers middleware
 */
const securityHeaders = (req, res, next) => {
  // Remove server information
  res.removeHeader('X-Powered-By');
  
  // Add custom security headers
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
  res.setHeader('Referrer-Policy', 'same-origin');
  res.setHeader('Permissions-Policy', 'geolocation=(), microphone=(), camera=()');
  
  next();
};

module.exports = {
  rateLimits,
  corsOptions,
  helmetOptions,
  securityLogger,
  validateApiKey,
  requestSizeLimiter,
  ipWhitelist,
  securityHeaders,
  createRateLimit
};
