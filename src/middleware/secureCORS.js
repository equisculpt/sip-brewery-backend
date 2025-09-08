/**
 * 🌐 SECURE CORS CONFIGURATION
 * 
 * Production-ready CORS configuration with strict security
 */

const cors = require('cors');
const logger = require('../utils/logger');

// Allowed origins (production)
const getAllowedOrigins = () => {
  if (process.env.NODE_ENV === 'production') {
    return process.env.ALLOWED_ORIGINS ? 
      process.env.ALLOWED_ORIGINS.split(',') : 
      ['https://sipbrewery.com', 'https://app.sipbrewery.com'];
  }
  
  // Development origins
  return [
    'http://localhost:3000',
    'http://localhost:3001',
    'http://localhost:8080',
    'http://127.0.0.1:3000'
  ];
};

const corsOptions = {
  origin: (origin, callback) => {
    const allowedOrigins = getAllowedOrigins();
    
    // Allow requests with no origin (mobile apps, etc.)
    if (!origin) return callback(null, true);
    
    if (allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      logger.warn('CORS blocked request from origin:', origin);
      callback(new Error('Not allowed by CORS'));
    }
  },
  
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  
  allowedHeaders: [
    'Origin',
    'X-Requested-With',
    'Content-Type',
    'Accept',
    'Authorization',
    'X-API-Key'
  ],
  
  credentials: true,
  
  // Preflight cache
  maxAge: 86400, // 24 hours
  
  // Security headers
  optionsSuccessStatus: 200
};

// CORS middleware with logging
const secureCORS = cors(corsOptions);

// Enhanced CORS with additional security
const enhancedCORS = (req, res, next) => {
  // Log CORS requests in production
  if (process.env.NODE_ENV === 'production') {
    logger.info('CORS request:', {
      origin: req.get('Origin'),
      method: req.method,
      path: req.path,
      ip: req.ip
    });
  }
  
  secureCORS(req, res, next);
};

module.exports = {
  corsOptions,
  secureCORS,
  enhancedCORS,
  getAllowedOrigins
};