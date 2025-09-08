/**
 * Enhanced Input Validation and Sanitization Middleware
 * Provides comprehensive security validation for all inputs
 * @module middleware/validation
 */

const validator = require('validator');
const { body, query, param, validationResult } = require('express-validator');
const logger = require('../utils/logger');
const rateLimit = require('express-rate-limit');

/**
 * XSS Protection - Sanitize HTML content
 */
const sanitizeHtml = (str) => {
  if (typeof str !== 'string') return str;
  return str
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');
};

/**
 * SQL Injection Protection - Sanitize SQL-like patterns
 */
const sanitizeSql = (str) => {
  if (typeof str !== 'string') return str;
  const sqlPatterns = [
    /('|(\-\-)|(;)|(\||\|)|(\*|\*))/gi,
    /(union|select|insert|delete|update|drop|create|alter|exec|execute)/gi
  ];
  
  let sanitized = str;
  sqlPatterns.forEach(pattern => {
    sanitized = sanitized.replace(pattern, '');
  });
  
  return sanitized.trim();
};

/**
 * NoSQL Injection Protection
 */
const sanitizeNoSql = (obj) => {
  if (typeof obj === 'string') {
    return obj.replace(/[${}]/g, '');
  }
  
  if (typeof obj === 'object' && obj !== null) {
    const sanitized = {};
    for (const [key, value] of Object.entries(obj)) {
      if (key.startsWith('$')) {
        logger.warn(`Potential NoSQL injection attempt: ${key}`);
        continue;
      }
      sanitized[key] = sanitizeNoSql(value);
    }
    return sanitized;
  }
  
  return obj;
};

/**
 * Comprehensive Input Sanitization Middleware
 */
const sanitizeInput = (req, res, next) => {
  try {
    // Sanitize body
    if (req.body && typeof req.body === 'object') {
      req.body = sanitizeNoSql(req.body);
      for (const [key, value] of Object.entries(req.body)) {
        if (typeof value === 'string') {
          req.body[key] = sanitizeSql(sanitizeHtml(value));
        }
      }
    }
    
    // Sanitize query parameters
    if (req.query && typeof req.query === 'object') {
      for (const [key, value] of Object.entries(req.query)) {
        if (typeof value === 'string') {
          req.query[key] = sanitizeSql(sanitizeHtml(value));
        }
      }
    }
    
    // Sanitize URL parameters
    if (req.params && typeof req.params === 'object') {
      for (const [key, value] of Object.entries(req.params)) {
        if (typeof value === 'string') {
          req.params[key] = sanitizeSql(sanitizeHtml(value));
        }
      }
    }
    
    next();
  } catch (error) {
    logger.error('Input sanitization error:', error);
    return res.status(400).json({
      success: false,
      message: 'Invalid input format',
      error: 'Input sanitization failed'
    });
  }
};

/**
 * Enhanced Parameter Validation
 */
const validateParams = (requiredParams) => {
  return (req, res, next) => {
    const missing = requiredParams.filter(param => {
      const value = req.params[param];
      return !value || value.trim() === '';
    });
    
    if (missing.length > 0) {
      logger.warn(`Missing required parameters: ${missing.join(', ')}`, {
        ip: req.ip,
        userAgent: req.get('User-Agent')
      });
      
      return res.status(400).json({
        success: false,
        message: 'Missing required parameters',
        missing,
        error: 'VALIDATION_ERROR'
      });
    }
    
    next();
  };
};

/**
 * Enhanced Request Validation with Security Checks
 */
const validateRequest = (schema) => {
  return (req, res, next) => {
    try {
      const errors = [];
      
      // Validate body parameters
      if (schema.body) {
        for (const [field, config] of Object.entries(schema.body)) {
          const value = req.body[field];
          
          // Required field check
          if (config.required && (value === undefined || value === null || value === '')) {
            errors.push({
              field,
              message: `${field} is required`,
              code: 'REQUIRED_FIELD_MISSING'
            });
            continue;
          }
          
          if (value !== undefined && value !== null && value !== '') {
            // Type validation
            if (config.type && !validateType(value, config.type)) {
              errors.push({
                field,
                message: `${field} must be of type ${config.type}`,
                code: 'INVALID_TYPE',
                expected: config.type,
                received: getValueType(value)
              });
            }
            
            // Length validation for strings
            if (config.minLength && typeof value === 'string' && value.length < config.minLength) {
              errors.push({
                field,
                message: `${field} must be at least ${config.minLength} characters long`,
                code: 'MIN_LENGTH_VIOLATION'
              });
            }
            
            if (config.maxLength && typeof value === 'string' && value.length > config.maxLength) {
              errors.push({
                field,
                message: `${field} must not exceed ${config.maxLength} characters`,
                code: 'MAX_LENGTH_VIOLATION'
              });
            }
            
            // Email validation
            if (config.isEmail && !validator.isEmail(value)) {
              errors.push({
                field,
                message: `${field} must be a valid email address`,
                code: 'INVALID_EMAIL'
              });
            }
            
            // URL validation
            if (config.isURL && !validator.isURL(value)) {
              errors.push({
                field,
                message: `${field} must be a valid URL`,
                code: 'INVALID_URL'
              });
            }
            
            // MongoDB ObjectId validation
            if (config.isMongoId && !validator.isMongoId(value)) {
              errors.push({
                field,
                message: `${field} must be a valid MongoDB ObjectId`,
                code: 'INVALID_MONGO_ID'
              });
            }
            
            // Numeric validation
            if (config.isNumeric && !validator.isNumeric(value.toString())) {
              errors.push({
                field,
                message: `${field} must be numeric`,
                code: 'INVALID_NUMERIC'
              });
            }
            
            // Custom validation
            if (config.custom && typeof config.custom === 'function') {
              const customResult = config.custom(value);
              if (customResult !== true) {
                errors.push({
                  field,
                  message: customResult || `${field} failed custom validation`,
                  code: 'CUSTOM_VALIDATION_FAILED'
                });
              }
            }
          }
        }
      }
      
      // Similar validation for query and params...
      if (schema.query) {
        validateSection(req.query, schema.query, 'query', errors);
      }
      
      if (schema.params) {
        validateSection(req.params, schema.params, 'params', errors);
      }
      
      if (errors.length > 0) {
        logger.warn('Validation errors:', {
          errors,
          ip: req.ip,
          userAgent: req.get('User-Agent'),
          path: req.path
        });
        
        return res.status(400).json({
          success: false,
          message: 'Validation failed',
          errors,
          error: 'VALIDATION_ERROR'
        });
      }
      
      next();
    } catch (error) {
      logger.error('Request validation error:', error);
      return res.status(500).json({
        success: false,
        message: 'Validation error occurred',
        error: 'INTERNAL_VALIDATION_ERROR'
      });
    }
  };
};

/**
 * Helper function to validate data type
 */
const validateType = (value, expectedType) => {
  switch (expectedType) {
    case 'string':
      return typeof value === 'string';
    case 'number':
      return typeof value === 'number' && !isNaN(value);
    case 'boolean':
      return typeof value === 'boolean';
    case 'array':
      return Array.isArray(value);
    case 'object':
      return typeof value === 'object' && !Array.isArray(value) && value !== null;
    case 'email':
      return typeof value === 'string' && validator.isEmail(value);
    case 'url':
      return typeof value === 'string' && validator.isURL(value);
    case 'mongoId':
      return typeof value === 'string' && validator.isMongoId(value);
    default:
      return true;
  }
};

/**
 * Helper function to get value type
 */
const getValueType = (value) => {
  if (Array.isArray(value)) return 'array';
  if (value === null) return 'null';
  return typeof value;
};

/**
 * Helper function to validate a section (query, params, etc.)
 */
const validateSection = (data, schema, sectionName, errors) => {
  for (const [field, config] of Object.entries(schema)) {
    const value = data[field];
    
    if (config.required && (value === undefined || value === null || value === '')) {
      errors.push({
        field: `${sectionName}.${field}`,
        message: `${field} is required in ${sectionName}`,
        code: 'REQUIRED_FIELD_MISSING'
      });
    }
    
    if (value && config.type && !validateType(value, config.type)) {
      errors.push({
        field: `${sectionName}.${field}`,
        message: `${field} in ${sectionName} must be of type ${config.type}`,
        code: 'INVALID_TYPE'
      });
    }
  }
};

/**
 * Express-validator error handler
 */
const handleValidationErrors = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    logger.warn('Express-validator errors:', {
      errors: errors.array(),
      ip: req.ip,
      path: req.path
    });
    
    return res.status(400).json({
      success: false,
      message: 'Validation failed',
      errors: errors.array(),
      error: 'VALIDATION_ERROR'
    });
  }
  next();
};

/**
 * Rate limiting for validation failures
 */
const validationRateLimit = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 10, // limit each IP to 10 validation failures per windowMs
  message: {
    success: false,
    message: 'Too many validation failures. Please try again later.',
    error: 'RATE_LIMIT_EXCEEDED'
  },
  standardHeaders: true,
  legacyHeaders: false,
  skip: (req) => {
    // Skip rate limiting for successful requests
    return req.validationPassed === true;
  }
});

/**
 * Common validation schemas
 */
const commonSchemas = {
  mongoId: {
    params: {
      id: { required: true, type: 'mongoId' }
    }
  },
  pagination: {
    query: {
      page: { type: 'number', custom: (value) => value > 0 || 'Page must be greater than 0' },
      limit: { type: 'number', custom: (value) => value > 0 && value <= 100 || 'Limit must be between 1 and 100' }
    }
  },
  userAuth: {
    body: {
      email: { required: true, type: 'email', maxLength: 255 },
      password: { required: true, type: 'string', minLength: 8, maxLength: 128 }
    }
  }
};

module.exports = {
  validateParams,
  validateRequest,
  handleValidationErrors,
  sanitizeInput,
  validationRateLimit,
  commonSchemas,
  // Express-validator helpers
  body,
  query,
  param
};