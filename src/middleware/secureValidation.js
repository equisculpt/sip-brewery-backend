/**
 * 🔍 SECURE INPUT VALIDATION MIDDLEWARE
 * 
 * Comprehensive input validation and sanitization
 */

const { body, param, query, validationResult } = require('express-validator');
const DOMPurify = require('isomorphic-dompurify');
const logger = require('../utils/logger');

// Sanitization middleware
const sanitizeInput = (req, res, next) => {
  // Sanitize all string inputs
  const sanitizeObject = (obj) => {
    for (const key in obj) {
      if (typeof obj[key] === 'string') {
        obj[key] = DOMPurify.sanitize(obj[key]);
      } else if (typeof obj[key] === 'object' && obj[key] !== null) {
        sanitizeObject(obj[key]);
      }
    }
  };

  if (req.body) sanitizeObject(req.body);
  if (req.query) sanitizeObject(req.query);
  if (req.params) sanitizeObject(req.params);

  next();
};

// Common validation rules
const commonValidations = {
  email: body('email')
    .isEmail()
    .normalizeEmail()
    .withMessage('Invalid email format'),
    
  password: body('password')
    .isLength({ min: 8, max: 128 })
    .matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*d)(?=.*[@$!%*?&])[A-Za-zd@$!%*?&]/)
    .withMessage('Password must contain at least 8 characters with uppercase, lowercase, number and special character'),
    
  pan: body('pan')
    .matches(/^[A-Z]{5}[0-9]{4}[A-Z]{1}$/)
    .withMessage('Invalid PAN format'),
    
  phone: body('phone')
    .isMobilePhone('en-IN')
    .withMessage('Invalid Indian mobile number'),
    
  fundCode: param('code')
    .isAlphanumeric()
    .isLength({ min: 3, max: 10 })
    .withMessage('Invalid fund code'),
    
  amount: body('amount')
    .isFloat({ min: 1, max: 10000000 })
    .withMessage('Amount must be between 1 and 10,000,000')
};

// Validation error handler
const handleValidationErrors = (req, res, next) => {
  const errors = validationResult(req);
  
  if (!errors.isEmpty()) {
    logger.warn('Validation failed:', {
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      errors: errors.array()
    });
    
    return res.status(400).json({
      success: false,
      message: 'Validation failed',
      errors: errors.array()
    });
  }
  
  next();
};

// SQL Injection prevention
const preventSQLInjection = (req, res, next) => {
  const sqlPatterns = [
    /('|(\')|(;)|(\;)|(--)|(\/\*)|(*\/)|(UNION)|(SELECT)|(INSERT)|(DELETE)|(UPDATE)|(DROP)|(CREATE)|(ALTER)/i
  ];
  
  const checkForSQL = (obj) => {
    for (const key in obj) {
      if (typeof obj[key] === 'string') {
        for (const pattern of sqlPatterns) {
          if (pattern.test(obj[key])) {
            logger.error('SQL injection attempt detected:', {
              ip: req.ip,
              userAgent: req.get('User-Agent'),
              input: obj[key]
            });
            return true;
          }
        }
      } else if (typeof obj[key] === 'object' && obj[key] !== null) {
        if (checkForSQL(obj[key])) return true;
      }
    }
    return false;
  };

  if (req.body && checkForSQL(req.body)) {
    return res.status(400).json({
      success: false,
      message: 'Invalid input detected'
    });
  }

  next();
};

module.exports = {
  sanitizeInput,
  commonValidations,
  handleValidationErrors,
  preventSQLInjection
};