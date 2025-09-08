/**
 * Response Utility
 * Standardized API response formatting
 */

const logger = require('./logger');

class ResponseUtil {
  /**
   * Send success response
   */
  static success(res, message = 'Success', data = null, statusCode = 200) {
    const response = {
      success: true,
      message,
      data,
      timestamp: new Date().toISOString()
    };

    logger.info(`API Success: ${message}`);
    return res.status(statusCode).json(response);
  }

  /**
   * Send error response
   */
  static error(res, message = 'Internal Server Error', error = null, statusCode = 500) {
    const response = {
      success: false,
      message,
      error: process.env.NODE_ENV === 'development' ? error : null,
      timestamp: new Date().toISOString()
    };

    logger.error(`API Error: ${message}`, error);
    return res.status(statusCode).json(response);
  }

  /**
   * Send validation error response
   */
  static validationError(res, message = 'Validation Error', errors = []) {
    const response = {
      success: false,
      message,
      errors,
      timestamp: new Date().toISOString()
    };

    logger.warn(`Validation Error: ${message}`, errors);
    return res.status(400).json(response);
  }

  /**
   * Send not found response
   */
  static notFound(res, message = 'Resource not found') {
    const response = {
      success: false,
      message,
      timestamp: new Date().toISOString()
    };

    logger.warn(`Not Found: ${message}`);
    return res.status(404).json(response);
  }

  /**
   * Send unauthorized response
   */
  static unauthorized(res, message = 'Unauthorized access') {
    const response = {
      success: false,
      message,
      timestamp: new Date().toISOString()
    };

    logger.warn(`Unauthorized: ${message}`);
    return res.status(401).json(response);
  }

  /**
   * Send forbidden response
   */
  static forbidden(res, message = 'Access forbidden') {
    const response = {
      success: false,
      message,
      timestamp: new Date().toISOString()
    };

    logger.warn(`Forbidden: ${message}`);
    return res.status(403).json(response);
  }
}

module.exports = ResponseUtil;
