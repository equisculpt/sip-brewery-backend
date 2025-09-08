/**
 * Enterprise Compression Middleware
 * High-performance request/response compression for financial data APIs
 * @module middleware/compression
 */

const compression = require('compression');
const logger = require('../utils/logger');

/**
 * Compression Configuration for Different Content Types
 */
const compressionConfig = {
  // Compression level (1-9, higher = better compression but slower)
  level: process.env.NODE_ENV === 'production' ? 6 : 4,
  
  // Threshold - only compress if response is larger than this (bytes)
  threshold: 1024, // 1KB
  
  // Memory level (1-9, higher = more memory but better compression)
  memLevel: 8,
  
  // Window size (8-15, higher = better compression but more memory)
  windowBits: 15,
  
  // Compression strategy
  strategy: compression.constants.Z_DEFAULT_STRATEGY,
  
  // Chunk size for streaming compression
  chunkSize: 16384, // 16KB chunks
  
  // Filter function to determine what to compress
  filter: (req, res) => {
    // Don't compress if client doesn't support it
    if (!req.headers['accept-encoding']) {
      return false;
    }

    // Don't compress if already compressed
    if (res.getHeader('Content-Encoding')) {
      return false;
    }

    // Don't compress small responses (handled by threshold)
    const contentLength = res.getHeader('Content-Length');
    if (contentLength && parseInt(contentLength) < 1024) {
      return false;
    }

    // Compress specific content types
    const contentType = res.getHeader('Content-Type');
    if (contentType) {
      const type = contentType.toLowerCase();
      
      // Compress JSON API responses (most common for financial APIs)
      if (type.includes('application/json')) {
        return true;
      }
      
      // Compress text-based content
      if (type.includes('text/')) {
        return true;
      }
      
      // Compress XML (for SOAP APIs or data feeds)
      if (type.includes('application/xml') || type.includes('text/xml')) {
        return true;
      }
      
      // Compress CSV data exports
      if (type.includes('text/csv')) {
        return true;
      }
      
      // Don't compress already compressed formats
      if (type.includes('image/') || 
          type.includes('video/') || 
          type.includes('audio/') ||
          type.includes('application/zip') ||
          type.includes('application/gzip')) {
        return false;
      }
    }

    // Default to compression for API routes
    if (req.path.startsWith('/api/')) {
      return true;
    }

    return false;
  }
};

/**
 * Create compression middleware with enterprise configuration
 */
const createCompressionMiddleware = () => {
  return compression({
    level: compressionConfig.level,
    threshold: compressionConfig.threshold,
    memLevel: compressionConfig.memLevel,
    windowBits: compressionConfig.windowBits,
    strategy: compressionConfig.strategy,
    chunkSize: compressionConfig.chunkSize,
    filter: compressionConfig.filter
  });
};

/**
 * Compression analytics middleware
 * Tracks compression performance and savings
 */
const compressionAnalytics = (req, res, next) => {
  const originalWrite = res.write;
  const originalEnd = res.end;
  
  let originalSize = 0;
  let compressedSize = 0;
  const startTime = Date.now();

  // Track original response size
  res.write = function(chunk, encoding) {
    if (chunk) {
      originalSize += Buffer.isBuffer(chunk) ? chunk.length : Buffer.byteLength(chunk, encoding);
    }
    return originalWrite.call(this, chunk, encoding);
  };

  res.end = function(chunk, encoding) {
    if (chunk) {
      originalSize += Buffer.isBuffer(chunk) ? chunk.length : Buffer.byteLength(chunk, encoding);
    }

    // Get compressed size from Content-Length header
    const contentLength = res.getHeader('Content-Length');
    if (contentLength) {
      compressedSize = parseInt(contentLength);
    }

    const endTime = Date.now();
    const compressionTime = endTime - startTime;
    const isCompressed = res.getHeader('Content-Encoding') === 'gzip';

    // Log compression statistics for monitoring
    if (isCompressed && originalSize > 0 && compressedSize > 0) {
      const compressionRatio = ((originalSize - compressedSize) / originalSize * 100).toFixed(2);
      const sizeSaved = originalSize - compressedSize;
      
      logger.debug('Compression Analytics', {
        path: req.path,
        method: req.method,
        originalSize: `${originalSize} bytes`,
        compressedSize: `${compressedSize} bytes`,
        sizeSaved: `${sizeSaved} bytes`,
        compressionRatio: `${compressionRatio}%`,
        compressionTime: `${compressionTime}ms`,
        contentType: res.getHeader('Content-Type')
      });

      // Track significant compression savings
      if (sizeSaved > 10240) { // Log if saved more than 10KB
        logger.info('High compression savings', {
          path: req.path,
          sizeSaved: `${Math.round(sizeSaved / 1024)}KB`,
          compressionRatio: `${compressionRatio}%`
        });
      }
    }

    return originalEnd.call(this, chunk, encoding);
  };

  next();
};

/**
 * Brotli compression middleware (for modern browsers)
 * More efficient than gzip but requires newer browser support
 */
const brotliCompressionMiddleware = (req, res, next) => {
  // Check if client supports Brotli
  const acceptEncoding = req.headers['accept-encoding'] || '';
  
  if (acceptEncoding.includes('br') && process.env.NODE_ENV === 'production') {
    // Set Brotli as preferred encoding for supported clients
    res.setHeader('Vary', 'Accept-Encoding');
    
    // Note: Brotli compression would require additional setup
    // For now, we'll use gzip as it's more universally supported
    logger.debug('Brotli compression available but using gzip for compatibility');
  }
  
  next();
};

/**
 * Response size monitoring middleware
 * Helps identify endpoints that could benefit from optimization
 */
const responseSizeMonitor = (req, res, next) => {
  const originalJson = res.json;
  
  res.json = function(data) {
    const jsonString = JSON.stringify(data);
    const responseSize = Buffer.byteLength(jsonString, 'utf8');
    
    // Log large responses that might need optimization
    if (responseSize > 102400) { // 100KB
      logger.warn('Large API response detected', {
        path: req.path,
        method: req.method,
        responseSize: `${Math.round(responseSize / 1024)}KB`,
        timestamp: new Date().toISOString()
      });
    }
    
    // Track response sizes for analytics
    if (process.env.NODE_ENV === 'development') {
      logger.debug('Response size', {
        path: req.path,
        size: `${responseSize} bytes`
      });
    }
    
    return originalJson.call(this, data);
  };
  
  next();
};

/**
 * Content-Type optimization middleware
 * Ensures proper content types for optimal compression
 */
const contentTypeOptimization = (req, res, next) => {
  const originalJson = res.json;
  
  res.json = function(data) {
    // Ensure proper JSON content type with charset
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    
    // Add cache headers for static-like API responses
    if (req.path.includes('/api/funds/') || req.path.includes('/api/market/')) {
      res.setHeader('Cache-Control', 'public, max-age=300'); // 5 minutes
    }
    
    return originalJson.call(this, data);
  };
  
  next();
};

/**
 * Export compression middleware stack
 */
module.exports = {
  // Main compression middleware
  compression: createCompressionMiddleware(),
  
  // Analytics and monitoring
  compressionAnalytics,
  responseSizeMonitor,
  contentTypeOptimization,
  brotliCompressionMiddleware,
  
  // Configuration
  compressionConfig,
  
  // Complete middleware stack for easy integration
  compressionStack: [
    brotliCompressionMiddleware,
    contentTypeOptimization,
    responseSizeMonitor,
    compressionAnalytics,
    createCompressionMiddleware()
  ]
};
