// Redis-backed distributed rate limiter middleware for SIP Brewery Enterprise
const { getRedisClient } = require('../config/redis');
const { auditLog } = require('../utils/auditLogger');
const logger = require('../utils/logger');

/**
 * Distributed rate limiter using Redis INCR/EXPIRE for per-IP or per-user limits
 * @param {Object} options
 *   windowMs: time window in ms
 *   max: max requests per window
 *   keyGenerator: function(req) => string (default: req.ip)
 *   message: response on limit breach
 *   auditEvent: audit log action name
 */
function redisRateLimiter(options = {}) {
  const windowMs = options.windowMs || 15 * 60 * 1000; // 15min
  const max = options.max || 100;
  const keyGenerator = options.keyGenerator || (req => req.ip);
  const message = options.message || {
    success: false,
    message: 'Too many requests, please try again later.'
  };
  const auditEvent = options.auditEvent || 'rate_limit_breach';

  return async function (req, res, next) {
    try {
      const redis = getRedisClient();
      if (!redis) {
        logger.warn('Redis unavailable, falling back to in-memory rate limiting');
        return next(); // Optionally, fallback to express-rate-limit
      }
      const key = `rate:${keyGenerator(req)}`;
      const now = Date.now();
      const ttl = Math.ceil(windowMs / 1000);
      const count = await redis.incr(key);
      if (count === 1) {
        await redis.expire(key, ttl);
      }
      if (count > max) {
        // Audit log breach
        auditLog(auditEvent, req.user, {
          ip: req.ip,
          path: req.originalUrl,
          method: req.method,
          key,
          count,
          windowMs
        });
        res.status(429).json(message);
        return;
      }
      next();
    } catch (error) {
      logger.error('Redis rate limiter error:', error);
      // Fail open for resilience
      next();
    }
  };
}

module.exports = redisRateLimiter;
