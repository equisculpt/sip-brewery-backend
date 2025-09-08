const redisRateLimiter = require('./redisRateLimiter');

// WhatsApp rate limiter - more restrictive for messaging endpoints
const whatsAppRateLimiter = redisRateLimiter({
  windowMs: 1 * 60 * 1000, // 1 minute
  max: process.env.NODE_ENV === 'test' ? 2 : 10,
  keyGenerator: (req) => req.ip + (req.body?.phoneNumber || req.params?.phoneNumber || ''),
  message: {
    success: false,
    message: 'Too many requests from this IP/phone, please try again later.'
  },
  auditEvent: 'whatsapp_rate_limit_breach'
});

// General API rate limiter
const apiRateLimiter = redisRateLimiter({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100,
  keyGenerator: req => req.ip,
  message: {
    success: false,
    message: 'Too many requests from this IP, please try again later.'
  },
  auditEvent: 'api_rate_limit_breach'
});

module.exports = {
  whatsAppRateLimiter,
  apiRateLimiter
};