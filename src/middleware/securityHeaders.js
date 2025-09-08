/**
 * 🛡️ SECURITY HEADERS MIDDLEWARE
 * 
 * Comprehensive security headers for production deployment
 */

const helmet = require('helmet');

// Content Security Policy
const cspDirectives = {
  defaultSrc: ["'self'"],
  scriptSrc: [
    "'self'",
    "'unsafe-inline'", // Only for development
    "https://cdn.jsdelivr.net",
    "https://unpkg.com"
  ],
  styleSrc: [
    "'self'",
    "'unsafe-inline'",
    "https://fonts.googleapis.com",
    "https://cdn.jsdelivr.net"
  ],
  fontSrc: [
    "'self'",
    "https://fonts.gstatic.com",
    "https://cdn.jsdelivr.net"
  ],
  imgSrc: [
    "'self'",
    "data:",
    "https:",
    "blob:"
  ],
  connectSrc: [
    "'self'",
    "https://api.sipbrewery.com",
    "wss://api.sipbrewery.com"
  ],
  frameSrc: ["'none'"],
  objectSrc: ["'none'"],
  mediaSrc: ["'self'"],
  manifestSrc: ["'self'"],
  workerSrc: ["'self'"],
  upgradeInsecureRequests: process.env.NODE_ENV === 'production' ? [] : null
};

// Helmet configuration
const helmetConfig = {
  contentSecurityPolicy: {
    directives: cspDirectives,
    reportOnly: false
  },
  
  crossOriginEmbedderPolicy: false, // Disable for API compatibility
  
  crossOriginOpenerPolicy: {
    policy: "same-origin"
  },
  
  crossOriginResourcePolicy: {
    policy: "cross-origin"
  },
  
  dnsPrefetchControl: {
    allow: false
  },
  
  frameguard: {
    action: 'deny'
  },
  
  hidePoweredBy: true,
  
  hsts: {
    maxAge: 31536000, // 1 year
    includeSubDomains: true,
    preload: true
  },
  
  ieNoOpen: true,
  
  noSniff: true,
  
  originAgentCluster: true,
  
  permittedCrossDomainPolicies: false,
  
  referrerPolicy: {
    policy: "strict-origin-when-cross-origin"
  },
  
  xssFilter: true
};

// Additional security headers
const additionalHeaders = (req, res, next) => {
  // Remove server information
  res.removeHeader('X-Powered-By');
  res.removeHeader('Server');
  
  // Add custom security headers
  res.setHeader('X-API-Version', '1.0.0');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains; preload');
  res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  
  // API-specific headers
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');
  res.setHeader('Surrogate-Control', 'no-store');
  
  next();
};

// Security headers middleware
const securityHeaders = [
  helmet(helmetConfig),
  additionalHeaders
];

module.exports = {
  securityHeaders,
  helmetConfig,
  additionalHeaders,
  cspDirectives
};