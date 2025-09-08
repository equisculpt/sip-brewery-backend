/**
 * 🔐 SECURE JWT CONFIGURATION
 * 
 * Military-grade JWT security configuration
 */

const crypto = require('crypto');

// Validate JWT_SECRET at startup
if (!process.env.JWT_SECRET) {
  console.error('CRITICAL SECURITY ERROR: JWT_SECRET environment variable not set');
  process.exit(1);
}

if (process.env.JWT_SECRET.length < 32) {
  console.error('CRITICAL SECURITY ERROR: JWT_SECRET must be at least 32 characters');
  process.exit(1);
}

// Generate secure random secrets if not provided
const generateSecureSecret = () => {
  return crypto.randomBytes(64).toString('hex');
};

module.exports = {
  secret: process.env.JWT_SECRET,
  expiresIn: process.env.JWT_EXPIRES_IN || '1h',
  refreshExpiresIn: process.env.JWT_REFRESH_EXPIRES_IN || '7d',
  
  // JWT Options
  options: {
    issuer: 'sipbrewery.com',
    audience: 'sipbrewery-users',
    algorithm: 'HS256',
    expiresIn: process.env.JWT_EXPIRES_IN || '1h'
  },
  
  // Refresh token options
  refreshOptions: {
    expiresIn: process.env.JWT_REFRESH_EXPIRES_IN || '7d',
    issuer: 'sipbrewery.com',
    audience: 'sipbrewery-users'
  },
  
  // Security validation
  validateSecret: () => {
    if (!process.env.JWT_SECRET || process.env.JWT_SECRET.length < 32) {
      throw new Error('JWT_SECRET must be at least 32 characters');
    }
    return true;
  },
  
  // Generate new secret (for rotation)
  generateSecret: generateSecureSecret
};