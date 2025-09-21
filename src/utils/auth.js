const jwt = require('jsonwebtoken');
const logger = require('./logger');
const crypto = require('crypto');

// Security configuration (fail-fast, no fallbacks)
const ISSUER = process.env.JWT_ISSUER || 'sip-brewery-backend';
const AUDIENCE = process.env.JWT_AUDIENCE || 'sip-brewery-users';
const ACCESS_TTL = process.env.JWT_ACCESS_TTL || '15m';
const REFRESH_TTL = process.env.JWT_REFRESH_TTL || '7d';

// Require asymmetric keys for RS256
const PRIVATE_KEY = process.env.JWT_PRIVATE_KEY; // PEM string
const PUBLIC_KEY = process.env.JWT_PUBLIC_KEY;   // PEM string

if (!PRIVATE_KEY || !PUBLIC_KEY) {
  // Do not allow server to run with missing keys
  logger.error('CRITICAL SECURITY ERROR: JWT_PRIVATE_KEY / JWT_PUBLIC_KEY not set');
  // Note: allow unit tests to import without crashing; runtime should validate at startup
}

function randomJti() {
  try {
    return crypto.randomUUID();
  } catch {
    return crypto.randomBytes(16).toString('hex');
  }
}

/**
 * Generate JWT token (RS256) with strict claims and jti
 * @param {Object} payload - custom claims (e.g., { userId, role })
 * @param {'access'|'refresh'} tokenUse - token type
 * @param {string|number} [expiresInOverride]
 * @returns {string}
 */
const generateToken = (payload, tokenUse = 'access', expiresInOverride) => {
  try {
    if (!PRIVATE_KEY) throw new Error('JWT private key not configured');
    const exp = expiresInOverride || (tokenUse === 'refresh' ? REFRESH_TTL : ACCESS_TTL);
    const now = Math.floor(Date.now() / 1000);
    const jti = randomJti();
    const fullPayload = {
      ...payload,
      token_use: tokenUse,
      iat: now,
      jti
    };
    return jwt.sign(fullPayload, PRIVATE_KEY, {
      algorithm: 'RS256',
      expiresIn: exp,
      issuer: ISSUER,
      audience: AUDIENCE,
      keyid: process.env.JWT_KID || undefined
    });
  } catch (error) {
    logger.error('Token generation failed', { error: error.message });
    throw new Error('Token generation failed');
  }
};

/**
 * Verify JWT token
 * @param {string} token - JWT token to verify
 * @returns {Object} Decoded token payload
 */
const verifyToken = (token) => {
  try {
    if (!PUBLIC_KEY) throw new Error('JWT public key not configured');
    const decoded = jwt.verify(token, PUBLIC_KEY, {
      algorithms: ['RS256'],
      issuer: ISSUER,
      audience: AUDIENCE,
      clockTolerance: 2 // seconds drift
    });

    // Basic structural checks
    if (!decoded.jti) throw new Error('Missing jti');
    if (!decoded.token_use) throw new Error('Missing token_use');
    if (!decoded.iat) throw new Error('Missing iat');
    return decoded;
  } catch (error) {
    logger.warn('Token verification failed', { error: error.message });
    throw new Error('Invalid token');
  }
};

/**
 * Decode JWT token without verification (for testing)
 * @param {string} token - JWT token to decode
 * @returns {Object} Decoded token payload
 */
const decodeToken = (token) => {
  try {
    return jwt.decode(token);
  } catch (error) {
    logger.error('Token decoding failed', { error: error.message });
    throw new Error('Token decoding failed');
  }
};

module.exports = {
  generateToken,
  verifyToken,
  decodeToken
};