const { User } = require('../models');
const logger = require('../utils/logger');
const { errorResponse } = require('../utils/response');
const { verifyToken } = require('../utils/auth');
const AuthService = require('../services/AuthService');

// Security: Validate JWT public key at startup
if (!process.env.JWT_PUBLIC_KEY) {
  logger.error('CRITICAL SECURITY ERROR: JWT_PUBLIC_KEY environment variable not set');
  process.exit(1); // Fail fast - do not start server without proper JWT configuration
}

const authenticateToken = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
      return errorResponse(res, 'Authentication required', null, 401);
    }

    // Verify with strict RS256 config from utils (issuer/audience/alg enforced)
    const decoded = verifyToken(token);

    // Allow only access tokens for API access
    if (decoded.token_use !== 'access') {
      return errorResponse(res, 'Invalid token type', null, 401);
    }

    // Enforce JTI presence for replay detection capability
    if (!decoded.jti) {
      return errorResponse(res, 'Invalid token (missing jti)', null, 401);
    }

    // Validate session via sid against DB
    if (!decoded.sid) {
      return errorResponse(res, 'Invalid session', null, 401);
    }

    const sessionQuery = `
      SELECT * FROM user_sessions 
      WHERE user_id = $1 AND session_token = $2 AND status = 'ACTIVE' AND expires_at > NOW()
      LIMIT 1
    `;
    const sessionRes = await AuthService.db.query(sessionQuery, [decoded.userId, decoded.sid]);
    if (sessionRes.rows.length === 0) {
      return errorResponse(res, 'Session expired or revoked', null, 401);
    }
    const session = sessionRes.rows[0];
    let user;
    
    try {
      user = await User.findById(decoded.userId);
    } catch (err) {
      // Handle invalid ObjectId (CastError)
      if (err.name === 'CastError') {
        return errorResponse(res, 'User not found', null, 401);
      }
      throw err;
    }

    if (!user) {
      return errorResponse(res, 'User not found', null, 401);
    }

    // Check if user is active
    if (user.status === 'inactive' || user.status === 'suspended') {
      return errorResponse(res, 'Account is inactive', null, 401);
    }

    req.user = user;
    req.tokenClaims = decoded; // expose claims for downstream middleware (e.g., JTI replay guard)
    req.session = {
      id: session.id,
      token: session.session_token,
      last_activity_at: session.last_activity_at
    };

    // Update session activity asynchronously (no await to avoid latency)
    AuthService.db.query(
      'UPDATE user_sessions SET last_activity_at = NOW() WHERE id = $1',
      [session.id]
    ).catch(err => logger.warn('Failed to update session activity', { err: err.message }));
    next();
  } catch (error) {
    if (error.name === 'JsonWebTokenError') {
      return errorResponse(res, 'Invalid token', null, 401);
    }
    if (error.name === 'TokenExpiredError') {
      return errorResponse(res, 'Token expired', null, 401);
    }
    
    logger.error('Authentication error:', error);
    return errorResponse(res, 'Authentication failed', null, 401);
  }
};

// Role-based authorization middleware
const authorizeRoles = (...roles) => {
  return (req, res, next) => {
    if (!req.user) {
      return errorResponse(res, 'Authentication required', null, 401);
    }

    if (!roles.includes(req.user.role)) {
      return errorResponse(res, 'Insufficient permissions', null, 403);
    }

    next();
  };
};

// Admin-only authorization
const requireAdmin = authorizeRoles('admin', 'super_admin');

// Premium user authorization
const requirePremium = (req, res, next) => {
  if (!req.user) {
    return errorResponse(res, 'Authentication required', null, 401);
  }

  if (!['premium', 'admin', 'super_admin'].includes(req.user.subscription)) {
    return errorResponse(res, 'Premium subscription required', null, 403);
  }

  next();
};

module.exports = {
  authenticateToken,
  authorizeRoles,
  requireAdmin,
  requirePremium
};
