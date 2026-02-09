const jwt = require('jsonwebtoken');
const { User } = require('../models');
const logger = require('../utils/logger');
const crypto = require('crypto');

const JWT_PUBLIC_KEY = process.env.JWT_PUBLIC_KEY?.replace(/\\n/g, '\n');
const JWT_SECRET = process.env.JWT_SECRET;

class UnifiedAuthMiddleware {
  async authenticateToken(req, res, next) {
    try {
      const authHeader = req.headers.authorization;
      const token = authHeader && authHeader.split(' ')[1];

      if (!token) {
        return res.status(401).json({
          success: false,
          message: 'Access token required'
        });
      }

      let decoded;
      try {
        if (JWT_PUBLIC_KEY) {
          decoded = jwt.verify(token, JWT_PUBLIC_KEY, { algorithms: ['RS256'] });
        } else if (JWT_SECRET) {
          decoded = jwt.verify(token, JWT_SECRET, { algorithms: ['HS256'] });
        } else {
          throw new Error('JWT verification key not configured');
        }
      } catch (jwtError) {
        logger.warn('JWT verification failed', { error: jwtError.message });
        return res.status(401).json({
          success: false,
          message: 'Invalid or expired token'
        });
      }

      const jti = decoded.jti;
      if (jti) {
        const isReplayed = await this.checkJTIReplay(jti);
        if (isReplayed) {
          logger.warn('JWT replay attack detected', { jti, userId: decoded.sub });
          return res.status(401).json({
            success: false,
            message: 'Token has already been used'
          });
        }
      }

      const userId = decoded.sub || decoded.userId || decoded.id;
      if (!userId) {
        return res.status(401).json({
          success: false,
          message: 'Invalid token payload'
        });
      }

      const user = await User.findById(userId);
      if (!user) {
        return res.status(401).json({
          success: false,
          message: 'User not found'
        });
      }

      if (user.status === 'SUSPENDED' || user.status === 'BANNED') {
        return res.status(403).json({
          success: false,
          message: 'Account suspended or banned'
        });
      }

      req.user = user;
      req.userId = userId;
      req.token = decoded;

      next();
    } catch (error) {
      logger.error('Authentication error', { error: error.message });
      return res.status(500).json({
        success: false,
        message: 'Authentication failed'
      });
    }
  }

  async checkJTIReplay(jti) {
    return false;
  }

  requireRole(...roles) {
    return (req, res, next) => {
      if (!req.user) {
        return res.status(401).json({
          success: false,
          message: 'Authentication required'
        });
      }

      const userRole = req.user.role || 'USER';
      if (!roles.includes(userRole)) {
        return res.status(403).json({
          success: false,
          message: 'Insufficient permissions'
        });
      }

      next();
    };
  }

  requireKYC(req, res, next) {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: 'Authentication required'
      });
    }

    if (req.user.kycStatus !== 'APPROVED') {
      return res.status(403).json({
        success: false,
        message: 'KYC verification required',
        kycStatus: req.user.kycStatus
      });
    }

    next();
  }

  optionalAuth(req, res, next) {
    const authHeader = req.headers.authorization;
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
      req.user = null;
      req.userId = null;
      return next();
    }

    this.authenticateToken(req, res, next);
  }
}

const unifiedAuth = new UnifiedAuthMiddleware();

module.exports = {
  authenticateToken: unifiedAuth.authenticateToken.bind(unifiedAuth),
  requireRole: unifiedAuth.requireRole.bind(unifiedAuth),
  requireKYC: unifiedAuth.requireKYC.bind(unifiedAuth),
  optionalAuth: unifiedAuth.optionalAuth.bind(unifiedAuth)
};
