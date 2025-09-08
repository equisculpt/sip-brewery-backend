const jwt = require('jsonwebtoken');
const { User } = require('../models');
const logger = require('../utils/logger');
const { errorResponse } = require('../utils/response');

const authenticateToken = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
      return errorResponse(res, 'Authentication required', null, 401);
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key');
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

    if (!user.isActive) {
      return errorResponse(res, 'Account is inactive', null, 401);
    }

    req.user = user;
    req.userId = user._id; // Set userId for compatibility with controllers
    next();
  } catch (error) {
    logger.error('Auth middleware error:', error);
    return errorResponse(res, 'Authentication failed', error.message, 401);
  }
};

const authenticateAdmin = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
      return errorResponse(res, 'Authentication required', null, 401);
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key');
    let user;
    try {
      user = await User.findById(decoded.userId);
    } catch (err) {
      if (err.name === 'CastError') {
        return errorResponse(res, 'User not found', null, 401);
      }
      throw err;
    }

    if (!user) {
      return errorResponse(res, 'User not found', null, 401);
    }

    if (user.role !== 'admin') {
      return errorResponse(res, 'Admin access required', null, 403);
    }

    req.user = user;
    req.userId = user._id; // Set userId for compatibility with controllers
    next();
  } catch (error) {
    logger.error('Admin auth middleware error:', error);
    return errorResponse(res, 'Authentication failed', error.message, 401);
  }
};

module.exports = {
  authenticateToken,
  authenticateAdmin
}; 