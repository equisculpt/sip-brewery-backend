const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const User = require('../models/User');
const { successResponse, errorResponse } = require('../utils/response');
const logger = require('../utils/logger');
const JWT_SECRET = process.env.JWT_SECRET || 'your-secure-key';
const { sendOtpEmail } = require('../utils/email');
const crypto = require('crypto');
const emailService = require('../services/emailService');

// Helper: Generate OTP
function generateOtp() {
  return Math.floor(100000 + Math.random() * 900000).toString();
}

// Helper: Issue JWT
function issueJwt(user) {
  return jwt.sign({ userId: user._id, phone: user.phone, authMode: user.authMode }, JWT_SECRET, { expiresIn: '7d' });
}

class AuthController {
  /**
   * Check authentication status
   */
  async checkAuth(req, res) {
    try {
      if (!req.user) {
        return errorResponse(res, 'Authentication required', null, 401);
      }

      return successResponse(res, 'Authentication successful', {
        userId: req.user._id,
        email: req.user.email,
        name: req.user.name,
        kycStatus: req.user.kycStatus,
        isActive: req.user.isActive
      }, 200);

    } catch (error) {
      logger.error('Error checking auth:', error);
      return errorResponse(res, 'Authentication check failed', error, 500);
    }
  }

  /**
   * Get KYC status
   */
  async getKYCStatus(req, res) {
    try {
      if (!req.user) {
        return errorResponse(res, 'Authentication required', null, 401);
      }

      const kycData = {
        status: req.user.kycStatus,
        isCompleted: req.user.kycStatus === 'VERIFIED',
        profile: {
          name: req.user.name,
          email: req.user.email,
          mobile: req.user.phone,
        }
      };

      return successResponse(res, 'KYC status retrieved', kycData, 200);
    } catch (error) {
      logger.error('Error getting KYC status:', error);
      return errorResponse(res, 'Failed to get KYC status', error, 500);
    }
  }

  // ... (other authentication methods can be restored as needed)
}

module.exports = new AuthController();
