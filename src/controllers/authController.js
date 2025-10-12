const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const User = require('../models/User');
const { successResponse, errorResponse } = require('../utils/response');
const logger = require('../utils/logger');
const JWT_SECRET = process.env.JWT_SECRET || 'your-secure-key';
const { sendOtpEmail } = require('../utils/email');
const crypto = require('crypto');
const emailService = require('../services/EmailService');

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

  /**
   * Get user profile
   */
  async getUserProfile(req, res) {
    try {
      if (!req.user) {
        return errorResponse(res, 'Authentication required', null, 401);
      }

      const user = await User.findById(req.user._id).select('-password');
      if (!user) {
        return errorResponse(res, 'User not found', null, 404);
      }

      return successResponse(res, 'Profile retrieved successfully', {
        user: {
          id: user._id,
          name: user.name,
          email: user.email,
          phone: user.phone,
          kycStatus: user.kycStatus,
          isActive: user.isActive,
          createdAt: user.createdAt
        }
      }, 200);
    } catch (error) {
      logger.error('Error getting user profile:', error);
      return errorResponse(res, 'Failed to get profile', error, 500);
    }
  }

  /**
   * Update KYC status (for testing/demo)
   */
  async updateKYCStatus(req, res) {
    try {
      if (!req.user) {
        return errorResponse(res, 'Authentication required', null, 401);
      }

      const { status } = req.body;
      const validStatuses = ['PENDING', 'IN_PROGRESS', 'VERIFIED', 'REJECTED'];
      
      if (!validStatuses.includes(status)) {
        return errorResponse(res, 'Invalid KYC status', null, 400);
      }

      // Update user KYC status (assuming User model exists)
      const user = await User.findById(req.user._id);
      if (!user) {
        return errorResponse(res, 'User not found', null, 404);
      }

      user.kycStatus = status;
      await user.save();

      return successResponse(res, 'KYC status updated successfully', {
        kycStatus: user.kycStatus
      }, 200);
    } catch (error) {
      logger.error('Error updating KYC status:', error);
      return errorResponse(res, 'Failed to update KYC status', error, 500);
    }
  }

  /**
   * Update user profile
   */
  async updateUserProfile(req, res) {
    try {
      if (!req.user) {
        return errorResponse(res, 'Authentication required', null, 401);
      }

      const { name, email, phone, pan } = req.body;
      const user = await User.findById(req.user._id);
      
      if (!user) {
        return errorResponse(res, 'User not found', null, 404);
      }

      if (name) user.name = name;
      if (email) user.email = email;
      if (phone) user.phone = phone;
      if (pan) user.pan = pan;

      await user.save();

      return successResponse(res, 'Profile updated successfully', {
        user: {
          id: user._id,
          name: user.name,
          email: user.email,
          phone: user.phone
        }
      }, 200);
    } catch (error) {
      logger.error('Error updating profile:', error);
      return errorResponse(res, 'Failed to update profile', error, 500);
    }
  }

  /**
   * Register new user
   */
  async register(req, res) {
    try {
      const { email, password, name, pan } = req.body;

      // Check if user exists
      const existingUser = await User.findOne({ email });
      if (existingUser) {
        return errorResponse(res, 'User already exists', null, 400);
      }

      // Hash password
      const hashedPassword = await bcrypt.hash(password, 10);

      // Create user
      const user = new User({
        email,
        password: hashedPassword,
        name,
        pan,
        kycStatus: 'PENDING',
        isActive: true
      });

      await user.save();

      // Generate verification token
      const verificationToken = crypto.randomBytes(32).toString('hex');
      user.emailVerificationToken = verificationToken;
      await user.save();

      // Send verification email
      await emailService.sendVerificationEmail(email, name, verificationToken);

      return successResponse(res, 'Registration successful. Please verify your email.', {
        userId: user._id
      }, 201);
    } catch (error) {
      logger.error('Error registering user:', error);
      return errorResponse(res, 'Registration failed', error, 500);
    }
  }

  /**
   * Verify email
   */
  async verifyEmail(req, res) {
    try {
      const { token } = req.query;

      if (!token) {
        return errorResponse(res, 'Verification token required', null, 400);
      }

      const user = await User.findOne({ emailVerificationToken: token });
      if (!user) {
        return errorResponse(res, 'Invalid or expired token', null, 400);
      }

      user.isEmailVerified = true;
      user.emailVerificationToken = undefined;
      await user.save();

      return successResponse(res, 'Email verified successfully', null, 200);
    } catch (error) {
      logger.error('Error verifying email:', error);
      return errorResponse(res, 'Email verification failed', error, 500);
    }
  }

  /**
   * Forgot password
   */
  async forgotPassword(req, res) {
    try {
      const { email } = req.body;

      const user = await User.findOne({ email });
      if (!user) {
        // Don't reveal if user exists
        return successResponse(res, 'If the email exists, a reset link has been sent', null, 200);
      }

      // Generate reset token
      const resetToken = crypto.randomBytes(32).toString('hex');
      user.passwordResetToken = resetToken;
      user.passwordResetExpires = Date.now() + 3600000; // 1 hour
      await user.save();

      // Send reset email
      await emailService.sendPasswordResetEmail(email, user.name, resetToken);

      return successResponse(res, 'Password reset link sent to your email', null, 200);
    } catch (error) {
      logger.error('Error in forgot password:', error);
      return errorResponse(res, 'Failed to process request', error, 500);
    }
  }

  /**
   * Reset password
   */
  async resetPassword(req, res) {
    try {
      const { token, password } = req.body;

      if (!token || !password) {
        return errorResponse(res, 'Token and password required', null, 400);
      }

      const user = await User.findOne({
        passwordResetToken: token,
        passwordResetExpires: { $gt: Date.now() }
      });

      if (!user) {
        return errorResponse(res, 'Invalid or expired reset token', null, 400);
      }

      // Hash new password
      user.password = await bcrypt.hash(password, 10);
      user.passwordResetToken = undefined;
      user.passwordResetExpires = undefined;
      await user.save();

      return successResponse(res, 'Password reset successfully', null, 200);
    } catch (error) {
      logger.error('Error resetting password:', error);
      return errorResponse(res, 'Password reset failed', error, 500);
    }
  }

  // ... (other authentication methods can be restored as needed)
}

module.exports = new AuthController();
