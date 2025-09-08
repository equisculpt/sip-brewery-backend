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
          pan: req.user.kycDetails?.panNumber,
          riskProfile: req.user.preferences?.riskTolerance,
          investorSince: req.user.createdAt,
          dateOfBirth: req.user.kycDetails?.dateOfBirth,
          address: req.user.kycDetails?.address,
          bankDetails: req.user.kycDetails?.bankDetails
        }
      };

      return successResponse(res, 'KYC status retrieved successfully', kycData);

    } catch (error) {
      logger.error('Error getting KYC status:', error);
      return errorResponse(res, 'Failed to get KYC status', error, 500);
    }
  }

  /**
   * Update KYC status (for testing/demo purposes)
   */
  async updateKYCStatus(req, res) {
    try {
      if (!req.user) {
        return errorResponse(res, 'Authentication required', null, 401);
      }

      const { status, profile } = req.body;

      // Validate status
      const validStatuses = ['PENDING', 'VERIFIED', 'REJECTED'];
      if (status && !validStatuses.includes(status)) {
        return errorResponse(res, 'Invalid KYC status', null, 400);
      }

      // Update user
      const updateData = {};
      if (status) updateData.kycStatus = status;
      if (profile) {
        updateData.name = profile.name || req.user.name;
        updateData.phone = profile.mobile || req.user.phone;
        
        // Update KYC details
        updateData.kycDetails = {
          panNumber: profile.pan || req.user.kycDetails?.panNumber,
          aadharNumber: profile.aadhar || req.user.kycDetails?.aadharNumber,
          dateOfBirth: profile.dateOfBirth || req.user.kycDetails?.dateOfBirth,
          address: profile.address || req.user.kycDetails?.address,
          bankDetails: profile.bankDetails || req.user.kycDetails?.bankDetails
        };
        
        // Update preferences
        if (profile.riskProfile) {
          updateData.preferences = {
            ...req.user.preferences,
            riskTolerance: profile.riskProfile
          };
        }
      }

      const updatedUser = await User.findByIdAndUpdate(
        req.user._id,
        updateData,
        { new: true }
      );

      return successResponse(res, 'KYC status updated successfully', {
        status: updatedUser.kycStatus,
        isCompleted: updatedUser.kycStatus === 'VERIFIED',
        profile: {
          name: updatedUser.name,
          email: updatedUser.email,
          mobile: updatedUser.phone,
          pan: updatedUser.kycDetails?.panNumber,
          riskProfile: updatedUser.preferences?.riskTolerance,
          investorSince: updatedUser.createdAt,
          dateOfBirth: updatedUser.kycDetails?.dateOfBirth,
          address: updatedUser.kycDetails?.address,
          bankDetails: updatedUser.kycDetails?.bankDetails
        }
      });

    } catch (error) {
      logger.error('Error updating KYC status:', error);
      return errorResponse(res, 'Failed to update KYC status', error, 500);
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

      const profile = {
        name: req.user.name,
        email: req.user.email,
        mobile: req.user.phone,
        pan: req.user.kycDetails?.panNumber || null,
        riskProfile: req.user.preferences?.riskTolerance || 'MODERATE',
        investorSince: req.user.createdAt,
        referralCode: req.user.referralCode,
        preferences: { ...(req.user.preferences || {}) },
        profile: req.user.kycDetails || {}
      };

      return successResponse(res, 'User profile retrieved successfully', profile);

    } catch (error) {
      logger.error('Error getting user profile:', error);
      return errorResponse(res, 'Failed to get user profile', error, 500);
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

      const { name, mobile, riskProfile, preferences } = req.body;

      // Validate risk profile
      const validRiskProfiles = ['CONSERVATIVE', 'MODERATE', 'AGGRESSIVE'];
      if (riskProfile && !validRiskProfiles.includes(riskProfile)) {
        return errorResponse(res, 'Invalid risk profile', null, 400);
      }

      // Validate mobile number format (basic validation)
      if (mobile && !/^\d{10,12}$/.test(mobile.replace(/\D/g, ''))) {
        return errorResponse(res, 'Invalid mobile number format', null, 400);
      }

      // Validate field lengths
      if (name && name.length > 100) {
        return errorResponse(res, 'Field too long', null, 400);
      }

      const updateData = {};
      if (name) updateData.name = name;
      if (mobile) updateData.phone = mobile;
      if (riskProfile) {
        updateData.preferences = {
          ...req.user.preferences,
          riskTolerance: riskProfile
        };
      } else if (req.user.preferences && req.user.preferences.riskTolerance) {
        updateData.preferences = {
          ...req.user.preferences,
          riskTolerance: req.user.preferences.riskTolerance
        };
      }
      if (preferences) {
        updateData.preferences = {
          ...req.user.preferences,
          ...updateData.preferences,
          ...preferences
        };
      }

      const updatedUser = await User.findByIdAndUpdate(
        req.user._id,
        updateData,
        { new: true }
      );

      return successResponse(res, 'Profile updated successfully', {
        name: updatedUser.name,
        email: updatedUser.email,
        mobile: updatedUser.phone,
        pan: updatedUser.kycDetails?.panNumber,
        riskProfile: updatedUser.preferences?.riskTolerance || 'MODERATE',
        investorSince: updatedUser.createdAt,
        referralCode: updatedUser.referralCode,
        preferences: updatedUser.preferences
      });

    } catch (error) {
      logger.error('Error updating user profile:', error);
      return errorResponse(res, 'Failed to update profile', error, 500);
    }
  }
}

// Register user with email verification
async function register(req, res) {
  try {
    const { name, email, phone, password } = req.body;
    if (!name || !email || !phone || !password) {
      return res.status(400).json({ success: false, message: 'Name, email, phone, and password are required' });
    }
    let user = await User.findOne({ $or: [{ phone }, { email }] });
    if (user) {
      return res.status(409).json({ success: false, message: 'User already exists', data: { userId: user._id } });
    }
    const hashedPassword = await bcrypt.hash(password, 10);
    const emailToken = crypto.randomBytes(32).toString('hex');
    const emailTokenExpiry = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours
    user = new User({ name, email, phone, password: hashedPassword, emailToken, emailTokenExpiry });
    await user.save();
    await emailService.sendVerificationEmail(email, name, emailToken);
    return res.json({ success: true, message: 'Registered successfully. Please verify your email.' });
  } catch (err) {
    return res.status(500).json({ success: false, message: 'Internal error', error: err.message });
  }
}

// Email verification endpoint
async function verifyEmail(req, res) {
  try {
    const { token } = req.query;
    if (!token) {
      return res.status(400).json({ success: false, message: 'Token is required' });
    }
    const user = await User.findOne({ emailToken: token, emailTokenExpiry: { $gt: Date.now() } });
    if (!user) {
      return res.status(400).json({ success: false, message: 'Invalid or expired token' });
    }
    user.isEmailVerified = true;
    user.emailToken = undefined;
    user.emailTokenExpiry = undefined;
    await user.save();
    return res.json({ success: true, message: 'Email verified successfully' });
  } catch (err) {
    return res.status(500).json({ success: false, message: 'Internal error', error: err.message });
  }
}

// Forgot password endpoint
async function forgotPassword(req, res) {
  try {
    const { email } = req.body;
    if (!email) {
      return res.status(400).json({ success: false, message: 'Email is required' });
    }
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({ success: false, message: 'User not found' });
    }
    const resetPasswordToken = crypto.randomBytes(32).toString('hex');
    const resetPasswordExpiry = new Date(Date.now() + 15 * 60 * 1000); // 15 minutes
    user.resetPasswordToken = resetPasswordToken;
    user.resetPasswordExpiry = resetPasswordExpiry;
    await user.save();
    await emailService.sendPasswordResetEmail(email, user.name, resetPasswordToken);
    return res.json({ success: true, message: 'Password reset link sent to your email.' });
  } catch (err) {
    return res.status(500).json({ success: false, message: 'Internal error', error: err.message });
  }
}

// Reset password using token
async function resetPassword(req, res) {
  try {
    const { token, newPassword } = req.body;
    if (!token || !newPassword) {
      return res.status(400).json({ success: false, message: 'Token and newPassword are required' });
    }
    const user = await User.findOne({ resetPasswordToken: token, resetPasswordExpiry: { $gt: Date.now() } });
    if (!user) {
      return res.status(400).json({ success: false, message: 'Invalid or expired token' });
    }
    user.password = await bcrypt.hash(newPassword, 10);
    user.resetPasswordToken = undefined;
    user.resetPasswordExpiry = undefined;
    await user.save();
    return res.json({ success: true, message: 'Password updated successfully' });
  } catch (err) {
    return res.status(500).json({ success: false, message: 'Internal error', error: err.message });
  }
}

// Login
async function login(req, res) {
  try {
    const { phone, password, method } = req.body;
    if (!phone || !method) {
      return res.status(400).json({ success: false, message: 'Phone and method are required' });
    }
    const user = await User.findOne({ phone });
    if (!user) {
      return res.status(400).json({ success: false, message: 'User not found' });
    }
    if (user.authMode === '1FA') {
      if (method === 'password') {
        if (!user.password) {
          return res.status(400).json({ success: false, message: 'Password login not set for this user' });
        }
        const match = await bcrypt.compare(password, user.password);
        if (!match) {
          return res.status(401).json({ success: false, message: 'Invalid password' });
        }
        const token = issueJwt(user);
        return res.json({ success: true, message: 'Login successful', token, data: { userId: user._id, name: user.name, phone: user.phone } });
      } else if (method === 'otp') {
        // Send OTP
        const otp = generateOtp();
        user.otp = otp;
        user.otpExpiry = new Date(Date.now() + 5 * 60 * 1000);
        await user.save();
        if (user.email) {
          sendOtpEmail({ to: user.email, otp });
        } else {
          console.log(`OTP for ${phone}: ${otp}`);
        }
        return res.json({ success: true, message: 'OTP sent' });
      } else {
        return res.status(400).json({ success: false, message: 'Invalid login method' });
      }
    } else if (user.authMode === '2FA') {
      // Always require password first
      if (!password) {
        return res.status(403).json({ success: false, message: 'Password required for 2FA' });
      }
      const match = await bcrypt.compare(password, user.password);
      if (!match) {
        return res.status(401).json({ success: false, message: 'Invalid password' });
      }
      // Send OTP for 2FA
      const otp = generateOtp();
      user.otp = otp;
      user.otpExpiry = new Date(Date.now() + 5 * 60 * 1000);
      await user.save();
      if (user.email) {
        sendOtpEmail({ to: user.email, otp });
      } else {
        console.log(`OTP for ${phone}: ${otp}`);
      }
      return res.json({ success: true, message: 'OTP sent. Please verify OTP to complete login.' });
    } else {
      return res.status(400).json({ success: false, message: 'Invalid auth mode' });
    }
  } catch (err) {
    return res.status(500).json({ success: false, message: 'Internal error', error: err.message });
  }
}

// Send OTP (for login or password reset)
async function sendOtp(req, res) {
  try {
    const { phone } = req.body;
    if (!phone) {
      return res.status(400).json({ success: false, message: 'Phone is required' });
    }
    let user = await User.findOne({ phone });
    if (!user) {
      return res.status(400).json({ success: false, message: 'User not found. Please register first.' });
    }
    const otp = generateOtp();
    user.otp = otp;
    user.otpExpiry = new Date(Date.now() + 5 * 60 * 1000);
    await user.save();
    if (user.email) {
      sendOtpEmail({ to: user.email, otp });
    } else {
      console.log(`OTP for ${phone}: ${otp}`);
    }
    return res.json({ success: true, message: 'OTP sent' });
  } catch (err) {
    return res.status(500).json({ success: false, message: 'Internal error', error: err.message });
  }
}

// Verify OTP (for login or 2FA)
async function verifyOtp(req, res) {
  try {
    const { phone, otp } = req.body;
    if (!phone || !otp) {
      return res.status(400).json({ success: false, message: 'Phone and OTP are required' });
    }
    const user = await User.findOne({ phone });
    if (!user || !user.otp || !user.otpExpiry) {
      return res.status(401).json({ success: false, message: 'Invalid or expired OTP' });
    }
    if (user.otp !== otp || user.otpExpiry < new Date()) {
      return res.status(401).json({ success: false, message: 'Invalid or expired OTP' });
    }
    user.otp = undefined;
    user.otpExpiry = undefined;
    await user.save();
    // For 1FA: login via OTP
    // For 2FA: login after password+OTP
    const token = issueJwt(user);
    return res.json({
      success: true,
      message: 'OTP verified successfully',
      token,
      user: { id: user._id, name: user.name, phone: user.phone, authMode: user.authMode }
    });
  } catch (err) {
    return res.status(500).json({ success: false, message: 'Internal error', error: err.message });
  }
}

const authController = new AuthController();

module.exports = {
  checkAuth: authController.checkAuth.bind(authController),
  getKYCStatus: authController.getKYCStatus.bind(authController),
  updateKYCStatus: authController.updateKYCStatus.bind(authController),
  getUserProfile: authController.getUserProfile.bind(authController),
  updateUserProfile: authController.updateUserProfile.bind(authController),
  register,
  verifyEmail,
  forgotPassword,
  resetPassword,
  login,
  sendOtp,
  verifyOtp
}; 