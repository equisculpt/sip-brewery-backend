const express = require('express');
const router = express.Router();
const authController = require('../controllers/authController');
const { authenticateToken } = require('../middleware/auth');
const { apiRateLimiter } = require('../middleware/rateLimiter');
const rateLimit = require('express-rate-limit');

// Strict rate limiting for authentication endpoints
const authRateLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // Limit each IP to 5 requests per windowMs
  message: {
    success: false,
    message: 'Too many authentication attempts, please try again later.'
  },
  standardHeaders: true,
  legacyHeaders: false
});

/**
 * @route   GET /api/auth/check
 * @desc    Check authentication status
 * @access  Private
 */
router.get('/check',
  authenticateToken,
  authController.checkAuth
);

/**
 * @route   GET /api/auth/kyc/status
 * @desc    Get KYC status
 * @access  Private
 */
router.get('/kyc/status',
  authenticateToken,
  authController.getKYCStatus
);

/**
 * @route   PUT /api/auth/kyc/status
 * @desc    Update KYC status (for testing/demo)
 * @access  Private
 */
router.put('/kyc/status',
  authenticateToken,
  authController.updateKYCStatus
);

/**
 * @route   GET /api/auth/profile
 * @desc    Get user profile
 * @access  Private
 */
router.get('/profile',
  authenticateToken,
  authController.getUserProfile
);

const { body, validationResult } = require('express-validator');
const { handleValidationErrors } = require('../middleware/validation');

// Profile update validation
const validateProfileUpdate = [
  body('email').optional().isEmail().withMessage('Invalid email'),
  body('phone').optional().isMobilePhone('en-IN').withMessage('Invalid phone'),
  body('name').optional().isLength({ min: 2 }).withMessage('Name must be at least 2 characters'),
  body('pan').optional().matches(/[A-Z]{5}[0-9]{4}[A-Z]{1}/).withMessage('Invalid PAN'),
  handleValidationErrors
];

/**
 * @route   PUT /api/auth/profile
 * @desc    Update user profile
 * @access  Private
 */
router.put('/profile',
  authenticateToken,
  validateProfileUpdate,
  authController.updateUserProfile
);

const validateRegistration = [
  body('email').isEmail().withMessage('Invalid email'),
  body('password').isLength({ min: 8 }).withMessage('Password must be at least 8 characters'),
  body('pan').matches(/[A-Z]{5}[0-9]{4}[A-Z]{1}/).withMessage('Invalid PAN'),
  (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ success: false, errors: errors.array() });
    }
    next();
  }
];

const { handleValidationErrors } = require('../middleware/validation');

// Apply strict rate limiting to critical auth endpoints
router.post('/register', authRateLimiter, validateRegistration, handleValidationErrors, authController.register);
router.get('/verify-email', apiRateLimiter, authController.verifyEmail); // New: Email verification
router.post('/forgot-password', authRateLimiter, authController.forgotPassword); // New: Forgot password
router.post('/reset-password', authController.resetPassword);

module.exports = router; 