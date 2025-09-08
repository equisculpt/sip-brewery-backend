/**
 * 📧📱 Email Verification & Custom OTP Routes
 * RESTful API endpoints for email verification and custom OTP service
 */

const express = require('express');
const { body, param, query, validationResult } = require('express-validator');
const rateLimit = require('express-rate-limit');
const EmailVerificationService = require('../services/EmailVerificationService');
const CustomOTPService = require('../services/CustomOTPService');
const authMiddleware = require('../middleware/authenticationMiddleware');

const router = express.Router();

// =====================================================
// RATE LIMITING CONFIGURATIONS
// =====================================================

// Email verification rate limiting
const emailVerificationLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // 5 email verifications per window
    message: {
        success: false,
        message: 'Too many email verification requests. Please try again later.',
        code: 'EMAIL_RATE_LIMIT_EXCEEDED'
    },
    standardHeaders: true,
    legacyHeaders: false
});

// OTP sending rate limiting
const otpSendLimiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 3, // 3 OTP requests per minute
    message: {
        success: false,
        message: 'Too many OTP requests. Please try again later.',
        code: 'OTP_RATE_LIMIT_EXCEEDED'
    },
    standardHeaders: true,
    legacyHeaders: false
});

// OTP verification rate limiting
const otpVerifyLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 10, // 10 verification attempts per window
    message: {
        success: false,
        message: 'Too many verification attempts. Please try again later.',
        code: 'VERIFY_RATE_LIMIT_EXCEEDED'
    },
    standardHeaders: true,
    legacyHeaders: false
});

// =====================================================
// VALIDATION MIDDLEWARE
// =====================================================

const validateEmail = [
    body('email')
        .isEmail()
        .normalizeEmail()
        .withMessage('Please provide a valid email address'),
    body('purpose')
        .optional()
        .isIn(['EMAIL_VERIFICATION', 'PASSWORD_RESET', 'LOGIN_VERIFICATION', 'ACCOUNT_SECURITY', 'TWO_FACTOR_AUTH'])
        .withMessage('Invalid purpose specified')
];

const validatePhoneNumber = [
    body('phoneNumber')
        .isMobilePhone('en-IN')
        .withMessage('Please provide a valid Indian mobile number'),
    body('purpose')
        .optional()
        .isIn(['LOGIN', 'SIGNUP', 'PASSWORD_RESET', 'TRANSACTION', 'TWO_FACTOR', 'PHONE_VERIFICATION'])
        .withMessage('Invalid purpose specified')
];

const validateOTPVerification = [
    body('otp')
        .isLength({ min: 4, max: 8 })
        .isNumeric()
        .withMessage('OTP must be 4-8 digits'),
    body('otpId')
        .isUUID()
        .withMessage('Invalid OTP ID'),
    body('recipient')
        .notEmpty()
        .withMessage('Recipient is required')
];

const validateMagicLinkVerification = [
    query('token')
        .isLength({ min: 32, max: 128 })
        .withMessage('Invalid verification token'),
    query('id')
        .isUUID()
        .withMessage('Invalid verification ID')
];

// =====================================================
// EMAIL VERIFICATION ROUTES
// =====================================================

/**
 * Send email verification OTP
 * POST /api/email-verification/send-otp
 */
router.post('/email-verification/send-otp', 
    emailVerificationLimiter,
    validateEmail,
    async (req, res) => {
        try {
            // Check validation errors
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array(),
                    code: 'VALIDATION_ERROR'
                });
            }

            const { email, purpose = 'EMAIL_VERIFICATION' } = req.body;
            const userId = req.user?.userId || null;

            const result = await EmailVerificationService.sendVerificationOTP(
                email, userId, purpose
            );

            res.json(result);

        } catch (error) {
            console.error('Email OTP send error:', error);
            res.status(400).json({
                success: false,
                message: error.message,
                code: 'EMAIL_OTP_SEND_FAILED'
            });
        }
    }
);

/**
 * Send email verification magic link
 * POST /api/email-verification/send-magic-link
 */
router.post('/email-verification/send-magic-link',
    emailVerificationLimiter,
    validateEmail,
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array(),
                    code: 'VALIDATION_ERROR'
                });
            }

            const { email, purpose = 'EMAIL_VERIFICATION' } = req.body;
            const userId = req.user?.userId || null;

            const result = await EmailVerificationService.sendMagicLink(
                email, userId, purpose
            );

            res.json(result);

        } catch (error) {
            console.error('Magic link send error:', error);
            res.status(400).json({
                success: false,
                message: error.message,
                code: 'MAGIC_LINK_SEND_FAILED'
            });
        }
    }
);

/**
 * Verify email OTP
 * POST /api/email-verification/verify-otp
 */
router.post('/email-verification/verify-otp',
    otpVerifyLimiter,
    [
        body('email').isEmail().normalizeEmail(),
        body('otp').isLength({ min: 6, max: 6 }).isNumeric(),
        body('verificationId').isUUID()
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array(),
                    code: 'VALIDATION_ERROR'
                });
            }

            const { email, otp, verificationId } = req.body;

            const result = await EmailVerificationService.verifyEmailOTP(
                email, otp, verificationId
            );

            res.json(result);

        } catch (error) {
            console.error('Email OTP verification error:', error);
            res.status(400).json({
                success: false,
                message: error.message,
                code: 'EMAIL_OTP_VERIFICATION_FAILED'
            });
        }
    }
);

/**
 * Verify magic link
 * GET /api/email-verification/verify-magic-link
 */
router.get('/email-verification/verify-magic-link',
    validateMagicLinkVerification,
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array(),
                    code: 'VALIDATION_ERROR'
                });
            }

            const { token, id } = req.query;

            const result = await EmailVerificationService.verifyMagicLink(token, id);

            // For successful verification, you might want to redirect to frontend
            if (result.success) {
                const redirectUrl = `${process.env.FRONTEND_URL}/email-verified?success=true`;
                return res.redirect(redirectUrl);
            }

            res.json(result);

        } catch (error) {
            console.error('Magic link verification error:', error);
            
            // Redirect to frontend with error
            const redirectUrl = `${process.env.FRONTEND_URL}/email-verified?success=false&error=${encodeURIComponent(error.message)}`;
            res.redirect(redirectUrl);
        }
    }
);

// =====================================================
// CUSTOM OTP SERVICE ROUTES
// =====================================================

/**
 * Send SMS OTP
 * POST /api/otp/send-sms
 */
router.post('/otp/send-sms',
    otpSendLimiter,
    validatePhoneNumber,
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array(),
                    code: 'VALIDATION_ERROR'
                });
            }

            const { phoneNumber, purpose = 'LOGIN' } = req.body;
            const userId = req.user?.userId || null;

            const result = await CustomOTPService.sendSMSOTP(
                phoneNumber, purpose, userId
            );

            res.json(result);

        } catch (error) {
            console.error('SMS OTP send error:', error);
            res.status(400).json({
                success: false,
                message: error.message,
                code: 'SMS_OTP_SEND_FAILED'
            });
        }
    }
);

/**
 * Send Email OTP (via custom service)
 * POST /api/otp/send-email
 */
router.post('/otp/send-email',
    otpSendLimiter,
    validateEmail,
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array(),
                    code: 'VALIDATION_ERROR'
                });
            }

            const { email, purpose = 'EMAIL_VERIFICATION' } = req.body;
            const userId = req.user?.userId || null;

            const result = await CustomOTPService.sendEmailOTP(
                email, purpose, userId
            );

            res.json(result);

        } catch (error) {
            console.error('Email OTP send error:', error);
            res.status(400).json({
                success: false,
                message: error.message,
                code: 'EMAIL_OTP_SEND_FAILED'
            });
        }
    }
);

/**
 * Verify OTP (SMS or Email)
 * POST /api/otp/verify
 */
router.post('/otp/verify',
    otpVerifyLimiter,
    validateOTPVerification,
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array(),
                    code: 'VALIDATION_ERROR'
                });
            }

            const { otpId, otp, recipient } = req.body;

            const result = await CustomOTPService.verifyOTP(otpId, otp, recipient);

            res.json(result);

        } catch (error) {
            console.error('OTP verification error:', error);
            res.status(400).json({
                success: false,
                message: error.message,
                code: 'OTP_VERIFICATION_FAILED'
            });
        }
    }
);

// =====================================================
// ADMIN AND MONITORING ROUTES
// =====================================================

/**
 * Get OTP statistics (Admin only)
 * GET /api/otp/stats
 */
router.get('/otp/stats',
    authMiddleware.requireRole(['ADMIN', 'SECURITY_ANALYST']),
    [
        query('timeRange')
            .optional()
            .isIn(['1 hour', '24 hours', '7 days', '30 days'])
            .withMessage('Invalid time range')
    ],
    async (req, res) => {
        try {
            const { timeRange = '24 hours' } = req.query;

            const [otpStats, emailStats] = await Promise.all([
                CustomOTPService.getOTPStats(timeRange),
                EmailVerificationService.getVerificationStats(timeRange)
            ]);

            res.json({
                success: true,
                data: {
                    otp: otpStats,
                    email: emailStats,
                    timeRange
                }
            });

        } catch (error) {
            console.error('Stats retrieval error:', error);
            res.status(500).json({
                success: false,
                message: 'Failed to retrieve statistics',
                code: 'STATS_RETRIEVAL_FAILED'
            });
        }
    }
);

/**
 * Test SMS providers (Admin only)
 * POST /api/otp/test-sms-providers
 */
router.post('/otp/test-sms-providers',
    authMiddleware.requireRole(['ADMIN']),
    async (req, res) => {
        try {
            const results = await CustomOTPService.testSMSProviders();

            res.json({
                success: true,
                data: results,
                message: 'SMS provider test completed'
            });

        } catch (error) {
            console.error('SMS provider test error:', error);
            res.status(500).json({
                success: false,
                message: 'Failed to test SMS providers',
                code: 'SMS_PROVIDER_TEST_FAILED'
            });
        }
    }
);

/**
 * Cleanup expired records (Admin only)
 * POST /api/otp/cleanup
 */
router.post('/otp/cleanup',
    authMiddleware.requireRole(['ADMIN']),
    async (req, res) => {
        try {
            const [otpCleanup, emailCleanup] = await Promise.all([
                CustomOTPService.cleanupExpiredOTPs(),
                EmailVerificationService.cleanupExpiredVerifications()
            ]);

            res.json({
                success: true,
                data: {
                    otpRecordsCleanedUp: otpCleanup,
                    emailRecordsCleanedUp: emailCleanup
                },
                message: 'Cleanup completed successfully'
            });

        } catch (error) {
            console.error('Cleanup error:', error);
            res.status(500).json({
                success: false,
                message: 'Failed to cleanup expired records',
                code: 'CLEANUP_FAILED'
            });
        }
    }
);

// =====================================================
// HEALTH CHECK ROUTES
// =====================================================

/**
 * Health check for email and OTP services
 * GET /api/otp/health
 */
router.get('/health', async (req, res) => {
    try {
        const health = {
            status: 'healthy',
            timestamp: new Date().toISOString(),
            services: {
                emailVerification: 'operational',
                customOTP: 'operational',
                database: 'operational'
            }
        };

        // Test database connectivity
        try {
            await CustomOTPService.db.query('SELECT 1');
        } catch (dbError) {
            health.services.database = 'degraded';
            health.status = 'degraded';
        }

        // Test email service
        try {
            await EmailVerificationService.transporter.verify();
        } catch (emailError) {
            health.services.emailVerification = 'degraded';
            health.status = 'degraded';
        }

        const statusCode = health.status === 'healthy' ? 200 : 503;
        res.status(statusCode).json(health);

    } catch (error) {
        console.error('Health check error:', error);
        res.status(503).json({
            status: 'unhealthy',
            timestamp: new Date().toISOString(),
            error: error.message
        });
    }
});

// =====================================================
// ERROR HANDLING MIDDLEWARE
// =====================================================

router.use((error, req, res, next) => {
    console.error('Email/OTP Route Error:', error);

    // Handle specific error types
    if (error.name === 'ValidationError') {
        return res.status(400).json({
            success: false,
            message: 'Validation failed',
            errors: error.details,
            code: 'VALIDATION_ERROR'
        });
    }

    if (error.name === 'RateLimitError') {
        return res.status(429).json({
            success: false,
            message: 'Rate limit exceeded',
            retryAfter: error.retryAfter,
            code: 'RATE_LIMIT_EXCEEDED'
        });
    }

    // Generic error response
    res.status(500).json({
        success: false,
        message: 'Internal server error',
        code: 'INTERNAL_SERVER_ERROR'
    });
});

module.exports = router;
