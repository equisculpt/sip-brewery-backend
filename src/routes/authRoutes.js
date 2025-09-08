/**
 * 🔐 Authentication Routes
 * Mobile-first OTP authentication with enterprise security
 */

const express = require('express');
const rateLimit = require('express-rate-limit');
const { body, validationResult } = require('express-validator');
const AuthService = require('../services/AuthService');
const { authenticateToken, getClientInfo } = require('../middleware/authMiddleware');
const { getJwks } = require('../utils/jwks');
const { decryptPII } = require('../utils/pii');
const { jtiReplayGuard } = require('../middleware/jtiReplayGuard');
const logger = require('../utils/logger');

const router = express.Router();

// Rate limiting configurations
const otpRateLimit = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 3, // 3 OTP requests per minute per IP
    message: {
        success: false,
        message: 'Too many OTP requests. Please wait before requesting again.',
        retryAfter: 60
    },
    standardHeaders: true,
    legacyHeaders: false,
});

/**
 * @route   POST /api/auth/admin/sessions/revoke-all-except-current
 * @desc    Admin: revoke all sessions for a user except current session
 * @access  Admin Only
 */
router.post('/admin/sessions/revoke-all-except-current', authenticateToken, jtiReplayGuard(), async (req, res) => {
    try {
        if (!req.user || !['admin', 'super_admin'].includes(req.user.role)) {
            return res.status(403).json({ success: false, message: 'Admin access required' });
        }
        const { userId } = req.body;
        if (!userId) {
            return res.status(400).json({ success: false, message: 'userId is required' });
        }
        const currentSid = req.session?.token;
        await AuthService.db.query(
            "UPDATE user_sessions SET status = 'REVOKED', revoked_at = NOW(), revoked_reason = 'ADMIN_REVOKE_OTHERS' WHERE user_id = $1 AND status = 'ACTIVE' AND session_token <> $2",
            [userId, currentSid || '__none__']
        );
        res.json({ success: true, message: 'Revoked all other sessions' });
    } catch (error) {
        console.error('Admin revoke-all-except-current error:', error);
        res.status(500).json({ success: false, message: 'Failed to revoke sessions' });
    }
});

/**
 * @route   POST /api/auth/admin/sessions/:sessionId/trust
 * @desc    Admin: set or unset device trust flag for a session
 * @access  Admin Only
 */
router.post('/admin/sessions/:sessionId/trust', authenticateToken, jtiReplayGuard(), async (req, res) => {
    try {
        if (!req.user || !['admin', 'super_admin'].includes(req.user.role)) {
            return res.status(403).json({ success: false, message: 'Admin access required' });
        }
        const { sessionId } = req.params;
        const { trusted } = req.body;
        if (!sessionId || typeof trusted !== 'boolean') {
            return res.status(400).json({ success: false, message: 'sessionId and trusted(boolean) are required' });
        }
        await AuthService.db.query(
            'UPDATE user_sessions SET is_trusted_device = $1 WHERE id = $2',
            [trusted, sessionId]
        );
        res.json({ success: true, message: 'Trust flag updated' });
    } catch (error) {
        console.error('Admin trust flag update error:', error);
        res.status(500).json({ success: false, message: 'Failed to update trust flag' });
    }
});

/**
 * @route   POST /api/auth/logout
 * @desc    Logout current session
 * @access  Private
 */
router.post('/logout', authenticateToken, jtiReplayGuard(), async (req, res) => {
    try {
        const userId = req.user.userId || req.user.id;
        const sessionToken = req.session?.token;

        if (!sessionToken) {
            return res.status(400).json({ success: false, message: 'No active session' });
        }

        await AuthService.db.query(
            "UPDATE user_sessions SET status = 'REVOKED', revoked_at = NOW(), revoked_reason = 'USER_LOGOUT' WHERE user_id = $1 AND session_token = $2",
            [userId, sessionToken]
        );

        // Clear refresh token cookie
        res.clearCookie('refresh_token', { httpOnly: true, secure: process.env.NODE_ENV === 'production', sameSite: 'strict', path: '/' });

        res.json({ success: true, message: 'Logged out successfully' });
    } catch (error) {
        console.error('Logout error:', error);
        res.status(500).json({ success: false, message: 'Failed to logout' });
    }
});

/**
 * @route   POST /api/auth/logout-all
 * @desc    Logout all sessions for current user
 * @access  Private
 */
router.post('/logout-all', authenticateToken, jtiReplayGuard(), async (req, res) => {
    try {
        const userId = req.user.userId || req.user.id;
        await AuthService.db.query(
            "UPDATE user_sessions SET status = 'REVOKED', revoked_at = NOW(), revoked_reason = 'USER_LOGOUT_ALL' WHERE user_id = $1 AND status = 'ACTIVE'",
            [userId]
        );
        res.clearCookie('refresh_token', { httpOnly: true, secure: process.env.NODE_ENV === 'production', sameSite: 'strict', path: '/' });
        res.json({ success: true, message: 'All sessions revoked' });
    } catch (error) {
        console.error('Logout-all error:', error);
        res.status(500).json({ success: false, message: 'Failed to revoke all sessions' });
    }
});

/**
 * @route   GET /api/auth/jwks.json
 * @desc    JWKS endpoint for RS256 public key(s)
 * @access  Public
 */
router.get('/jwks.json', async (req, res) => {
    try {
        const jwks = getJwks();
        // basic access logging for observability
        try {
            logger.info('JWKS served', {
                ip: req.ip,
                ua: req.headers['user-agent'] || '',
                keys: Array.isArray(jwks.keys) ? jwks.keys.length : 0
            });
        } catch (_) { /* noop */ }
        res.set('Cache-Control', 'public, max-age=600');
        res.json(jwks);
    } catch (error) {
        res.status(500).json({ success: false, message: 'JWKS not available' });
    }
});

const loginRateLimit = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 10, // 10 login attempts per 15 minutes per IP
    message: {
        success: false,
        message: 'Too many login attempts. Please try again later.',
        retryAfter: 900
    }
});

const generalRateLimit = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // 100 requests per 15 minutes per IP
    message: {
        success: false,
        message: 'Too many requests. Please try again later.'
    }
});

// Apply general rate limiting to all auth routes
router.use(generalRateLimit);

// Validation rules
const phoneValidation = [
    body('phone')
        .isMobilePhone('en-IN')
        .withMessage('Please enter a valid Indian mobile number')
        .customSanitizer(value => {
            // Normalize phone number
            const digits = value.replace(/\D/g, '');
            if (digits.startsWith('91') && digits.length === 12) {
                return '+' + digits;
            } else if (digits.length === 10) {
                return '+91' + digits;
            }
            return value;
        })
];

const otpValidation = [
    body('otp')
        .isLength({ min: 6, max: 6 })
        .isNumeric()
        .withMessage('OTP must be 6 digits')
];

const nameValidation = [
    body('name')
        .trim()
        .isLength({ min: 2, max: 50 })
        .matches(/^[a-zA-Z\s]+$/)
        .withMessage('Name must be 2-50 characters and contain only letters')
];

const emailValidation = [
    body('email')
        .optional()
        .isEmail()
        .normalizeEmail()
        .withMessage('Please enter a valid email address')
];

const passwordValidation = [
    body('password')
        .isLength({ min: 8 })
        .matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/)
        .withMessage('Password must be at least 8 characters with uppercase, lowercase, number and special character')
];

/**
 * @route   POST /api/auth/signup/initiate
 * @desc    Initiate signup process by sending OTP
 * @access  Public
 */
router.post('/signup/initiate', 
    otpRateLimit,
    phoneValidation,
    async (req, res) => {
        try {
            // Validate input
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { phone } = req.body;
            const clientInfo = getClientInfo(req);

            const result = await AuthService.initiateSignup(
                phone,
                clientInfo.ipAddress,
                clientInfo.userAgent
            );

            res.json(result);

        } catch (error) {
            console.error('Signup initiation error:', error);
            res.status(400).json({
                success: false,
                message: error.message
            });
        }
    }
);

/**
 * @route   POST /api/auth/signup/complete
 * @desc    Complete signup after OTP verification
 * @access  Public
 */
router.post('/signup/complete',
    loginRateLimit,
    [
        ...phoneValidation,
        ...otpValidation,
        ...nameValidation,
        ...emailValidation
    ],
    async (req, res) => {
        try {
            // Validate input
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { phone, otp, name, email } = req.body;
            const clientInfo = getClientInfo(req);

            const result = await AuthService.completeSignup(
                phone,
                otp,
                name,
                email,
                clientInfo.ipAddress,
                clientInfo.userAgent
            );

            // Set secure HTTP-only cookie for refresh token
            res.cookie('refreshToken', result.tokens.refreshToken, {
                httpOnly: true,
                secure: process.env.NODE_ENV === 'production',
                sameSite: 'strict',
                maxAge: 7 * 24 * 60 * 60 * 1000 // 7 days
            });

            // Remove refresh token from response body
            const response = {
                ...result,
                tokens: {
                    accessToken: result.tokens.accessToken,
                    sessionToken: result.tokens.sessionToken,
                    expiresIn: result.tokens.expiresIn,
                    tokenType: result.tokens.tokenType
                }
            };

            res.status(201).json(response);

        } catch (error) {
            console.error('Signup completion error:', error);
            res.status(400).json({
                success: false,
                message: error.message
            });
        }
    }
);

/**
 * @route   POST /api/auth/login/initiate
 * @desc    Initiate login process by sending OTP
 * @access  Public
 */
router.post('/login/initiate',
    otpRateLimit,
    phoneValidation,
    async (req, res) => {
        try {
            // Validate input
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { phone } = req.body;
            const clientInfo = getClientInfo(req);

            const result = await AuthService.initiateLogin(
                phone,
                clientInfo.ipAddress,
                clientInfo.userAgent
            );

            res.json(result);

        } catch (error) {
            console.error('Login initiation error:', error);
            res.status(400).json({
                success: false,
                message: error.message
            });
        }
    }
);

/**
 * @route   POST /api/auth/login/complete
 * @desc    Complete login with OTP verification
 * @access  Public
 */
router.post('/login/complete',
    loginRateLimit,
    [
        ...phoneValidation,
        ...otpValidation
    ],
    async (req, res) => {
        try {
            // Validate input
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { phone, otp } = req.body;
            const clientInfo = getClientInfo(req);

            // Extract device info from headers
            const deviceInfo = {
                deviceId: req.headers['x-device-id'],
                deviceName: req.headers['x-device-name'],
                deviceType: req.headers['x-device-type'] || 'UNKNOWN',
                browser: clientInfo.browser,
                os: clientInfo.os
            };

            const result = await AuthService.completeLogin(
                phone,
                otp,
                clientInfo.ipAddress,
                clientInfo.userAgent,
                deviceInfo
            );

            // Set secure HTTP-only cookie for refresh token
            res.cookie('refreshToken', result.tokens.refreshToken, {
                httpOnly: true,
                secure: process.env.NODE_ENV === 'production',
                sameSite: 'strict',
                maxAge: 7 * 24 * 60 * 60 * 1000 // 7 days
            });

            // Remove refresh token from response body
            const response = {
                ...result,
                tokens: {
                    accessToken: result.tokens.accessToken,
                    sessionToken: result.tokens.sessionToken,
                    expiresIn: result.tokens.expiresIn,
                    tokenType: result.tokens.tokenType
                }
            };

            res.json(response);

        } catch (error) {
            console.error('Login completion error:', error);
            res.status(400).json({
                success: false,
                message: error.message
            });
        }
    }
);

/**
 * @route   POST /api/auth/login/password
 * @desc    Login with phone and password (optional method)
 * @access  Public
 */
router.post('/login/password',
    loginRateLimit,
    [
        ...phoneValidation,
        body('password').notEmpty().withMessage('Password is required')
    ],
    async (req, res) => {
        try {
            // Validate input
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { phone, password } = req.body;
            const clientInfo = getClientInfo(req);

            const deviceInfo = {
                deviceId: req.headers['x-device-id'],
                deviceName: req.headers['x-device-name'],
                deviceType: req.headers['x-device-type'] || 'UNKNOWN',
                browser: clientInfo.browser,
                os: clientInfo.os
            };

            const result = await AuthService.loginWithPassword(
                phone,
                password,
                clientInfo.ipAddress,
                clientInfo.userAgent,
                deviceInfo
            );

            // Set secure HTTP-only cookie for refresh token
            res.cookie('refreshToken', result.tokens.refreshToken, {
                httpOnly: true,
                secure: process.env.NODE_ENV === 'production',
                sameSite: 'strict',
                maxAge: 7 * 24 * 60 * 60 * 1000 // 7 days
            });

            // Remove refresh token from response body
            const response = {
                ...result,
                tokens: {
                    accessToken: result.tokens.accessToken,
                    sessionToken: result.tokens.sessionToken,
                    expiresIn: result.tokens.expiresIn,
                    tokenType: result.tokens.tokenType
                }
            };

            res.json(response);

        } catch (error) {
            console.error('Password login error:', error);
            res.status(400).json({
                success: false,
                message: error.message
            });
        }
    }
);

/**
 * @route   POST /api/auth/refresh
 * @desc    Refresh access token
 * @access  Public (requires refresh token)
 */
router.post('/refresh', async (req, res) => {
    try {
        const refreshToken = req.cookies.refreshToken || req.body.refreshToken;
        
        if (!refreshToken) {
            return res.status(401).json({
                success: false,
                message: 'Refresh token not provided'
            });
        }

        const clientInfo = getClientInfo(req);

        const result = await AuthService.refreshToken(
            refreshToken,
            clientInfo.ipAddress,
            clientInfo.userAgent
        );

        res.json(result);

    } catch (error) {
        console.error('Token refresh error:', error);
        res.status(401).json({
            success: false,
            message: 'Token refresh failed'
        });
    }
});

/**
 * @route   POST /api/auth/logout
 * @desc    Logout user and revoke session
 * @access  Private
 */
router.post('/logout', authenticateToken, async (req, res) => {
    try {
        const userId = req.user.userId;
        const sessionToken = req.headers['x-session-token'];
        const clientInfo = getClientInfo(req);

        await AuthService.logout(
            userId,
            sessionToken,
            clientInfo.ipAddress,
            clientInfo.userAgent
        );

        // Clear refresh token cookie
        res.clearCookie('refreshToken');

        res.json({
            success: true,
            message: 'Logged out successfully'
        });

    } catch (error) {
        console.error('Logout error:', error);
        res.status(400).json({
            success: false,
            message: error.message
        });
    }
});

/**
 * @route   POST /api/auth/set-password
 * @desc    Set password for user account (optional)
 * @access  Private
 */
router.post('/set-password',
    authenticateToken,
    jtiReplayGuard(),
    passwordValidation,
    async (req, res) => {
        try {
            // Validate input
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const userId = req.user.userId;
            const { password } = req.body;
            const clientInfo = getClientInfo(req);

            const result = await AuthService.setPassword(
                userId,
                password,
                clientInfo.ipAddress,
                clientInfo.userAgent
            );

            res.json(result);

        } catch (error) {
            console.error('Set password error:', error);
            res.status(400).json({
                success: false,
                message: error.message
            });
        }
    }
);

/**
 * @route   GET /api/auth/me
 * @desc    Get current user profile
 * @access  Private
 */
router.get('/me', authenticateToken, async (req, res) => {
    try {
        const userId = req.user.userId;

        // Get user details
        const userQuery = `
            SELECT u.id, u.phone, u.name, u.email, u.phone_verified, 
                   u.email_verified, u.status, u.kyc_status, u.two_fa_enabled,
                   u.created_at, u.last_login_at,
                   up.first_name, up.last_name, up.date_of_birth,
                   up.investment_experience, up.risk_tolerance
            FROM users u
            LEFT JOIN user_profiles up ON u.id = up.user_id
            WHERE u.id = $1
        `;

        const result = await AuthService.db.query(userQuery, [userId]);
        
        if (result.rows.length === 0) {
            return res.status(404).json({
                success: false,
                message: 'User not found'
            });
        }

        const user = result.rows[0];
        const email = decryptPII(user.email) || user.email;
        const phone = decryptPII(user.phone) || user.phone;

        res.json({
            success: true,
            user: {
                id: user.id,
                phone,
                name: user.name,
                email,
                phoneVerified: user.phone_verified,
                emailVerified: user.email_verified,
                status: user.status,
                kycStatus: user.kyc_status,
                twoFAEnabled: user.two_fa_enabled,
                profile: {
                    firstName: user.first_name,
                    lastName: user.last_name,
                    dateOfBirth: user.date_of_birth,
                    investmentExperience: user.investment_experience,
                    riskTolerance: user.risk_tolerance
                },
                createdAt: user.created_at,
                lastLoginAt: user.last_login_at
            }
        });

    } catch (error) {
        console.error('Get user profile error:', error);
        res.status(500).json({
            success: false,
            message: 'Failed to get user profile'
        });
    }
});

/**
 * @route   GET /api/auth/sessions
 * @desc    Get user's active sessions
 * @access  Private
 */
router.get('/sessions', authenticateToken, async (req, res) => {
    try {
        const userId = req.user.userId;

        const sessionsQuery = `
            SELECT id, device_name, device_type, browser, os,
                   ip_address, location, is_trusted_device,
                   created_at, last_activity_at
            FROM user_sessions
            WHERE user_id = $1 AND status = 'ACTIVE'
            ORDER BY last_activity_at DESC
        `;

        const result = await AuthService.db.query(sessionsQuery, [userId]);

        res.json({
            success: true,
            sessions: result.rows
        });

    } catch (error) {
        console.error('Get sessions error:', error);
        res.status(500).json({
            success: false,
            message: 'Failed to get sessions'
        });
    }
});

/**
 * @route   POST /api/auth/revoke-session
 * @desc    Revoke a specific session
 * @access  Private
 */
router.post('/revoke-session', authenticateToken, jtiReplayGuard(), async (req, res) => {
    try {
        const userId = req.user.userId;
        const { sessionId } = req.body;

        if (!sessionId) {
            return res.status(400).json({
                success: false,
                message: 'Session ID is required'
            });
        }

        const revokeQuery = `
            UPDATE user_sessions 
            SET status = 'REVOKED', revoked_at = NOW(), revoked_reason = 'USER_REVOKED'
            WHERE id = $1 AND user_id = $2
        `;

        await AuthService.db.query(revokeQuery, [sessionId, userId]);

        res.json({
            success: true,
            message: 'Session revoked successfully'
        });

    } catch (error) {
        console.error('Revoke session error:', error);
        res.status(500).json({
            success: false,
            message: 'Failed to revoke session'
        });
    }
});

module.exports = router;
