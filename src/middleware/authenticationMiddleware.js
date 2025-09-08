/**
 * 🔐 Authentication Middleware
 * Enterprise-grade JWT authentication with comprehensive security
 */

const jwt = require('jsonwebtoken');
const { Pool } = require('pg');
const useragent = require('useragent');

const db = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
});

/**
 * Extract client information from request
 */
const getClientInfo = (req) => {
    const userAgent = req.headers['user-agent'] || '';
    const agent = useragent.parse(userAgent);
    
    // Get IP address (considering proxy headers)
    const ipAddress = req.headers['x-forwarded-for']?.split(',')[0] || 
                     req.headers['x-real-ip'] || 
                     req.connection.remoteAddress || 
                     req.socket.remoteAddress ||
                     req.ip;

    return {
        ipAddress: ipAddress?.replace('::ffff:', ''), // Clean IPv4-mapped IPv6
        userAgent: userAgent,
        browser: `${agent.family} ${agent.major}`,
        os: `${agent.os.family} ${agent.os.major}`,
        device: agent.device.family
    };
};

/**
 * Authenticate JWT token middleware
 */
const authenticateToken = async (req, res, next) => {
    try {
        // Extract token from Authorization header
        const authHeader = req.headers['authorization'];
        const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

        if (!token) {
            return res.status(401).json({
                success: false,
                message: 'Access token required',
                code: 'TOKEN_MISSING'
            });
        }

        // Verify JWT token
        let decoded;
        try {
            decoded = jwt.verify(token, process.env.JWT_SECRET);
        } catch (jwtError) {
            if (jwtError.name === 'TokenExpiredError') {
                return res.status(401).json({
                    success: false,
                    message: 'Access token expired',
                    code: 'TOKEN_EXPIRED'
                });
            } else if (jwtError.name === 'JsonWebTokenError') {
                return res.status(401).json({
                    success: false,
                    message: 'Invalid access token',
                    code: 'TOKEN_INVALID'
                });
            } else {
                throw jwtError;
            }
        }

        // Validate token type
        if (decoded.type !== 'access') {
            return res.status(401).json({
                success: false,
                message: 'Invalid token type',
                code: 'TOKEN_INVALID_TYPE'
            });
        }

        // Check if user exists and is active
        const userQuery = `
            SELECT id, phone, name, email, status, kyc_status, 
                   two_fa_enabled, locked_until
            FROM users 
            WHERE id = $1
        `;

        const userResult = await db.query(userQuery, [decoded.userId]);

        if (userResult.rows.length === 0) {
            return res.status(401).json({
                success: false,
                message: 'User not found',
                code: 'USER_NOT_FOUND'
            });
        }

        const user = userResult.rows[0];

        // Check if user account is active
        if (user.status !== 'ACTIVE') {
            return res.status(403).json({
                success: false,
                message: 'Account is suspended',
                code: 'ACCOUNT_SUSPENDED'
            });
        }

        // Check if account is locked
        if (user.locked_until && new Date(user.locked_until) > new Date()) {
            return res.status(403).json({
                success: false,
                message: 'Account is temporarily locked',
                code: 'ACCOUNT_LOCKED',
                lockedUntil: user.locked_until
            });
        }

        // Check if session exists and is valid
        const sessionToken = req.headers['x-session-token'];
        if (sessionToken) {
            const sessionQuery = `
                SELECT id, status, expires_at, last_activity_at
                FROM user_sessions
                WHERE user_id = $1 AND session_token = $2
            `;

            const sessionResult = await db.query(sessionQuery, [decoded.userId, sessionToken]);

            if (sessionResult.rows.length === 0) {
                return res.status(401).json({
                    success: false,
                    message: 'Invalid session',
                    code: 'SESSION_INVALID'
                });
            }

            const session = sessionResult.rows[0];

            if (session.status !== 'ACTIVE') {
                return res.status(401).json({
                    success: false,
                    message: 'Session has been revoked',
                    code: 'SESSION_REVOKED'
                });
            }

            if (new Date(session.expires_at) < new Date()) {
                return res.status(401).json({
                    success: false,
                    message: 'Session has expired',
                    code: 'SESSION_EXPIRED'
                });
            }

            // Update session activity
            await db.query(
                'UPDATE user_sessions SET last_activity_at = NOW() WHERE id = $1',
                [session.id]
            );
        }

        // Add user info to request object
        req.user = {
            userId: user.id,
            phone: user.phone,
            name: user.name,
            email: user.email,
            status: user.status,
            kycStatus: user.kyc_status,
            twoFAEnabled: user.two_fa_enabled
        };

        // Add client info to request
        req.clientInfo = getClientInfo(req);

        next();

    } catch (error) {
        console.error('Authentication middleware error:', error);
        res.status(500).json({
            success: false,
            message: 'Authentication failed',
            code: 'AUTH_ERROR'
        });
    }
};

/**
 * Optional authentication middleware (doesn't fail if no token)
 */
const optionalAuth = async (req, res, next) => {
    try {
        const authHeader = req.headers['authorization'];
        const token = authHeader && authHeader.split(' ')[1];

        if (!token) {
            req.user = null;
            req.clientInfo = getClientInfo(req);
            return next();
        }

        // Try to authenticate, but don't fail if token is invalid
        try {
            const decoded = jwt.verify(token, process.env.JWT_SECRET);
            
            if (decoded.type === 'access') {
                const userQuery = `
                    SELECT id, phone, name, email, status, kyc_status, two_fa_enabled
                    FROM users 
                    WHERE id = $1 AND status = 'ACTIVE'
                `;

                const userResult = await db.query(userQuery, [decoded.userId]);

                if (userResult.rows.length > 0) {
                    const user = userResult.rows[0];
                    req.user = {
                        userId: user.id,
                        phone: user.phone,
                        name: user.name,
                        email: user.email,
                        status: user.status,
                        kycStatus: user.kyc_status,
                        twoFAEnabled: user.two_fa_enabled
                    };
                }
            }
        } catch (jwtError) {
            // Ignore JWT errors in optional auth
            req.user = null;
        }

        req.clientInfo = getClientInfo(req);
        next();

    } catch (error) {
        console.error('Optional auth middleware error:', error);
        req.user = null;
        req.clientInfo = getClientInfo(req);
        next();
    }
};

/**
 * Require KYC verification middleware
 */
const requireKYC = (req, res, next) => {
    if (!req.user) {
        return res.status(401).json({
            success: false,
            message: 'Authentication required',
            code: 'AUTH_REQUIRED'
        });
    }

    if (req.user.kycStatus !== 'VERIFIED') {
        return res.status(403).json({
            success: false,
            message: 'KYC verification required',
            code: 'KYC_REQUIRED',
            kycStatus: req.user.kycStatus
        });
    }

    next();
};

/**
 * Require 2FA verification middleware
 */
const require2FA = (req, res, next) => {
    if (!req.user) {
        return res.status(401).json({
            success: false,
            message: 'Authentication required',
            code: 'AUTH_REQUIRED'
        });
    }

    if (req.user.twoFAEnabled) {
        // Check if 2FA was verified in this session
        const twoFAVerified = req.headers['x-2fa-verified'];
        if (!twoFAVerified || twoFAVerified !== 'true') {
            return res.status(403).json({
                success: false,
                message: '2FA verification required',
                code: '2FA_REQUIRED'
            });
        }
    }

    next();
};

/**
 * Role-based access control middleware
 */
const requireRole = (roles) => {
    return async (req, res, next) => {
        if (!req.user) {
            return res.status(401).json({
                success: false,
                message: 'Authentication required',
                code: 'AUTH_REQUIRED'
            });
        }

        // Get user role from database
        const roleQuery = `
            SELECT role FROM users WHERE id = $1
        `;

        try {
            const result = await db.query(roleQuery, [req.user.userId]);
            
            if (result.rows.length === 0) {
                return res.status(401).json({
                    success: false,
                    message: 'User not found',
                    code: 'USER_NOT_FOUND'
                });
            }

            const userRole = result.rows[0].role || 'USER';

            if (!roles.includes(userRole)) {
                return res.status(403).json({
                    success: false,
                    message: 'Insufficient permissions',
                    code: 'INSUFFICIENT_PERMISSIONS',
                    requiredRoles: roles,
                    userRole: userRole
                });
            }

            req.user.role = userRole;
            next();

        } catch (error) {
            console.error('Role check error:', error);
            res.status(500).json({
                success: false,
                message: 'Authorization failed',
                code: 'AUTH_ERROR'
            });
        }
    };
};

/**
 * Rate limiting by user ID
 */
const userRateLimit = (maxRequests, windowMs) => {
    const requests = new Map();

    return (req, res, next) => {
        if (!req.user) {
            return next();
        }

        const userId = req.user.userId;
        const now = Date.now();
        const windowStart = now - windowMs;

        // Clean old entries
        if (requests.has(userId)) {
            const userRequests = requests.get(userId);
            const validRequests = userRequests.filter(timestamp => timestamp > windowStart);
            requests.set(userId, validRequests);
        }

        // Check current requests
        const currentRequests = requests.get(userId) || [];
        
        if (currentRequests.length >= maxRequests) {
            return res.status(429).json({
                success: false,
                message: 'Too many requests',
                code: 'RATE_LIMIT_EXCEEDED',
                retryAfter: Math.ceil(windowMs / 1000)
            });
        }

        // Add current request
        currentRequests.push(now);
        requests.set(userId, currentRequests);

        next();
    };
};

/**
 * Log security events middleware
 */
const logSecurityEvent = (eventType, category = 'SECURITY', severity = 'INFO') => {
    return async (req, res, next) => {
        try {
            const userId = req.user?.userId || null;
            const clientInfo = req.clientInfo || getClientInfo(req);

            const logQuery = `
                INSERT INTO security_events (
                    user_id, event_type, event_category, severity,
                    description, metadata, ip_address, user_agent, endpoint
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            `;

            const metadata = {
                method: req.method,
                path: req.path,
                query: req.query,
                body: req.method === 'POST' ? Object.keys(req.body) : undefined
            };

            await db.query(logQuery, [
                userId,
                eventType,
                category,
                severity,
                `${eventType} - ${req.method} ${req.path}`,
                JSON.stringify(metadata),
                clientInfo.ipAddress,
                clientInfo.userAgent,
                req.originalUrl
            ]);

            next();

        } catch (error) {
            console.error('Security logging error:', error);
            next(); // Don't fail the request due to logging errors
        }
    };
};

module.exports = {
    authenticateToken,
    optionalAuth,
    requireKYC,
    require2FA,
    requireRole,
    userRateLimit,
    logSecurityEvent,
    getClientInfo
};
