/**
 * 🔐 Enterprise Authentication Service
 * 35+ years experience implementation with mobile-first OTP authentication
 */

const bcrypt = require('bcryptjs');
const crypto = require('crypto');
const { Pool } = require('pg');
const OTPService = require('./OTPService');
const EmailService = require('./EmailService');
const { generateToken, verifyToken } = require('../utils/auth');
const { encryptPII } = require('../utils/pii');

class AuthService {
    constructor() {
        this.db = new Pool({
            connectionString: process.env.DATABASE_URL,
            ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
        });

        // JWT configuration now handled by utils/auth (RS256). TTLs via env.

        // Security Configuration
        this.maxLoginAttempts = 5;
        this.lockoutDuration = 30; // minutes
        this.passwordSaltRounds = 12;

        // Feature flags determined at runtime based on DB schema
        this.supportsPhoneHash = false; // users.phone_hash
        this.supportsEncryptedPhone = true; // we'll write encrypted phone when hash is supported

        // Initialize schema feature detection (no await)
        this._init().catch(() => {});
    }

    async _init() {
        try {
            const q = `
                SELECT 1
                FROM information_schema.columns 
                WHERE table_name = 'users' AND column_name = 'phone_hash'
            `;
            const r = await this.db.query(q);
            this.supportsPhoneHash = r.rows.length > 0;
        } catch (e) {
            this.supportsPhoneHash = false;
        }
    }

    /**
     * User Registration with Mobile OTP
     * Simple flow: Phone → OTP → Name → Account Created
     */
    async initiateSignup(phone, ipAddress, userAgent) {
        try {
            // Validate phone number
            if (!this.isValidPhoneNumber(phone)) {
                throw new Error('Please enter a valid Indian mobile number');
            }

            // Normalize phone number
            const normalizedPhone = this.normalizePhoneNumber(phone);

            // Check if user already exists
            const existingUser = await this.findUserByPhone(normalizedPhone);
            if (existingUser && existingUser.phone_verified) {
                throw new Error('Account already exists. Please login instead.');
            }

            // Send OTP
            const otpResult = await OTPService.sendOTP(
                normalizedPhone, 
                'SIGNUP', 
                ipAddress, 
                userAgent
            );

            return {
                success: true,
                message: 'OTP sent successfully',
                phone: this.maskPhone(normalizedPhone),
                otpId: otpResult.otpId,
                expiresIn: otpResult.expiresIn
            };

        } catch (error) {
            await this.logSecurityEvent(null, 'SIGNUP_INITIATION_FAILED', 'AUTHENTICATION', 'WARNING',
                `Signup initiation failed: ${error.message}`, {
                    phone: this.maskPhone(phone),
                    error: error.message
                }, ipAddress, userAgent);
            
            throw error;
        }
    }

    /**
     * Complete Signup after OTP verification
     */
    async completeSignup(phone, otp, name, email = null, ipAddress, userAgent) {
        const client = await this.db.connect();
        
        try {
            await client.query('BEGIN');

            // Normalize phone number
            const normalizedPhone = this.normalizePhoneNumber(phone);

            // Verify OTP
            await OTPService.verifyOTP(normalizedPhone, otp, 'SIGNUP', ipAddress, userAgent);

            // Validate name
            if (!name || name.trim().length < 2) {
                throw new Error('Please enter a valid name (minimum 2 characters)');
            }

            // Validate email if provided
            if (email && !this.isValidEmail(email)) {
                throw new Error('Please enter a valid email address');
            }

            // Check if user already exists (double-check)
            const existingUser = await this.findUserByPhone(normalizedPhone);
            if (existingUser && existingUser.phone_verified) {
                throw new Error('Account already exists');
            }

            let userId;

            if (existingUser) {
                // Update existing unverified user
                const updateQuery = `
                    UPDATE users 
                    SET name = $1, email = $2, phone_verified = TRUE, 
                        status = 'ACTIVE', updated_at = NOW()
                    WHERE ${this.supportsPhoneHash ? 'phone_hash' : 'phone'} = $3
                    RETURNING id
                `;
                const result = await client.query(updateQuery, [
                    name.trim(),
                    email ? email.trim().toLowerCase() : null,
                    this.supportsPhoneHash ? this.computePhoneHash(normalizedPhone) : existingUser.phone
                ]);
                userId = result.rows[0].id;
            } else {
                // Create new user
                const userQuery = this.supportsPhoneHash
                    ? `INSERT INTO users (phone, phone_hash, name, email, phone_verified, status)
                       VALUES ($1, $2, $3, $4, TRUE, 'ACTIVE') RETURNING id`
                    : `INSERT INTO users (phone, name, email, phone_verified, status)
                       VALUES ($1, $2, $3, TRUE, 'ACTIVE') RETURNING id`;

                const encEmail = email ? encryptPII(email.trim().toLowerCase()) : null;
                const phoneToStore = this.supportsPhoneHash ? encryptPII(normalizedPhone) : normalizedPhone;
                const phoneHash = this.supportsPhoneHash ? this.computePhoneHash(normalizedPhone) : null;
                const params = this.supportsPhoneHash
                    ? [phoneToStore, phoneHash, name.trim(), encEmail]
                    : [phoneToStore, name.trim(), encEmail];
                const userResult = await client.query(userQuery, params);
                userId = userResult.rows[0].id;
            }

            // Create user profile
            await this.createUserProfile(userId, { name: name.trim() }, client);

            // Create notification preferences
            await this.createNotificationPreferences(userId, client);

            // Generate tokens
            const tokens = await this.generateTokens(userId, ipAddress, userAgent, client);

            await client.query('COMMIT');

            // Log successful signup
            await this.logSecurityEvent(userId, 'SIGNUP_COMPLETED', 'AUTHENTICATION', 'INFO',
                'User signup completed successfully', {
                    phone: this.maskPhone(normalizedPhone),
                    hasEmail: !!email
                }, ipAddress, userAgent);

            return {
                success: true,
                message: 'Account created successfully',
                user: {
                    id: userId,
                    phone: normalizedPhone,
                    name: name.trim(),
                    email: email
                },
                tokens
            };

        } catch (error) {
            await client.query('ROLLBACK');
            
            await this.logSecurityEvent(null, 'SIGNUP_COMPLETION_FAILED', 'AUTHENTICATION', 'ERROR',
                `Signup completion failed: ${error.message}`, {
                    phone: this.maskPhone(phone),
                    error: error.message
                }, ipAddress, userAgent);
            
            throw error;
        } finally {
            client.release();
        }
    }

    /**
     * Login with Mobile OTP (Primary Method)
     */
    async initiateLogin(phone, ipAddress, userAgent) {
        try {
            // Validate phone number
            if (!this.isValidPhoneNumber(phone)) {
                throw new Error('Please enter a valid mobile number');
            }

            // Normalize phone number
            const normalizedPhone = this.normalizePhoneNumber(phone);

            // Check if user exists
            const user = await this.findUserByPhone(normalizedPhone);
            if (!user) {
                throw new Error('Account not found. Please signup first.');
            }

            // Check if account is active
            if (user.status !== 'ACTIVE') {
                throw new Error('Account is suspended. Please contact support.');
            }

            // Check if account is locked
            if (user.locked_until && new Date(user.locked_until) > new Date()) {
                const unlockTime = new Date(user.locked_until).toLocaleString();
                throw new Error(`Account is temporarily locked until ${unlockTime}`);
            }

            // Send OTP
            const otpResult = await OTPService.sendOTP(
                normalizedPhone, 
                'LOGIN', 
                ipAddress, 
                userAgent
            );

            return {
                success: true,
                message: 'OTP sent successfully',
                phone: this.maskPhone(normalizedPhone),
                otpId: otpResult.otpId,
                expiresIn: otpResult.expiresIn,
                hasPassword: !!user.password_hash
            };

        } catch (error) {
            await this.logSecurityEvent(null, 'LOGIN_INITIATION_FAILED', 'AUTHENTICATION', 'WARNING',
                `Login initiation failed: ${error.message}`, {
                    phone: this.maskPhone(phone),
                    error: error.message
                }, ipAddress, userAgent);
            
            throw error;
        }
    }

    /**
     * Complete Login with OTP verification
     */
    async completeLogin(phone, otp, ipAddress, userAgent, deviceInfo = {}) {
        const client = await this.db.connect();
        
        try {
            await client.query('BEGIN');

            // Normalize phone number
            const normalizedPhone = this.normalizePhoneNumber(phone);

            // Verify OTP
            await OTPService.verifyOTP(normalizedPhone, otp, 'LOGIN', ipAddress, userAgent);

            // Get user details
            const user = await this.findUserByPhone(normalizedPhone);
            if (!user) {
                throw new Error('User not found');
            }

            // Reset login attempts on successful login
            await this.resetLoginAttempts(user.id, client);

            // Update last login
            await this.updateLastLogin(user.id, ipAddress, client);

            // Generate tokens
            const tokens = await this.generateTokens(user.id, ipAddress, userAgent, client, deviceInfo);

            await client.query('COMMIT');

            // Log successful login
            await this.logSecurityEvent(user.id, 'LOGIN_SUCCESS', 'AUTHENTICATION', 'INFO',
                'User logged in successfully', {
                    phone: this.maskPhone(normalizedPhone),
                    method: 'OTP',
                    deviceInfo
                }, ipAddress, userAgent);

            return {
                success: true,
                message: 'Login successful',
                user: {
                    id: user.id,
                    phone: user.phone,
                    name: user.name,
                    email: user.email,
                    kyc_status: user.kyc_status,
                    two_fa_enabled: user.two_fa_enabled
                },
                tokens
            };

        } catch (error) {
            await client.query('ROLLBACK');
            
            // Record failed login attempt
            if (error.message.includes('Invalid OTP')) {
                await this.recordFailedLogin(phone, 'INVALID_OTP', ipAddress, userAgent);
            }

            await this.logSecurityEvent(null, 'LOGIN_FAILED', 'AUTHENTICATION', 'WARNING',
                `Login failed: ${error.message}`, {
                    phone: this.maskPhone(phone),
                    error: error.message
                }, ipAddress, userAgent);
            
            throw error;
        } finally {
            client.release();
        }
    }

    /**
     * Password-based login (Optional secondary method)
     */
    async loginWithPassword(phone, password, ipAddress, userAgent, deviceInfo = {}) {
        const client = await this.db.connect();
        
        try {
            await client.query('BEGIN');

            // Normalize phone number
            const normalizedPhone = this.normalizePhoneNumber(phone);

            // Get user
            const user = await this.findUserByPhone(normalizedPhone);
            if (!user) {
                throw new Error('Invalid credentials');
            }

            // Check if password is set
            if (!user.password_hash) {
                throw new Error('Password not set. Please use OTP login.');
            }

            // Verify password
            const isValidPassword = await bcrypt.compare(password, user.password_hash);
            if (!isValidPassword) {
                await this.recordFailedLogin(normalizedPhone, 'INVALID_PASSWORD', ipAddress, userAgent);
                throw new Error('Invalid credentials');
            }

            // Check account status and locks (same as OTP login)
            if (user.status !== 'ACTIVE') {
                throw new Error('Account is suspended');
            }

            if (user.locked_until && new Date(user.locked_until) > new Date()) {
                throw new Error('Account is temporarily locked');
            }

            // Reset login attempts
            await this.resetLoginAttempts(user.id, client);

            // Update last login
            await this.updateLastLogin(user.id, ipAddress, client);

            // Generate tokens
            const tokens = await this.generateTokens(user.id, ipAddress, userAgent, client, deviceInfo);

            await client.query('COMMIT');

            // Log successful login
            await this.logSecurityEvent(user.id, 'LOGIN_SUCCESS', 'AUTHENTICATION', 'INFO',
                'User logged in with password', {
                    phone: this.maskPhone(normalizedPhone),
                    method: 'PASSWORD'
                }, ipAddress, userAgent);

            return {
                success: true,
                message: 'Login successful',
                user: {
                    id: user.id,
                    phone: user.phone,
                    name: user.name,
                    email: user.email,
                    kyc_status: user.kyc_status,
                    two_fa_enabled: user.two_fa_enabled
                },
                tokens
            };

        } catch (error) {
            await client.query('ROLLBACK');
            throw error;
        } finally {
            client.release();
        }
    }

    /**
     * Generate JWT tokens and create session
     */
    async generateTokens(userId, ipAddress, userAgent, client = null, deviceInfo = {}) {
        const db = client || this.db;

        // Create session record first
        const sessionToken = crypto.randomUUID(); // sid embedded in tokens
        const expiresAt = new Date();
        expiresAt.setDate(expiresAt.getDate() + 7); // session expiry for refresh token lifecycle

        const sessionQuery = `
            INSERT INTO user_sessions (
                user_id, session_token, refresh_token_hash,
                device_id, device_name, device_type, browser, os,
                ip_address, expires_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
        `;

        // Temporarily use placeholder for refresh hash until token is generated
        const placeholderHash = await bcrypt.hash(crypto.randomBytes(32).toString('hex'), 10);

        const insertRes = await db.query(sessionQuery, [
            userId, sessionToken, placeholderHash,
            deviceInfo.deviceId || null,
            deviceInfo.deviceName || null,
            deviceInfo.deviceType || 'UNKNOWN',
            deviceInfo.browser || null,
            deviceInfo.os || null,
            ipAddress, expiresAt
        ]);

        // Now issue tokens with sid (sessionToken)
        const accessToken = generateToken({ userId, sid: sessionToken }, 'access');
        const refreshToken = generateToken({ userId, sid: sessionToken }, 'refresh');

        // Hash refresh token for storage (rotate placeholder)
        const refreshTokenHash = await bcrypt.hash(refreshToken, 10);
        await db.query('UPDATE user_sessions SET refresh_token_hash = $1 WHERE session_token = $2', [refreshTokenHash, sessionToken]);

        return {
            accessToken,
            refreshToken,
            sessionToken,
            expiresIn: 15 * 60, // 15 minutes in seconds
            tokenType: 'Bearer'
        };
    }

    /**
     * Refresh access token
     */
    async refreshToken(refreshToken, ipAddress, userAgent) {
        try {
            // Verify refresh token using RS256
            const decoded = verifyToken(refreshToken);

            // Enforce refresh token type
            if (decoded.token_use !== 'refresh') {
                throw new Error('Invalid token type');
            }

            // Validate session by sid
            const sessionQuery = `
                SELECT us.*, u.status AS user_status
                FROM user_sessions us
                JOIN users u ON us.user_id = u.id
                WHERE us.user_id = $1 
                  AND us.session_token = $2
                  AND us.status = 'ACTIVE'
                  AND us.expires_at > NOW()
                LIMIT 1
            `;

            const result = await this.db.query(sessionQuery, [decoded.userId, decoded.sid]);
            if (result.rows.length === 0) {
                throw new Error('Session not found or expired');
            }
            const session = result.rows[0];

            // Compare provided refresh token with stored hash
            const match = await bcrypt.compare(refreshToken, session.refresh_token_hash);
            if (!match) {
                // Possible reuse/compromise -> revoke session
                await this.db.query(
                    "UPDATE user_sessions SET status = 'REVOKED', revoked_at = NOW(), revoked_reason = 'REFRESH_MISMATCH' WHERE id = $1",
                    [session.id]
                );
                await this.logSecurityEvent(session.user_id, 'REFRESH_TOKEN_REUSE', 'SECURITY', 'CRITICAL',
                    'Refresh token mismatch detected; session revoked', { sessionToken: session.session_token }, ipAddress, userAgent);
                throw new Error('Invalid refresh token');
            }

            // Rotate refresh token
            const newRefreshToken = generateToken({ userId: decoded.userId, sid: session.session_token }, 'refresh');
            const newRefreshHash = await bcrypt.hash(newRefreshToken, 10);

            // Generate new access token
            const newAccessToken = generateToken({ userId: decoded.userId, sid: session.session_token }, 'access');

            // Update session
            await this.db.query(
                'UPDATE user_sessions SET refresh_token_hash = $1, last_activity_at = NOW() WHERE id = $2',
                [newRefreshHash, session.id]
            );

            return {
                success: true,
                accessToken: newAccessToken,
                refreshToken: newRefreshToken,
                expiresIn: 15 * 60
            };

        } catch (error) {
            throw new Error('Token refresh failed');
        }
    }

    /**
     * Logout user
     */
    async logout(userId, sessionToken, ipAddress, userAgent) {
        try {
            // Revoke session
            const query = `
                UPDATE user_sessions 
                SET status = 'REVOKED', revoked_at = NOW(), revoked_reason = 'USER_LOGOUT'
                WHERE user_id = $1 AND session_token = $2
            `;

            await this.db.query(query, [userId, sessionToken]);

            // Log logout
            await this.logSecurityEvent(userId, 'LOGOUT', 'AUTHENTICATION', 'INFO',
                'User logged out', {}, ipAddress, userAgent);

            return { success: true, message: 'Logged out successfully' };

        } catch (error) {
            throw new Error('Logout failed');
        }
    }

    /**
     * Set password for user (optional)
     */
    async setPassword(userId, password, ipAddress, userAgent) {
        try {
            // Validate password strength
            if (!this.isStrongPassword(password)) {
                throw new Error('Password must be at least 8 characters with uppercase, lowercase, number and special character');
            }

            // Hash password
            const passwordHash = await bcrypt.hash(password, this.passwordSaltRounds);

            // Update user
            const query = `
                UPDATE users 
                SET password_hash = $1, password_changed_at = NOW(), updated_at = NOW()
                WHERE id = $2
            `;

            await this.db.query(query, [passwordHash, userId]);

            // Log password set
            await this.logSecurityEvent(userId, 'PASSWORD_SET', 'SECURITY', 'INFO',
                'User set password', {}, ipAddress, userAgent);

            return { success: true, message: 'Password set successfully' };

        } catch (error) {
            throw error;
        }
    }

    // Helper methods
    async findUserByPhone(phone) {
        // Use hash lookup if supported
        if (this.supportsPhoneHash) {
            const query = 'SELECT * FROM users WHERE phone_hash = $1';
            const result = await this.db.query(query, [this.computePhoneHash(phone)]);
            return result.rows[0] || null;
        }
        const query = 'SELECT * FROM users WHERE phone = $1';
        const result = await this.db.query(query, [phone]);
        return result.rows[0] || null;
    }

    async createUserProfile(userId, profileData, client = null) {
        const db = client || this.db;
        const query = `
            INSERT INTO user_profiles (user_id, first_name)
            VALUES ($1, $2)
        `;
        await db.query(query, [userId, profileData.name]);
    }

    async createNotificationPreferences(userId, client = null) {
        const db = client || this.db;
        const query = `
            INSERT INTO notification_preferences (user_id)
            VALUES ($1)
        `;
        await db.query(query, [userId]);
    }

    async resetLoginAttempts(userId, client = null) {
        const db = client || this.db;
        const query = `
            UPDATE users 
            SET login_attempts = 0, locked_until = NULL
            WHERE id = $1
        `;
        await db.query(query, [userId]);
    }

    async updateLastLogin(userId, ipAddress, client = null) {
        const db = client || this.db;
        const query = `
            UPDATE users 
            SET last_login_at = NOW(), last_login_ip = $2
            WHERE id = $1
        `;
        await db.query(query, [userId, ipAddress]);
    }

    async recordFailedLogin(identifier, reason, ipAddress, userAgent) {
        // This is handled by OTPService for OTP failures
        // Add additional logic here if needed
    }

    async logSecurityEvent(userId, eventType, category, severity, description, metadata, ipAddress, userAgent) {
        const query = `
            INSERT INTO security_events (
                user_id, event_type, event_category, severity,
                description, metadata, ip_address, user_agent
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        `;

        await this.db.query(query, [
            userId, eventType, category, severity,
            description, JSON.stringify(metadata), ipAddress, userAgent
        ]);
    }

    normalizePhoneNumber(phone) {
        // Remove all non-digits and normalize to +91 format
        const digits = phone.replace(/\D/g, '');
        if (digits.startsWith('91') && digits.length === 12) {
            return '+' + digits;
        } else if (digits.length === 10) {
            return '+91' + digits;
        }
        return phone; // Return as-is if can't normalize
    }

    computePhoneHash(phone) {
        const normalized = this.normalizePhoneNumber(phone);
        return crypto.createHash('sha256').update(normalized).digest('hex');
    }

    isValidPhoneNumber(phone) {
        const phoneRegex = /^(\+91|91)?[6-9]\d{9}$/;
        return phoneRegex.test(phone.replace(/\s+/g, ''));
    }

    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    isStrongPassword(password) {
        // At least 8 characters, 1 uppercase, 1 lowercase, 1 number, 1 special char
        const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
        return passwordRegex.test(password);
    }

    maskPhone(phone) {
        if (phone.length <= 4) return phone;
        return phone.slice(0, -4).replace(/\d/g, '*') + phone.slice(-4);
    }
}

module.exports = new AuthService();
