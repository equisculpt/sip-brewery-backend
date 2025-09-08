/**
 * 📧 Email Verification Service - Enterprise Grade
 * Secure email verification with OTP and magic links
 */

const crypto = require('crypto');
const nodemailer = require('nodemailer');
const { Pool } = require('pg');
const validator = require('validator');

class EmailVerificationService {
    constructor() {
        this.db = new Pool({
            connectionString: process.env.DATABASE_URL,
            ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
        });

        // Email configuration
        this.emailConfig = {
            // Primary email provider (SendGrid)
            primary: {
                service: 'SendGrid',
                auth: {
                    user: 'apikey',
                    pass: process.env.SENDGRID_API_KEY
                }
            },
            
            // Backup email provider (AWS SES)
            backup: {
                host: 'email-smtp.ap-south-1.amazonaws.com',
                port: 587,
                secure: false,
                auth: {
                    user: process.env.AWS_SES_ACCESS_KEY,
                    pass: process.env.AWS_SES_SECRET_KEY
                }
            },
            
            // SMTP fallback
            smtp: {
                host: process.env.SMTP_HOST,
                port: parseInt(process.env.SMTP_PORT) || 587,
                secure: false,
                auth: {
                    user: process.env.SMTP_USER,
                    pass: process.env.SMTP_PASS
                }
            }
        };

        // Verification settings
        this.settings = {
            otpLength: 6,
            otpExpiry: 10 * 60 * 1000, // 10 minutes
            magicLinkExpiry: 30 * 60 * 1000, // 30 minutes
            maxAttempts: 5,
            rateLimitWindow: 60 * 1000, // 1 minute
            maxEmailsPerWindow: 3,
            resendCooldown: 60 * 1000 // 1 minute
        };

        // Initialize email transporter
        this.transporter = null;
        this.initializeTransporter();

        // Rate limiting storage
        this.rateLimitStore = new Map();
    }

    /**
     * Initialize email transporter with fallback
     */
    async initializeTransporter() {
        try {
            // Try primary provider first
            this.transporter = nodemailer.createTransporter(this.emailConfig.primary);
            await this.transporter.verify();
            console.log('✅ Primary email provider (SendGrid) connected');
        } catch (error) {
            console.log('⚠️ Primary email provider failed, trying backup...');
            
            try {
                // Try backup provider
                this.transporter = nodemailer.createTransporter(this.emailConfig.backup);
                await this.transporter.verify();
                console.log('✅ Backup email provider (AWS SES) connected');
            } catch (backupError) {
                console.log('⚠️ Backup email provider failed, using SMTP...');
                
                // Fallback to SMTP
                this.transporter = nodemailer.createTransporter(this.emailConfig.smtp);
                console.log('✅ SMTP email provider connected');
            }
        }
    }

    /**
     * Send email verification OTP
     */
    async sendVerificationOTP(email, userId = null, purpose = 'EMAIL_VERIFICATION') {
        try {
            // Validate email
            if (!validator.isEmail(email)) {
                throw new Error('Invalid email address');
            }

            // Check rate limiting
            await this.checkRateLimit(email);

            // Generate secure OTP
            const otp = this.generateSecureOTP();
            const hashedOTP = this.hashOTP(otp);
            const expiresAt = new Date(Date.now() + this.settings.otpExpiry);

            // Store OTP in database
            const verificationId = await this.storeEmailVerification({
                email,
                userId,
                hashedOTP,
                purpose,
                expiresAt,
                type: 'OTP'
            });

            // Send email
            await this.sendOTPEmail(email, otp, purpose);

            // Log security event
            await this.logSecurityEvent(userId, 'EMAIL_OTP_SENT', 'INFO', {
                email: this.maskEmail(email),
                purpose,
                verificationId
            });

            return {
                success: true,
                verificationId,
                message: 'Verification OTP sent successfully',
                expiresAt
            };

        } catch (error) {
            await this.logSecurityEvent(userId, 'EMAIL_OTP_SEND_FAILED', 'ERROR', {
                email: this.maskEmail(email),
                error: error.message
            });
            throw error;
        }
    }

    /**
     * Verify email OTP
     */
    async verifyEmailOTP(email, otp, verificationId) {
        try {
            // Validate inputs
            if (!validator.isEmail(email)) {
                throw new Error('Invalid email address');
            }

            if (!otp || otp.length !== this.settings.otpLength) {
                throw new Error('Invalid OTP format');
            }

            // Get verification record
            const verification = await this.getEmailVerification(verificationId, email);
            
            if (!verification) {
                throw new Error('Verification record not found');
            }

            // Check if expired
            if (new Date() > verification.expires_at) {
                await this.markVerificationExpired(verificationId);
                throw new Error('OTP has expired');
            }

            // Check if already used
            if (verification.is_verified) {
                throw new Error('OTP has already been used');
            }

            // Check attempts
            if (verification.attempts >= this.settings.maxAttempts) {
                await this.lockVerification(verificationId);
                throw new Error('Maximum verification attempts exceeded');
            }

            // Verify OTP
            const hashedOTP = this.hashOTP(otp);
            const isValid = this.verifyOTPHash(hashedOTP, verification.hashed_code);

            // Update attempts
            await this.updateVerificationAttempts(verificationId, verification.attempts + 1);

            if (!isValid) {
                await this.logSecurityEvent(verification.user_id, 'EMAIL_OTP_VERIFICATION_FAILED', 'WARNING', {
                    email: this.maskEmail(email),
                    attempts: verification.attempts + 1,
                    verificationId
                });
                throw new Error('Invalid OTP');
            }

            // Mark as verified
            await this.markVerificationSuccess(verificationId);

            // Update user email verification status
            if (verification.user_id && verification.purpose === 'EMAIL_VERIFICATION') {
                await this.updateUserEmailVerification(verification.user_id, email);
            }

            await this.logSecurityEvent(verification.user_id, 'EMAIL_OTP_VERIFIED', 'INFO', {
                email: this.maskEmail(email),
                purpose: verification.purpose,
                verificationId
            });

            return {
                success: true,
                userId: verification.user_id,
                purpose: verification.purpose,
                message: 'Email verified successfully'
            };

        } catch (error) {
            throw error;
        }
    }

    /**
     * Send magic link for email verification
     */
    async sendMagicLink(email, userId = null, purpose = 'EMAIL_VERIFICATION') {
        try {
            // Validate email
            if (!validator.isEmail(email)) {
                throw new Error('Invalid email address');
            }

            // Check rate limiting
            await this.checkRateLimit(email);

            // Generate secure token
            const token = this.generateSecureToken();
            const hashedToken = this.hashToken(token);
            const expiresAt = new Date(Date.now() + this.settings.magicLinkExpiry);

            // Store verification record
            const verificationId = await this.storeEmailVerification({
                email,
                userId,
                hashedOTP: hashedToken,
                purpose,
                expiresAt,
                type: 'MAGIC_LINK'
            });

            // Create magic link
            const magicLink = `${process.env.FRONTEND_URL}/verify-email?token=${token}&id=${verificationId}`;

            // Send email
            await this.sendMagicLinkEmail(email, magicLink, purpose);

            await this.logSecurityEvent(userId, 'EMAIL_MAGIC_LINK_SENT', 'INFO', {
                email: this.maskEmail(email),
                purpose,
                verificationId
            });

            return {
                success: true,
                verificationId,
                message: 'Magic link sent successfully',
                expiresAt
            };

        } catch (error) {
            await this.logSecurityEvent(userId, 'EMAIL_MAGIC_LINK_SEND_FAILED', 'ERROR', {
                email: this.maskEmail(email),
                error: error.message
            });
            throw error;
        }
    }

    /**
     * Verify magic link token
     */
    async verifyMagicLink(token, verificationId) {
        try {
            if (!token || !verificationId) {
                throw new Error('Invalid verification parameters');
            }

            // Get verification record
            const verification = await this.getEmailVerificationById(verificationId);
            
            if (!verification) {
                throw new Error('Verification record not found');
            }

            // Check if expired
            if (new Date() > verification.expires_at) {
                await this.markVerificationExpired(verificationId);
                throw new Error('Magic link has expired');
            }

            // Check if already used
            if (verification.is_verified) {
                throw new Error('Magic link has already been used');
            }

            // Verify token
            const hashedToken = this.hashToken(token);
            const isValid = this.verifyOTPHash(hashedToken, verification.hashed_code);

            if (!isValid) {
                await this.logSecurityEvent(verification.user_id, 'EMAIL_MAGIC_LINK_VERIFICATION_FAILED', 'WARNING', {
                    email: this.maskEmail(verification.email),
                    verificationId
                });
                throw new Error('Invalid magic link');
            }

            // Mark as verified
            await this.markVerificationSuccess(verificationId);

            // Update user email verification status
            if (verification.user_id && verification.purpose === 'EMAIL_VERIFICATION') {
                await this.updateUserEmailVerification(verification.user_id, verification.email);
            }

            await this.logSecurityEvent(verification.user_id, 'EMAIL_MAGIC_LINK_VERIFIED', 'INFO', {
                email: this.maskEmail(verification.email),
                purpose: verification.purpose,
                verificationId
            });

            return {
                success: true,
                userId: verification.user_id,
                email: verification.email,
                purpose: verification.purpose,
                message: 'Email verified successfully'
            };

        } catch (error) {
            throw error;
        }
    }

    /**
     * Generate secure OTP
     */
    generateSecureOTP() {
        // Use cryptographically secure random number generation
        const buffer = crypto.randomBytes(4);
        const number = buffer.readUInt32BE(0);
        const otp = (number % Math.pow(10, this.settings.otpLength))
            .toString()
            .padStart(this.settings.otpLength, '0');
        
        return otp;
    }

    /**
     * Generate secure token for magic links
     */
    generateSecureToken() {
        return crypto.randomBytes(32).toString('hex');
    }

    /**
     * Hash OTP/token for secure storage
     */
    hashOTP(otp) {
        return crypto
            .createHash('sha256')
            .update(otp + process.env.OTP_SALT || 'default_salt')
            .digest('hex');
    }

    /**
     * Hash token for secure storage
     */
    hashToken(token) {
        return crypto
            .createHash('sha256')
            .update(token + process.env.TOKEN_SALT || 'default_salt')
            .digest('hex');
    }

    /**
     * Verify OTP hash using timing-safe comparison
     */
    verifyOTPHash(providedHash, storedHash) {
        return crypto.timingSafeEqual(
            Buffer.from(providedHash, 'hex'),
            Buffer.from(storedHash, 'hex')
        );
    }

    /**
     * Send OTP email
     */
    async sendOTPEmail(email, otp, purpose) {
        const subject = this.getEmailSubject(purpose);
        const template = this.getOTPEmailTemplate(otp, purpose);

        const mailOptions = {
            from: `"SIP Brewery Security" <${process.env.FROM_EMAIL || 'security@sipbrewery.com'}>`,
            to: email,
            subject,
            html: template,
            text: `Your verification code is: ${otp}. This code will expire in 10 minutes.`
        };

        await this.sendEmailWithRetry(mailOptions);
    }

    /**
     * Send magic link email
     */
    async sendMagicLinkEmail(email, magicLink, purpose) {
        const subject = this.getEmailSubject(purpose);
        const template = this.getMagicLinkEmailTemplate(magicLink, purpose);

        const mailOptions = {
            from: `"SIP Brewery Security" <${process.env.FROM_EMAIL || 'security@sipbrewery.com'}>`,
            to: email,
            subject,
            html: template,
            text: `Click this link to verify your email: ${magicLink}`
        };

        await this.sendEmailWithRetry(mailOptions);
    }

    /**
     * Send email with retry logic
     */
    async sendEmailWithRetry(mailOptions, maxRetries = 3) {
        let lastError;

        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                const result = await this.transporter.sendMail(mailOptions);
                console.log(`✅ Email sent successfully: ${result.messageId}`);
                return result;
            } catch (error) {
                console.log(`❌ Email send attempt ${attempt} failed:`, error.message);
                lastError = error;

                if (attempt < maxRetries) {
                    // Wait before retry (exponential backoff)
                    await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
                    
                    // Try to reinitialize transporter
                    await this.initializeTransporter();
                }
            }
        }

        throw new Error(`Failed to send email after ${maxRetries} attempts: ${lastError.message}`);
    }

    /**
     * Get email subject based on purpose
     */
    getEmailSubject(purpose) {
        const subjects = {
            'EMAIL_VERIFICATION': '🔐 Verify Your Email - SIP Brewery',
            'PASSWORD_RESET': '🔑 Reset Your Password - SIP Brewery',
            'LOGIN_VERIFICATION': '🛡️ Login Verification - SIP Brewery',
            'ACCOUNT_SECURITY': '⚠️ Account Security Alert - SIP Brewery',
            'TWO_FACTOR_AUTH': '🔒 Two-Factor Authentication - SIP Brewery'
        };

        return subjects[purpose] || '🔐 Email Verification - SIP Brewery';
    }

    /**
     * Get OTP email template
     */
    getOTPEmailTemplate(otp, purpose) {
        const purposeText = {
            'EMAIL_VERIFICATION': 'verify your email address',
            'PASSWORD_RESET': 'reset your password',
            'LOGIN_VERIFICATION': 'verify your login',
            'ACCOUNT_SECURITY': 'secure your account',
            'TWO_FACTOR_AUTH': 'complete two-factor authentication'
        };

        return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Email Verification</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }
                .container { max-width: 600px; margin: 0 auto; background-color: #ffffff; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 20px; text-align: center; }
                .header h1 { color: #ffffff; margin: 0; font-size: 28px; font-weight: 300; }
                .content { padding: 40px 20px; }
                .otp-box { background-color: #f8f9fa; border: 2px dashed #667eea; border-radius: 10px; padding: 30px; text-align: center; margin: 30px 0; }
                .otp-code { font-size: 36px; font-weight: bold; color: #667eea; letter-spacing: 8px; margin: 20px 0; }
                .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 15px; margin: 20px 0; color: #856404; }
                .footer { background-color: #f8f9fa; padding: 20px; text-align: center; color: #6c757d; font-size: 14px; }
                .security-tips { background-color: #e7f3ff; border-left: 4px solid #2196F3; padding: 15px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔐 SIP Brewery</h1>
                    <p style="color: #ffffff; margin: 10px 0 0 0; opacity: 0.9;">Secure Investment Platform</p>
                </div>
                
                <div class="content">
                    <h2 style="color: #333; margin-bottom: 20px;">Email Verification Required</h2>
                    
                    <p style="color: #555; line-height: 1.6;">
                        Hello! We need to ${purposeText[purpose] || 'verify your email'}. 
                        Please use the verification code below:
                    </p>
                    
                    <div class="otp-box">
                        <p style="margin: 0; color: #667eea; font-weight: 600;">Your Verification Code</p>
                        <div class="otp-code">${otp}</div>
                        <p style="margin: 0; color: #888; font-size: 14px;">Valid for 10 minutes</p>
                    </div>
                    
                    <div class="warning">
                        <strong>⚠️ Security Notice:</strong> Never share this code with anyone. 
                        SIP Brewery will never ask for your verification code via phone or email.
                    </div>
                    
                    <div class="security-tips">
                        <h4 style="margin-top: 0; color: #2196F3;">🛡️ Security Tips:</h4>
                        <ul style="margin: 10px 0; padding-left: 20px; color: #555;">
                            <li>This code expires in 10 minutes</li>
                            <li>Only enter this code on the SIP Brewery website</li>
                            <li>If you didn't request this, please ignore this email</li>
                        </ul>
                    </div>
                    
                    <p style="color: #555; line-height: 1.6; margin-top: 30px;">
                        If you're having trouble, please contact our support team at 
                        <a href="mailto:support@sipbrewery.com" style="color: #667eea;">support@sipbrewery.com</a>
                    </p>
                </div>
                
                <div class="footer">
                    <p>© 2024 SIP Brewery. All rights reserved.</p>
                    <p>This is an automated security email. Please do not reply.</p>
                </div>
            </div>
        </body>
        </html>
        `;
    }

    /**
     * Get magic link email template
     */
    getMagicLinkEmailTemplate(magicLink, purpose) {
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Email Verification</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }
                .container { max-width: 600px; margin: 0 auto; background-color: #ffffff; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 20px; text-align: center; }
                .header h1 { color: #ffffff; margin: 0; font-size: 28px; font-weight: 300; }
                .content { padding: 40px 20px; }
                .button { display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff; text-decoration: none; padding: 15px 30px; border-radius: 5px; font-weight: 600; margin: 20px 0; }
                .button:hover { opacity: 0.9; }
                .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 15px; margin: 20px 0; color: #856404; }
                .footer { background-color: #f8f9fa; padding: 20px; text-align: center; color: #6c757d; font-size: 14px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔐 SIP Brewery</h1>
                    <p style="color: #ffffff; margin: 10px 0 0 0; opacity: 0.9;">Secure Investment Platform</p>
                </div>
                
                <div class="content">
                    <h2 style="color: #333; margin-bottom: 20px;">Verify Your Email</h2>
                    
                    <p style="color: #555; line-height: 1.6;">
                        Click the button below to verify your email address and complete your account setup:
                    </p>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="${magicLink}" class="button">✅ Verify Email Address</a>
                    </div>
                    
                    <p style="color: #888; font-size: 14px; line-height: 1.6;">
                        If the button doesn't work, copy and paste this link into your browser:<br>
                        <a href="${magicLink}" style="color: #667eea; word-break: break-all;">${magicLink}</a>
                    </p>
                    
                    <div class="warning">
                        <strong>⚠️ Security Notice:</strong> This link will expire in 30 minutes. 
                        If you didn't request this verification, please ignore this email.
                    </div>
                </div>
                
                <div class="footer">
                    <p>© 2024 SIP Brewery. All rights reserved.</p>
                    <p>This is an automated security email. Please do not reply.</p>
                </div>
            </div>
        </body>
        </html>
        `;
    }

    /**
     * Store email verification record
     */
    async storeEmailVerification({ email, userId, hashedOTP, purpose, expiresAt, type }) {
        const query = `
            INSERT INTO email_verifications (
                verification_id, email, user_id, hashed_code, 
                purpose, verification_type, expires_at, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            RETURNING verification_id
        `;

        const verificationId = crypto.randomUUID();
        
        const result = await this.db.query(query, [
            verificationId, email, userId, hashedOTP, 
            purpose, type, expiresAt
        ]);

        return verificationId;
    }

    /**
     * Get email verification record
     */
    async getEmailVerification(verificationId, email) {
        const query = `
            SELECT * FROM email_verifications 
            WHERE verification_id = $1 AND email = $2 AND is_active = true
        `;

        const result = await this.db.query(query, [verificationId, email]);
        return result.rows[0];
    }

    /**
     * Get email verification by ID only
     */
    async getEmailVerificationById(verificationId) {
        const query = `
            SELECT * FROM email_verifications 
            WHERE verification_id = $1 AND is_active = true
        `;

        const result = await this.db.query(query, [verificationId]);
        return result.rows[0];
    }

    /**
     * Update verification attempts
     */
    async updateVerificationAttempts(verificationId, attempts) {
        const query = `
            UPDATE email_verifications 
            SET attempts = $2, last_attempt = NOW()
            WHERE verification_id = $1
        `;

        await this.db.query(query, [verificationId, attempts]);
    }

    /**
     * Mark verification as successful
     */
    async markVerificationSuccess(verificationId) {
        const query = `
            UPDATE email_verifications 
            SET is_verified = true, verified_at = NOW()
            WHERE verification_id = $1
        `;

        await this.db.query(query, [verificationId]);
    }

    /**
     * Mark verification as expired
     */
    async markVerificationExpired(verificationId) {
        const query = `
            UPDATE email_verifications 
            SET is_active = false, status = 'EXPIRED'
            WHERE verification_id = $1
        `;

        await this.db.query(query, [verificationId]);
    }

    /**
     * Lock verification due to too many attempts
     */
    async lockVerification(verificationId) {
        const query = `
            UPDATE email_verifications 
            SET is_locked = true, status = 'LOCKED'
            WHERE verification_id = $1
        `;

        await this.db.query(query, [verificationId]);
    }

    /**
     * Update user email verification status
     */
    async updateUserEmailVerification(userId, email) {
        const query = `
            UPDATE users 
            SET email_verified = true, email_verified_at = NOW()
            WHERE user_id = $1 AND email = $2
        `;

        await this.db.query(query, [userId, email]);
    }

    /**
     * Check rate limiting
     */
    async checkRateLimit(email) {
        const now = Date.now();
        const windowStart = now - this.settings.rateLimitWindow;
        
        // Clean old entries
        for (const [key, data] of this.rateLimitStore.entries()) {
            if (data.timestamp < windowStart) {
                this.rateLimitStore.delete(key);
            }
        }

        // Check current rate limit
        const emailKey = `email:${email}`;
        const emailData = this.rateLimitStore.get(emailKey) || { count: 0, timestamp: now };

        if (emailData.count >= this.settings.maxEmailsPerWindow) {
            throw new Error('Rate limit exceeded. Please try again later.');
        }

        // Update rate limit
        emailData.count++;
        emailData.timestamp = now;
        this.rateLimitStore.set(emailKey, emailData);
    }

    /**
     * Mask email for logging
     */
    maskEmail(email) {
        const [local, domain] = email.split('@');
        const maskedLocal = local.length > 2 
            ? local.substring(0, 2) + '*'.repeat(local.length - 2)
            : local;
        return `${maskedLocal}@${domain}`;
    }

    /**
     * Log security events
     */
    async logSecurityEvent(userId, eventType, severity, metadata) {
        try {
            const query = `
                INSERT INTO security_events (
                    event_type, event_category, severity, description,
                    metadata, user_id, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
            `;

            await this.db.query(query, [
                eventType,
                'EMAIL_VERIFICATION',
                severity,
                `${eventType} for user`,
                JSON.stringify(metadata),
                userId
            ]);
        } catch (error) {
            console.error('Error logging security event:', error);
        }
    }

    /**
     * Cleanup expired verifications (run periodically)
     */
    async cleanupExpiredVerifications() {
        const query = `
            UPDATE email_verifications 
            SET is_active = false, status = 'EXPIRED'
            WHERE expires_at < NOW() AND is_active = true
        `;

        const result = await this.db.query(query);
        console.log(`🧹 Cleaned up ${result.rowCount} expired email verifications`);
        
        return result.rowCount;
    }

    /**
     * Get verification statistics
     */
    async getVerificationStats(timeRange = '24 hours') {
        const query = `
            SELECT 
                purpose,
                verification_type,
                COUNT(*) as total_sent,
                COUNT(CASE WHEN is_verified = true THEN 1 END) as verified,
                COUNT(CASE WHEN is_locked = true THEN 1 END) as locked,
                AVG(attempts) as avg_attempts
            FROM email_verifications 
            WHERE created_at > NOW() - INTERVAL '${timeRange}'
            GROUP BY purpose, verification_type
            ORDER BY total_sent DESC
        `;

        const result = await this.db.query(query);
        return result.rows;
    }
}

module.exports = new EmailVerificationService();
