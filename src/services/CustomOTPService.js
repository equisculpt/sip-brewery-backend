/**
 * 📱 Custom OTP Service - Self-Hosted SMS & Email
 * Ultra-secure, cost-effective OTP delivery without third-party dependencies
 */

const crypto = require('crypto');
const https = require('https');
const http = require('http');
const { Pool } = require('pg');
const validator = require('validator');

class CustomOTPService {
    constructor() {
        this.db = new Pool({
            connectionString: process.env.DATABASE_URL,
            ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
        });

        // OTP Configuration
        this.config = {
            sms: {
                length: 6,
                expiry: 5 * 60 * 1000, // 5 minutes
                maxAttempts: 3,
                rateLimitWindow: 60 * 1000, // 1 minute
                maxSMSPerWindow: 3,
                resendCooldown: 60 * 1000 // 1 minute
            },
            email: {
                length: 6,
                expiry: 10 * 60 * 1000, // 10 minutes
                maxAttempts: 5,
                rateLimitWindow: 60 * 1000,
                maxEmailsPerWindow: 3,
                resendCooldown: 60 * 1000
            },
            security: {
                hashAlgorithm: 'sha256',
                encryptionAlgorithm: 'aes-256-gcm',
                saltRounds: 12
            }
        };

        // SMS Provider Configurations (Multiple providers for redundancy)
        this.smsProviders = {
            // Primary: Direct Telecom APIs (Indian providers)
            primary: {
                name: 'BSNL_API',
                endpoint: 'https://bulksms.bsnl.in/api/send',
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.BSNL_API_KEY}`
                },
                formatPayload: (phone, message) => ({
                    to: phone,
                    message: message,
                    sender_id: process.env.SMS_SENDER_ID || 'SIPBREW',
                    route: 'transactional'
                })
            },

            // Backup: Airtel Business SMS
            backup1: {
                name: 'AIRTEL_BUSINESS',
                endpoint: 'https://api.airtel.in/sms/send',
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': process.env.AIRTEL_API_KEY
                },
                formatPayload: (phone, message) => ({
                    mobile: phone,
                    text: message,
                    sender: process.env.SMS_SENDER_ID || 'SIPBREW'
                })
            },

            // Backup: JIO Business SMS
            backup2: {
                name: 'JIO_BUSINESS',
                endpoint: 'https://jioapi.jio.com/sms/v1/send',
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Api-Key ${process.env.JIO_API_KEY}`
                },
                formatPayload: (phone, message) => ({
                    destination: phone,
                    message: message,
                    source: process.env.SMS_SENDER_ID || 'SIPBREW'
                })
            },

            // Fallback: HTTP SMS Gateway (Local telecom provider)
            fallback: {
                name: 'HTTP_GATEWAY',
                endpoint: process.env.SMS_GATEWAY_URL || 'http://localhost:8080/send-sms',
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.SMS_GATEWAY_TOKEN}`
                },
                formatPayload: (phone, message) => ({
                    phone: phone,
                    message: message,
                    sender: process.env.SMS_SENDER_ID || 'SIPBREW'
                })
            }
        };

        // Email Configuration (Self-hosted SMTP)
        this.emailConfig = {
            smtp: {
                host: process.env.SMTP_HOST || 'localhost',
                port: parseInt(process.env.SMTP_PORT) || 587,
                secure: process.env.SMTP_SECURE === 'true',
                auth: {
                    user: process.env.SMTP_USER,
                    pass: process.env.SMTP_PASS
                }
            },
            from: process.env.FROM_EMAIL || 'otp@sipbrewery.com',
            templates: {
                subject: '🔐 Your SIP Brewery Verification Code',
                html: this.getEmailTemplate(),
                text: 'Your SIP Brewery verification code is: {{OTP}}. Valid for {{EXPIRY}} minutes.'
            }
        };

        // Rate limiting and caching
        this.rateLimitStore = new Map();
        this.otpCache = new Map();

        // Initialize cleanup intervals
        this.startCleanupIntervals();
    }

    /**
     * Send OTP via SMS
     */
    async sendSMSOTP(phoneNumber, purpose = 'LOGIN', userId = null) {
        try {
            // Validate phone number
            const normalizedPhone = this.normalizePhoneNumber(phoneNumber);
            if (!this.isValidPhoneNumber(normalizedPhone)) {
                throw new Error('Invalid phone number format');
            }

            // Check rate limiting
            await this.checkRateLimit('sms', normalizedPhone);

            // Generate secure OTP
            const otp = this.generateSecureOTP(this.config.sms.length);
            const hashedOTP = this.hashOTP(otp);
            const expiresAt = new Date(Date.now() + this.config.sms.expiry);

            // Store OTP in database
            const otpId = await this.storeOTP({
                type: 'SMS',
                recipient: normalizedPhone,
                hashedOTP,
                purpose,
                userId,
                expiresAt
            });

            // Create SMS message
            const message = this.formatSMSMessage(otp, purpose);

            // Send SMS with fallback providers
            const deliveryResult = await this.sendSMSWithFallback(normalizedPhone, message);

            // Log success
            await this.logOTPEvent(userId, 'SMS_OTP_SENT', 'INFO', {
                phone: this.maskPhoneNumber(normalizedPhone),
                purpose,
                otpId,
                provider: deliveryResult.provider
            });

            return {
                success: true,
                otpId,
                message: 'SMS OTP sent successfully',
                expiresAt,
                provider: deliveryResult.provider
            };

        } catch (error) {
            await this.logOTPEvent(userId, 'SMS_OTP_SEND_FAILED', 'ERROR', {
                phone: this.maskPhoneNumber(phoneNumber),
                error: error.message
            });
            throw error;
        }
    }

    /**
     * Send OTP via Email
     */
    async sendEmailOTP(email, purpose = 'EMAIL_VERIFICATION', userId = null) {
        try {
            // Validate email
            if (!validator.isEmail(email)) {
                throw new Error('Invalid email address');
            }

            // Check rate limiting
            await this.checkRateLimit('email', email);

            // Generate secure OTP
            const otp = this.generateSecureOTP(this.config.email.length);
            const hashedOTP = this.hashOTP(otp);
            const expiresAt = new Date(Date.now() + this.config.email.expiry);

            // Store OTP in database
            const otpId = await this.storeOTP({
                type: 'EMAIL',
                recipient: email,
                hashedOTP,
                purpose,
                userId,
                expiresAt
            });

            // Send email
            await this.sendEmailWithSMTP(email, otp, purpose);

            // Log success
            await this.logOTPEvent(userId, 'EMAIL_OTP_SENT', 'INFO', {
                email: this.maskEmail(email),
                purpose,
                otpId
            });

            return {
                success: true,
                otpId,
                message: 'Email OTP sent successfully',
                expiresAt
            };

        } catch (error) {
            await this.logOTPEvent(userId, 'EMAIL_OTP_SEND_FAILED', 'ERROR', {
                email: this.maskEmail(email),
                error: error.message
            });
            throw error;
        }
    }

    /**
     * Verify OTP
     */
    async verifyOTP(otpId, providedOTP, recipient) {
        try {
            // Get OTP record
            const otpRecord = await this.getOTPRecord(otpId, recipient);
            
            if (!otpRecord) {
                throw new Error('OTP record not found');
            }

            // Check if expired
            if (new Date() > otpRecord.expires_at) {
                await this.markOTPExpired(otpId);
                throw new Error('OTP has expired');
            }

            // Check if already verified
            if (otpRecord.is_verified) {
                throw new Error('OTP has already been used');
            }

            // Check attempts
            if (otpRecord.attempts >= this.getMaxAttempts(otpRecord.otp_type)) {
                await this.lockOTP(otpId);
                throw new Error('Maximum verification attempts exceeded');
            }

            // Verify OTP using timing-safe comparison
            const hashedProvidedOTP = this.hashOTP(providedOTP);
            const isValid = this.verifyOTPHash(hashedProvidedOTP, otpRecord.hashed_otp);

            // Update attempts
            await this.updateOTPAttempts(otpId, otpRecord.attempts + 1);

            if (!isValid) {
                await this.logOTPEvent(otpRecord.user_id, 'OTP_VERIFICATION_FAILED', 'WARNING', {
                    otpId,
                    attempts: otpRecord.attempts + 1,
                    type: otpRecord.otp_type
                });
                throw new Error('Invalid OTP');
            }

            // Mark as verified
            await this.markOTPVerified(otpId);

            // Log success
            await this.logOTPEvent(otpRecord.user_id, 'OTP_VERIFIED', 'INFO', {
                otpId,
                purpose: otpRecord.purpose,
                type: otpRecord.otp_type
            });

            return {
                success: true,
                userId: otpRecord.user_id,
                purpose: otpRecord.purpose,
                message: 'OTP verified successfully'
            };

        } catch (error) {
            throw error;
        }
    }

    /**
     * Generate cryptographically secure OTP
     */
    generateSecureOTP(length = 6) {
        // Use crypto.randomInt for cryptographically secure random numbers
        let otp = '';
        for (let i = 0; i < length; i++) {
            otp += crypto.randomInt(0, 10).toString();
        }
        return otp;
    }

    /**
     * Hash OTP for secure storage
     */
    hashOTP(otp) {
        const salt = process.env.OTP_SALT || 'default_otp_salt_change_in_production';
        return crypto
            .createHash(this.config.security.hashAlgorithm)
            .update(otp + salt)
            .digest('hex');
    }

    /**
     * Verify OTP hash using timing-safe comparison
     */
    verifyOTPHash(providedHash, storedHash) {
        try {
            return crypto.timingSafeEqual(
                Buffer.from(providedHash, 'hex'),
                Buffer.from(storedHash, 'hex')
            );
        } catch (error) {
            return false;
        }
    }

    /**
     * Send SMS with fallback providers
     */
    async sendSMSWithFallback(phoneNumber, message) {
        const providers = ['primary', 'backup1', 'backup2', 'fallback'];
        let lastError;

        for (const providerKey of providers) {
            const provider = this.smsProviders[providerKey];
            if (!provider) continue;

            try {
                const result = await this.sendSMSViaProvider(provider, phoneNumber, message);
                console.log(`✅ SMS sent via ${provider.name}: ${result.messageId || 'Success'}`);
                
                return {
                    success: true,
                    provider: provider.name,
                    messageId: result.messageId
                };
            } catch (error) {
                console.log(`❌ SMS failed via ${provider.name}:`, error.message);
                lastError = error;
                
                // Wait before trying next provider
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }

        throw new Error(`All SMS providers failed. Last error: ${lastError?.message}`);
    }

    /**
     * Send SMS via specific provider
     */
    async sendSMSViaProvider(provider, phoneNumber, message) {
        return new Promise((resolve, reject) => {
            const payload = provider.formatPayload(phoneNumber, message);
            const postData = JSON.stringify(payload);

            const url = new URL(provider.endpoint);
            const options = {
                hostname: url.hostname,
                port: url.port || (url.protocol === 'https:' ? 443 : 80),
                path: url.pathname + url.search,
                method: provider.method,
                headers: {
                    ...provider.headers,
                    'Content-Length': Buffer.byteLength(postData)
                }
            };

            const client = url.protocol === 'https:' ? https : http;
            
            const req = client.request(options, (res) => {
                let data = '';
                
                res.on('data', (chunk) => {
                    data += chunk;
                });
                
                res.on('end', () => {
                    if (res.statusCode >= 200 && res.statusCode < 300) {
                        try {
                            const response = JSON.parse(data);
                            resolve(response);
                        } catch (error) {
                            resolve({ success: true, raw: data });
                        }
                    } else {
                        reject(new Error(`HTTP ${res.statusCode}: ${data}`));
                    }
                });
            });

            req.on('error', (error) => {
                reject(error);
            });

            req.on('timeout', () => {
                req.destroy();
                reject(new Error('Request timeout'));
            });

            req.setTimeout(10000); // 10 second timeout
            req.write(postData);
            req.end();
        });
    }

    /**
     * Send email via SMTP
     */
    async sendEmailWithSMTP(email, otp, purpose) {
        const nodemailer = require('nodemailer');
        
        // Create transporter
        const transporter = nodemailer.createTransporter(this.emailConfig.smtp);

        // Prepare email content
        const subject = this.emailConfig.templates.subject;
        const html = this.emailConfig.templates.html
            .replace('{{OTP}}', otp)
            .replace('{{EXPIRY}}', Math.floor(this.config.email.expiry / (60 * 1000)))
            .replace('{{PURPOSE}}', purpose);
        
        const text = this.emailConfig.templates.text
            .replace('{{OTP}}', otp)
            .replace('{{EXPIRY}}', Math.floor(this.config.email.expiry / (60 * 1000)));

        // Send email
        const result = await transporter.sendMail({
            from: this.emailConfig.from,
            to: email,
            subject,
            html,
            text
        });

        return result;
    }

    /**
     * Format SMS message
     */
    formatSMSMessage(otp, purpose) {
        const templates = {
            'LOGIN': `Your SIP Brewery login code is ${otp}. Valid for 5 minutes. Don't share this code.`,
            'SIGNUP': `Welcome to SIP Brewery! Your verification code is ${otp}. Valid for 5 minutes.`,
            'PASSWORD_RESET': `Your SIP Brewery password reset code is ${otp}. Valid for 5 minutes.`,
            'TRANSACTION': `Your SIP Brewery transaction OTP is ${otp}. Valid for 5 minutes.`,
            'TWO_FACTOR': `Your SIP Brewery 2FA code is ${otp}. Valid for 5 minutes.`
        };

        return templates[purpose] || `Your SIP Brewery verification code is ${otp}. Valid for 5 minutes.`;
    }

    /**
     * Get email template
     */
    getEmailTemplate() {
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px 20px; text-align: center; color: white; }
                .content { padding: 30px 20px; }
                .otp-box { background-color: #f8f9fa; border: 2px dashed #667eea; border-radius: 8px; padding: 20px; text-align: center; margin: 20px 0; }
                .otp-code { font-size: 32px; font-weight: bold; color: #667eea; letter-spacing: 4px; margin: 10px 0; }
                .footer { background-color: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 14px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔐 SIP Brewery</h1>
                    <p>Secure Investment Platform</p>
                </div>
                <div class="content">
                    <h2>Your Verification Code</h2>
                    <p>Use the following code to complete your verification:</p>
                    <div class="otp-box">
                        <div class="otp-code">{{OTP}}</div>
                        <p style="margin: 0; color: #888;">Valid for {{EXPIRY}} minutes</p>
                    </div>
                    <p><strong>Security Notice:</strong> Never share this code with anyone. SIP Brewery will never ask for your OTP via phone or email.</p>
                </div>
                <div class="footer">
                    <p>© 2024 SIP Brewery. All rights reserved.</p>
                    <p>This is an automated message. Please do not reply.</p>
                </div>
            </div>
        </body>
        </html>
        `;
    }

    /**
     * Store OTP in database
     */
    async storeOTP({ type, recipient, hashedOTP, purpose, userId, expiresAt }) {
        const query = `
            INSERT INTO custom_otps (
                otp_id, otp_type, recipient, hashed_otp, 
                purpose, user_id, expires_at, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            RETURNING otp_id
        `;

        const otpId = crypto.randomUUID();
        
        await this.db.query(query, [
            otpId, type, recipient, hashedOTP, 
            purpose, userId, expiresAt
        ]);

        return otpId;
    }

    /**
     * Get OTP record
     */
    async getOTPRecord(otpId, recipient) {
        const query = `
            SELECT * FROM custom_otps 
            WHERE otp_id = $1 AND recipient = $2 AND is_active = true
        `;

        const result = await this.db.query(query, [otpId, recipient]);
        return result.rows[0];
    }

    /**
     * Update OTP attempts
     */
    async updateOTPAttempts(otpId, attempts) {
        const query = `
            UPDATE custom_otps 
            SET attempts = $2, last_attempt = NOW()
            WHERE otp_id = $1
        `;

        await this.db.query(query, [otpId, attempts]);
    }

    /**
     * Mark OTP as verified
     */
    async markOTPVerified(otpId) {
        const query = `
            UPDATE custom_otps 
            SET is_verified = true, verified_at = NOW()
            WHERE otp_id = $1
        `;

        await this.db.query(query, [otpId]);
    }

    /**
     * Mark OTP as expired
     */
    async markOTPExpired(otpId) {
        const query = `
            UPDATE custom_otps 
            SET is_active = false, status = 'EXPIRED'
            WHERE otp_id = $1
        `;

        await this.db.query(query, [otpId]);
    }

    /**
     * Lock OTP due to too many attempts
     */
    async lockOTP(otpId) {
        const query = `
            UPDATE custom_otps 
            SET is_locked = true, status = 'LOCKED'
            WHERE otp_id = $1
        `;

        await this.db.query(query, [otpId]);
    }

    /**
     * Normalize phone number
     */
    normalizePhoneNumber(phoneNumber) {
        // Remove all non-digit characters
        let normalized = phoneNumber.replace(/\D/g, '');
        
        // Add country code if missing (assuming India +91)
        if (normalized.length === 10) {
            normalized = '91' + normalized;
        }
        
        return normalized;
    }

    /**
     * Validate phone number
     */
    isValidPhoneNumber(phoneNumber) {
        // Indian phone number validation
        const indianPhoneRegex = /^91[6-9]\d{9}$/;
        return indianPhoneRegex.test(phoneNumber);
    }

    /**
     * Check rate limiting
     */
    async checkRateLimit(type, recipient) {
        const now = Date.now();
        const config = this.config[type];
        const windowStart = now - config.rateLimitWindow;
        
        // Clean old entries
        for (const [key, data] of this.rateLimitStore.entries()) {
            if (data.timestamp < windowStart) {
                this.rateLimitStore.delete(key);
            }
        }

        // Check current rate limit
        const key = `${type}:${recipient}`;
        const data = this.rateLimitStore.get(key) || { count: 0, timestamp: now };

        const maxPerWindow = type === 'sms' ? config.maxSMSPerWindow : config.maxEmailsPerWindow;
        
        if (data.count >= maxPerWindow) {
            throw new Error(`Rate limit exceeded. Please try again later.`);
        }

        // Update rate limit
        data.count++;
        data.timestamp = now;
        this.rateLimitStore.set(key, data);
    }

    /**
     * Get max attempts based on OTP type
     */
    getMaxAttempts(type) {
        return type === 'SMS' ? this.config.sms.maxAttempts : this.config.email.maxAttempts;
    }

    /**
     * Mask phone number for logging
     */
    maskPhoneNumber(phoneNumber) {
        if (phoneNumber.length > 6) {
            return phoneNumber.substring(0, 4) + '*'.repeat(phoneNumber.length - 6) + phoneNumber.slice(-2);
        }
        return phoneNumber;
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
     * Log OTP events
     */
    async logOTPEvent(userId, eventType, severity, metadata) {
        try {
            const query = `
                INSERT INTO security_events (
                    event_type, event_category, severity, description,
                    metadata, user_id, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
            `;

            await this.db.query(query, [
                eventType,
                'CUSTOM_OTP',
                severity,
                `${eventType} for user`,
                JSON.stringify(metadata),
                userId
            ]);
        } catch (error) {
            console.error('Error logging OTP event:', error);
        }
    }

    /**
     * Start cleanup intervals
     */
    startCleanupIntervals() {
        // Clean expired OTPs every 5 minutes
        setInterval(() => {
            this.cleanupExpiredOTPs();
        }, 5 * 60 * 1000);

        // Clean rate limit store every minute
        setInterval(() => {
            const now = Date.now();
            for (const [key, data] of this.rateLimitStore.entries()) {
                if (now - data.timestamp > 60 * 1000) {
                    this.rateLimitStore.delete(key);
                }
            }
        }, 60 * 1000);
    }

    /**
     * Cleanup expired OTPs
     */
    async cleanupExpiredOTPs() {
        try {
            const query = `
                UPDATE custom_otps 
                SET is_active = false, status = 'EXPIRED'
                WHERE expires_at < NOW() AND is_active = true
            `;

            const result = await this.db.query(query);
            if (result.rowCount > 0) {
                console.log(`🧹 Cleaned up ${result.rowCount} expired OTPs`);
            }
        } catch (error) {
            console.error('Error cleaning up expired OTPs:', error);
        }
    }

    /**
     * Get OTP statistics
     */
    async getOTPStats(timeRange = '24 hours') {
        const query = `
            SELECT 
                otp_type,
                purpose,
                COUNT(*) as total_sent,
                COUNT(CASE WHEN is_verified = true THEN 1 END) as verified,
                COUNT(CASE WHEN is_locked = true THEN 1 END) as locked,
                AVG(attempts) as avg_attempts,
                AVG(EXTRACT(EPOCH FROM (verified_at - created_at))) as avg_verification_time
            FROM custom_otps 
            WHERE created_at > NOW() - INTERVAL '${timeRange}'
            GROUP BY otp_type, purpose
            ORDER BY total_sent DESC
        `;

        const result = await this.db.query(query);
        return result.rows;
    }

    /**
     * Test SMS provider connectivity
     */
    async testSMSProviders() {
        const testPhone = process.env.TEST_PHONE_NUMBER || '919999999999';
        const testMessage = 'Test message from SIP Brewery OTP Service';
        
        const results = {};
        
        for (const [key, provider] of Object.entries(this.smsProviders)) {
            try {
                const result = await this.sendSMSViaProvider(provider, testPhone, testMessage);
                results[provider.name] = { status: 'SUCCESS', result };
            } catch (error) {
                results[provider.name] = { status: 'FAILED', error: error.message };
            }
        }
        
        return results;
    }
}

module.exports = new CustomOTPService();
