/**
 * 🔐 Enterprise-Grade OTP Service
 * 35+ years experience implementation with top-notch security
 */

const crypto = require('crypto');
const bcrypt = require('bcryptjs');
const axios = require('axios');
const { Pool } = require('pg');

class OTPService {
    constructor() {
        this.db = new Pool({
            connectionString: process.env.DATABASE_URL,
            ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
        });
        
        // OTP Configuration
        this.OTP_LENGTH = 6;
        this.OTP_EXPIRY_MINUTES = 5;
        this.MAX_ATTEMPTS = 3;
        this.RATE_LIMIT_WINDOW = 60; // seconds
        this.MAX_OTP_PER_WINDOW = 3;
        
        // SMS Provider Configuration
        this.smsProvider = process.env.SMS_PROVIDER || 'MSG91'; // MSG91, TWILIO, TEXTLOCAL
        this.initializeSMSProvider();
    }

    /**
     * Initialize SMS provider based on configuration
     */
    initializeSMSProvider() {
        switch (this.smsProvider) {
            case 'MSG91':
                this.smsConfig = {
                    authKey: process.env.MSG91_AUTH_KEY,
                    templateId: process.env.MSG91_TEMPLATE_ID,
                    senderId: process.env.MSG91_SENDER_ID || 'SIPBRY',
                    baseUrl: 'https://api.msg91.com/api/v5/otp'
                };
                break;
            case 'TWILIO':
                this.smsConfig = {
                    accountSid: process.env.TWILIO_ACCOUNT_SID,
                    authToken: process.env.TWILIO_AUTH_TOKEN,
                    fromNumber: process.env.TWILIO_PHONE_NUMBER
                };
                break;
            case 'TEXTLOCAL':
                this.smsConfig = {
                    apiKey: process.env.TEXTLOCAL_API_KEY,
                    sender: process.env.TEXTLOCAL_SENDER || 'SIPBRY',
                    baseUrl: 'https://api.textlocal.in/send'
                };
                break;
        }
    }

    /**
     * Generate cryptographically secure OTP
     */
    generateOTP() {
        const digits = '0123456789';
        let otp = '';
        
        for (let i = 0; i < this.OTP_LENGTH; i++) {
            const randomIndex = crypto.randomInt(0, digits.length);
            otp += digits[randomIndex];
        }
        
        return otp;
    }

    /**
     * Hash OTP for secure storage
     */
    async hashOTP(otp) {
        return await bcrypt.hash(otp, 10);
    }

    /**
     * Verify OTP against hash
     */
    async verifyOTPHash(otp, hash) {
        return await bcrypt.compare(otp, hash);
    }

    /**
     * Check rate limiting for OTP requests
     */
    async checkRateLimit(phone, ipAddress) {
        const query = `
            SELECT COUNT(*) as count
            FROM otp_verifications 
            WHERE (phone = $1 OR ip_address = $2)
            AND created_at > NOW() - INTERVAL '${this.RATE_LIMIT_WINDOW} seconds'
        `;
        
        const result = await this.db.query(query, [phone, ipAddress]);
        const count = parseInt(result.rows[0].count);
        
        if (count >= this.MAX_OTP_PER_WINDOW) {
            throw new Error(`Rate limit exceeded. Please wait ${this.RATE_LIMIT_WINDOW} seconds before requesting another OTP.`);
        }
        
        return true;
    }

    /**
     * Check if phone number is blocked due to failed attempts
     */
    async checkPhoneBlocked(phone) {
        const query = `
            SELECT blocked_until
            FROM failed_login_attempts 
            WHERE identifier = $1 
            AND identifier_type = 'PHONE'
            AND blocked_until > NOW()
            ORDER BY last_attempt_at DESC
            LIMIT 1
        `;
        
        const result = await this.db.query(query, [phone]);
        
        if (result.rows.length > 0) {
            const blockedUntil = result.rows[0].blocked_until;
            throw new Error(`Phone number is temporarily blocked until ${blockedUntil.toISOString()}`);
        }
        
        return false;
    }

    /**
     * Send OTP via SMS using configured provider
     */
    async sendSMS(phone, otp, purpose = 'LOGIN') {
        try {
            let message = this.generateOTPMessage(otp, purpose);
            let response;

            switch (this.smsProvider) {
                case 'MSG91':
                    response = await this.sendViaMSG91(phone, otp, message);
                    break;
                case 'TWILIO':
                    response = await this.sendViaTwilio(phone, message);
                    break;
                case 'TEXTLOCAL':
                    response = await this.sendViaTextLocal(phone, message);
                    break;
                default:
                    throw new Error('SMS provider not configured');
            }

            return response;
        } catch (error) {
            console.error('SMS sending failed:', error);
            throw new Error('Failed to send OTP. Please try again.');
        }
    }

    /**
     * Send OTP via MSG91 (Indian SMS provider)
     */
    async sendViaMSG91(phone, otp, message) {
        const cleanPhone = phone.replace(/^\+91/, '').replace(/\D/g, '');
        
        const response = await axios.post(`${this.smsConfig.baseUrl}/send`, {
            template_id: this.smsConfig.templateId,
            mobile: cleanPhone,
            authkey: this.smsConfig.authKey,
            otp: otp,
            sender: this.smsConfig.senderId
        }, {
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.data.type === 'success') {
            return {
                success: true,
                messageId: response.data.request_id,
                provider: 'MSG91'
            };
        } else {
            throw new Error(response.data.message || 'MSG91 SMS failed');
        }
    }

    /**
     * Send OTP via Twilio
     */
    async sendViaTwilio(phone, message) {
        const twilio = require('twilio')(
            this.smsConfig.accountSid,
            this.smsConfig.authToken
        );

        const response = await twilio.messages.create({
            body: message,
            from: this.smsConfig.fromNumber,
            to: phone
        });

        return {
            success: true,
            messageId: response.sid,
            provider: 'TWILIO'
        };
    }

    /**
     * Send OTP via TextLocal
     */
    async sendViaTextLocal(phone, message) {
        const cleanPhone = phone.replace(/^\+91/, '').replace(/\D/g, '');
        
        const response = await axios.post(this.smsConfig.baseUrl, {
            apikey: this.smsConfig.apiKey,
            numbers: cleanPhone,
            message: message,
            sender: this.smsConfig.sender
        }, {
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        });

        if (response.data.status === 'success') {
            return {
                success: true,
                messageId: response.data.batch_id,
                provider: 'TEXTLOCAL'
            };
        } else {
            throw new Error(response.data.errors?.[0]?.message || 'TextLocal SMS failed');
        }
    }

    /**
     * Generate OTP message based on purpose
     */
    generateOTPMessage(otp, purpose) {
        const messages = {
            SIGNUP: `Welcome to SIP Brewery! Your verification code is ${otp}. Valid for ${this.OTP_EXPIRY_MINUTES} minutes. Do not share this code.`,
            LOGIN: `Your SIP Brewery login code is ${otp}. Valid for ${this.OTP_EXPIRY_MINUTES} minutes. Do not share this code.`,
            RESET_PASSWORD: `Your SIP Brewery password reset code is ${otp}. Valid for ${this.OTP_EXPIRY_MINUTES} minutes. Do not share this code.`,
            TRANSACTION: `Your SIP Brewery transaction verification code is ${otp}. Valid for ${this.OTP_EXPIRY_MINUTES} minutes. Do not share this code.`,
            PROFILE_UPDATE: `Your SIP Brewery profile update code is ${otp}. Valid for ${this.OTP_EXPIRY_MINUTES} minutes. Do not share this code.`
        };

        return messages[purpose] || messages.LOGIN;
    }

    /**
     * Send OTP to user
     */
    async sendOTP(phone, purpose = 'LOGIN', ipAddress, userAgent = '') {
        try {
            // Validate phone number
            if (!this.isValidPhoneNumber(phone)) {
                throw new Error('Invalid phone number format');
            }

            // Check rate limiting
            await this.checkRateLimit(phone, ipAddress);

            // Check if phone is blocked
            await this.checkPhoneBlocked(phone);

            // Generate OTP
            const otp = this.generateOTP();
            const otpHash = await this.hashOTP(otp);

            // Calculate expiry
            const expiresAt = new Date();
            expiresAt.setMinutes(expiresAt.getMinutes() + this.OTP_EXPIRY_MINUTES);

            // Store OTP in database
            const query = `
                INSERT INTO otp_verifications (
                    phone, otp_code, otp_hash, otp_type, purpose,
                    expires_at, ip_address, user_agent
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            `;

            const result = await this.db.query(query, [
                phone, otp, otpHash, 'SMS', purpose,
                expiresAt, ipAddress, userAgent
            ]);

            const otpId = result.rows[0].id;

            // Send SMS
            const smsResponse = await this.sendSMS(phone, otp, purpose);

            // Log security event
            await this.logSecurityEvent(null, 'OTP_SENT', 'AUTHENTICATION', 'INFO', 
                `OTP sent for ${purpose}`, {
                    phone: this.maskPhone(phone),
                    purpose,
                    provider: smsResponse.provider,
                    messageId: smsResponse.messageId
                }, ipAddress, userAgent);

            return {
                success: true,
                otpId,
                message: `OTP sent to ${this.maskPhone(phone)}`,
                expiresIn: this.OTP_EXPIRY_MINUTES * 60, // seconds
                attemptsRemaining: this.MAX_ATTEMPTS
            };

        } catch (error) {
            // Log failed OTP attempt
            await this.logSecurityEvent(null, 'OTP_SEND_FAILED', 'AUTHENTICATION', 'ERROR',
                `Failed to send OTP: ${error.message}`, {
                    phone: this.maskPhone(phone),
                    purpose,
                    error: error.message
                }, ipAddress, userAgent);

            throw error;
        }
    }

    /**
     * Verify OTP
     */
    async verifyOTP(phone, otp, purpose = 'LOGIN', ipAddress, userAgent = '') {
        try {
            // Find valid OTP
            const query = `
                SELECT id, otp_hash, attempts, expires_at, status
                FROM otp_verifications
                WHERE phone = $1 
                AND purpose = $2
                AND status = 'PENDING'
                AND expires_at > NOW()
                ORDER BY created_at DESC
                LIMIT 1
            `;

            const result = await this.db.query(query, [phone, purpose]);

            if (result.rows.length === 0) {
                await this.recordFailedAttempt(phone, 'OTP_NOT_FOUND', ipAddress, userAgent);
                throw new Error('Invalid or expired OTP');
            }

            const otpRecord = result.rows[0];

            // Check attempts
            if (otpRecord.attempts >= this.MAX_ATTEMPTS) {
                await this.updateOTPStatus(otpRecord.id, 'BLOCKED');
                throw new Error('Maximum OTP attempts exceeded');
            }

            // Verify OTP
            const isValid = await this.verifyOTPHash(otp, otpRecord.otp_hash);

            if (!isValid) {
                // Increment attempts
                await this.incrementOTPAttempts(otpRecord.id);
                await this.recordFailedAttempt(phone, 'INVALID_OTP', ipAddress, userAgent);
                
                const remainingAttempts = this.MAX_ATTEMPTS - (otpRecord.attempts + 1);
                throw new Error(`Invalid OTP. ${remainingAttempts} attempts remaining.`);
            }

            // Mark OTP as verified
            await this.updateOTPStatus(otpRecord.id, 'VERIFIED');

            // Log successful verification
            await this.logSecurityEvent(null, 'OTP_VERIFIED', 'AUTHENTICATION', 'INFO',
                `OTP verified successfully for ${purpose}`, {
                    phone: this.maskPhone(phone),
                    purpose,
                    otpId: otpRecord.id
                }, ipAddress, userAgent);

            return {
                success: true,
                message: 'OTP verified successfully',
                otpId: otpRecord.id
            };

        } catch (error) {
            // Log verification failure
            await this.logSecurityEvent(null, 'OTP_VERIFICATION_FAILED', 'AUTHENTICATION', 'WARNING',
                `OTP verification failed: ${error.message}`, {
                    phone: this.maskPhone(phone),
                    purpose,
                    error: error.message
                }, ipAddress, userAgent);

            throw error;
        }
    }

    /**
     * Update OTP status
     */
    async updateOTPStatus(otpId, status) {
        const query = `
            UPDATE otp_verifications 
            SET status = $1, verified_at = CASE WHEN $1 = 'VERIFIED' THEN NOW() ELSE verified_at END
            WHERE id = $2
        `;
        await this.db.query(query, [status, otpId]);
    }

    /**
     * Increment OTP attempts
     */
    async incrementOTPAttempts(otpId) {
        const query = `
            UPDATE otp_verifications 
            SET attempts = attempts + 1
            WHERE id = $1
        `;
        await this.db.query(query, [otpId]);
    }

    /**
     * Record failed login attempt
     */
    async recordFailedAttempt(phone, reason, ipAddress, userAgent) {
        const query = `
            INSERT INTO failed_login_attempts (
                identifier, identifier_type, failure_reason, 
                ip_address, user_agent, attempt_count
            ) VALUES ($1, $2, $3, $4, $5, 1)
            ON CONFLICT (identifier, ip_address) 
            DO UPDATE SET 
                attempt_count = failed_login_attempts.attempt_count + 1,
                last_attempt_at = NOW(),
                blocked_until = CASE 
                    WHEN failed_login_attempts.attempt_count + 1 >= 5 
                    THEN NOW() + INTERVAL '30 minutes'
                    ELSE failed_login_attempts.blocked_until
                END
        `;

        await this.db.query(query, [phone, 'PHONE', reason, ipAddress, userAgent]);
    }

    /**
     * Log security event
     */
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

    /**
     * Validate phone number format
     */
    isValidPhoneNumber(phone) {
        // Indian phone number validation
        const phoneRegex = /^(\+91|91)?[6-9]\d{9}$/;
        return phoneRegex.test(phone.replace(/\s+/g, ''));
    }

    /**
     * Mask phone number for logging
     */
    maskPhone(phone) {
        if (phone.length <= 4) return phone;
        return phone.slice(0, -4).replace(/\d/g, '*') + phone.slice(-4);
    }

    /**
     * Clean expired OTPs (called by cron job)
     */
    async cleanupExpiredOTPs() {
        const query = `
            UPDATE otp_verifications 
            SET status = 'EXPIRED'
            WHERE expires_at < NOW() 
            AND status = 'PENDING'
        `;

        const result = await this.db.query(query);
        return result.rowCount;
    }

    /**
     * Get OTP statistics for monitoring
     */
    async getOTPStats(timeframe = '24 hours') {
        const query = `
            SELECT 
                COUNT(*) as total_sent,
                COUNT(CASE WHEN status = 'VERIFIED' THEN 1 END) as verified,
                COUNT(CASE WHEN status = 'EXPIRED' THEN 1 END) as expired,
                COUNT(CASE WHEN status = 'BLOCKED' THEN 1 END) as blocked,
                AVG(attempts) as avg_attempts
            FROM otp_verifications
            WHERE created_at > NOW() - INTERVAL '${timeframe}'
        `;

        const result = await this.db.query(query);
        return result.rows[0];
    }
}

module.exports = new OTPService();
