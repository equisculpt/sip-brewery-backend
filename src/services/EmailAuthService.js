// Email Authentication Service with OTP and Magic Link support
const nodemailer = require('nodemailer');
const crypto = require('crypto');
const validator = require('validator');

class EmailAuthService {
    constructor() {
        this.otpStorage = new Map(); // Use Redis in production
        this.magicLinkStorage = new Map();
        this.otpExpiry = 10 * 60 * 1000; // 10 minutes
        this.magicLinkExpiry = 30 * 60 * 1000; // 30 minutes
        
        // Initialize email transporter with fallback
        this.initializeTransporter();
    }

    async initializeTransporter() {
        // Primary: SendGrid
        if (process.env.SENDGRID_API_KEY) {
            this.transporter = nodemailer.createTransporter({
                service: 'SendGrid',
                auth: {
                    user: 'apikey',
                    pass: process.env.SENDGRID_API_KEY
                }
            });
            this.provider = 'SendGrid';
            return;
        }

        // Backup: SMTP
        if (process.env.SMTP_HOST) {
            this.transporter = nodemailer.createTransporter({
                host: process.env.SMTP_HOST,
                port: parseInt(process.env.SMTP_PORT) || 587,
                secure: false,
                auth: {
                    user: process.env.SMTP_USER,
                    pass: process.env.SMTP_PASS
                }
            });
            this.provider = 'SMTP';
            return;
        }

        // Fallback: Gmail
        this.transporter = nodemailer.createTransporter({
            service: 'gmail',
            auth: {
                user: process.env.GMAIL_USER,
                pass: process.env.GMAIL_APP_PASSWORD
            }
        });
        this.provider = 'Gmail';
    }

    // Generate secure 6-digit OTP
    generateOTP() {
        return crypto.randomInt(100000, 999999).toString();
    }

    // Generate secure magic link token
    generateMagicToken() {
        return crypto.randomBytes(32).toString('hex');
    }

    // Hash sensitive data
    hashData(data) {
        return crypto.createHash('sha256')
            .update(data + process.env.EMAIL_SALT)
            .digest('hex');
    }

    // Send Email OTP
    async sendEmailOTP(email, purpose = 'EMAIL_VERIFICATION') {
        try {
            // Validate email
            if (!validator.isEmail(email)) {
                throw new Error('Invalid email format');
            }

            const otp = this.generateOTP();
            const otpId = crypto.randomUUID();
            const expiresAt = new Date(Date.now() + this.otpExpiry);

            // Store OTP securely
            this.otpStorage.set(otpId, {
                otp: this.hashData(otp),
                email: email.toLowerCase(),
                purpose,
                attempts: 0,
                expiresAt,
                createdAt: new Date()
            });

            // Send email
            const mailOptions = {
                from: `"SIP Brewery" <${process.env.FROM_EMAIL}>`,
                to: email,
                subject: this.getEmailSubject(purpose),
                html: this.getOTPEmailTemplate(otp, purpose, email),
                text: this.getOTPEmailText(otp, purpose)
            };

            const info = await this.transporter.sendMail(mailOptions);

            console.log(`Email OTP sent via ${this.provider}:`, {
                email: this.maskEmail(email),
                otpId,
                purpose,
                messageId: info.messageId
            });

            return {
                success: true,
                otpId,
                message: 'Email OTP sent successfully',
                expiresAt,
                provider: this.provider
            };

        } catch (error) {
            console.error('Email OTP Error:', error);
            throw new Error('Failed to send email OTP');
        }
    }

    // Send Magic Link
    async sendMagicLink(email, purpose = 'EMAIL_VERIFICATION') {
        try {
            if (!validator.isEmail(email)) {
                throw new Error('Invalid email format');
            }

            const token = this.generateMagicToken();
            const linkId = crypto.randomUUID();
            const expiresAt = new Date(Date.now() + this.magicLinkExpiry);

            // Store magic link
            this.magicLinkStorage.set(linkId, {
                token: this.hashData(token),
                email: email.toLowerCase(),
                purpose,
                used: false,
                expiresAt,
                createdAt: new Date()
            });

            // Create magic link URL
            const magicLink = `${process.env.FRONTEND_URL}/auth/verify-magic-link?linkId=${linkId}&token=${token}&email=${encodeURIComponent(email)}`;

            // Send email
            const mailOptions = {
                from: `"SIP Brewery" <${process.env.FROM_EMAIL}>`,
                to: email,
                subject: this.getMagicLinkSubject(purpose),
                html: this.getMagicLinkEmailTemplate(magicLink, purpose, email),
                text: this.getMagicLinkEmailText(magicLink, purpose)
            };

            const info = await this.transporter.sendMail(mailOptions);

            console.log(`Magic link sent via ${this.provider}:`, {
                email: this.maskEmail(email),
                linkId,
                purpose,
                messageId: info.messageId
            });

            return {
                success: true,
                linkId,
                message: 'Magic link sent successfully',
                expiresAt,
                provider: this.provider
            };

        } catch (error) {
            console.error('Magic Link Error:', error);
            throw new Error('Failed to send magic link');
        }
    }

    // Verify Email OTP
    async verifyEmailOTP(otpId, providedOTP, email) {
        try {
            const otpData = this.otpStorage.get(otpId);

            if (!otpData) {
                return {
                    success: false,
                    message: 'Invalid or expired OTP request'
                };
            }

            // Check expiry
            if (new Date() > otpData.expiresAt) {
                this.otpStorage.delete(otpId);
                return {
                    success: false,
                    message: 'OTP has expired'
                };
            }

            // Check attempts
            if (otpData.attempts >= 3) {
                this.otpStorage.delete(otpId);
                return {
                    success: false,
                    message: 'Maximum verification attempts exceeded'
                };
            }

            // Verify email matches
            if (otpData.email !== email.toLowerCase()) {
                return {
                    success: false,
                    message: 'Email mismatch'
                };
            }

            // Verify OTP (timing-safe comparison)
            const hashedProvidedOTP = this.hashData(providedOTP);
            const isValid = crypto.timingSafeEqual(
                Buffer.from(otpData.otp, 'hex'),
                Buffer.from(hashedProvidedOTP, 'hex')
            );

            if (isValid) {
                this.otpStorage.delete(otpId);
                
                console.log('Email OTP verified successfully:', {
                    email: this.maskEmail(email),
                    purpose: otpData.purpose
                });

                return {
                    success: true,
                    message: 'Email verified successfully',
                    purpose: otpData.purpose,
                    email: email.toLowerCase()
                };
            } else {
                // Increment attempts
                otpData.attempts += 1;
                this.otpStorage.set(otpId, otpData);
                
                return {
                    success: false,
                    message: `Invalid OTP. ${3 - otpData.attempts} attempts remaining`
                };
            }

        } catch (error) {
            console.error('Email OTP Verification Error:', error);
            return {
                success: false,
                message: 'OTP verification failed'
            };
        }
    }

    // Verify Magic Link
    async verifyMagicLink(linkId, token, email) {
        try {
            const linkData = this.magicLinkStorage.get(linkId);

            if (!linkData) {
                return {
                    success: false,
                    message: 'Invalid or expired magic link'
                };
            }

            // Check if already used
            if (linkData.used) {
                return {
                    success: false,
                    message: 'Magic link has already been used'
                };
            }

            // Check expiry
            if (new Date() > linkData.expiresAt) {
                this.magicLinkStorage.delete(linkId);
                return {
                    success: false,
                    message: 'Magic link has expired'
                };
            }

            // Verify email matches
            if (linkData.email !== email.toLowerCase()) {
                return {
                    success: false,
                    message: 'Email mismatch'
                };
            }

            // Verify token (timing-safe comparison)
            const hashedToken = this.hashData(token);
            const isValid = crypto.timingSafeEqual(
                Buffer.from(linkData.token, 'hex'),
                Buffer.from(hashedToken, 'hex')
            );

            if (isValid) {
                // Mark as used and delete
                this.magicLinkStorage.delete(linkId);
                
                console.log('Magic link verified successfully:', {
                    email: this.maskEmail(email),
                    purpose: linkData.purpose
                });

                return {
                    success: true,
                    message: 'Email verified successfully via magic link',
                    purpose: linkData.purpose,
                    email: email.toLowerCase()
                };
            } else {
                return {
                    success: false,
                    message: 'Invalid magic link token'
                };
            }

        } catch (error) {
            console.error('Magic Link Verification Error:', error);
            return {
                success: false,
                message: 'Magic link verification failed'
            };
        }
    }

    // Email Templates
    getOTPEmailTemplate(otp, purpose, email) {
        const templates = {
            EMAIL_VERIFICATION: `
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Email Verification - SIP Brewery</title>
                </head>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 0; background-color: #f4f4f4;">
                    <div style="max-width: 600px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1);">
                        <div style="text-align: center; margin-bottom: 30px;">
                            <h1 style="color: #2563eb; margin: 0;">🍺 SIP Brewery</h1>
                            <p style="color: #6b7280; margin: 5px 0;">Premium Investment Platform</p>
                        </div>
                        
                        <h2 style="color: #1f2937; text-align: center;">Email Verification</h2>
                        
                        <p>Hello,</p>
                        <p>Thank you for joining SIP Brewery! Please use the verification code below to confirm your email address:</p>
                        
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; text-align: center; border-radius: 8px; margin: 30px 0;">
                            <div style="background: white; padding: 15px; border-radius: 5px; display: inline-block;">
                                <span style="font-size: 32px; font-weight: bold; color: #1f2937; letter-spacing: 5px;">${otp}</span>
                            </div>
                        </div>
                        
                        <div style="background: #fef3c7; padding: 15px; border-radius: 5px; border-left: 4px solid #f59e0b; margin: 20px 0;">
                            <p style="margin: 0; color: #92400e;"><strong>⏰ Important:</strong> This code expires in 10 minutes</p>
                        </div>
                        
                        <p>If you didn't request this verification, please ignore this email.</p>
                        
                        <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 30px 0;">
                        
                        <div style="text-align: center; color: #6b7280; font-size: 14px;">
                            <p>© 2024 SIP Brewery. All rights reserved.</p>
                            <p>Secure • Reliable • Profitable</p>
                        </div>
                    </div>
                </body>
                </html>
            `,
            TWO_FACTOR: `
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>Two-Factor Authentication - SIP Brewery</title>
                </head>
                <body style="font-family: Arial, sans-serif; background-color: #f4f4f4;">
                    <div style="max-width: 600px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px;">
                        <h1 style="color: #dc2626; text-align: center;">🔐 Security Verification</h1>
                        <p>A login attempt requires verification. Your security code is:</p>
                        <div style="background: #fef2f2; padding: 20px; text-align: center; border-radius: 8px; margin: 20px 0;">
                            <span style="font-size: 32px; font-weight: bold; color: #dc2626; letter-spacing: 3px;">${otp}</span>
                        </div>
                        <p><strong>⚠️ Security Alert:</strong> If you didn't request this code, your account may be compromised.</p>
                    </div>
                </body>
                </html>
            `
        };

        return templates[purpose] || templates.EMAIL_VERIFICATION;
    }

    getMagicLinkEmailTemplate(magicLink, purpose, email) {
        return `
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Magic Link - SIP Brewery</title>
            </head>
            <body style="font-family: Arial, sans-serif; background-color: #f4f4f4;">
                <div style="max-width: 600px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px;">
                    <h1 style="color: #2563eb; text-align: center;">🪄 Magic Link Login</h1>
                    <p>Click the button below to securely access your SIP Brewery account:</p>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="${magicLink}" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; text-decoration: none; border-radius: 25px; font-weight: bold; display: inline-block;">
                            🚀 Access My Account
                        </a>
                    </div>
                    
                    <p style="color: #6b7280; font-size: 14px;">This link expires in 30 minutes and can only be used once.</p>
                    <p style="color: #6b7280; font-size: 14px;">If the button doesn't work, copy and paste this link: ${magicLink}</p>
                </div>
            </body>
            </html>
        `;
    }

    // Text versions for email clients that don't support HTML
    getOTPEmailText(otp, purpose) {
        return `
SIP Brewery - Email Verification

Your verification code is: ${otp}

This code expires in 10 minutes.
If you didn't request this verification, please ignore this email.

© 2024 SIP Brewery. All rights reserved.
        `.trim();
    }

    getMagicLinkEmailText(magicLink, purpose) {
        return `
SIP Brewery - Magic Link Login

Click this link to access your account: ${magicLink}

This link expires in 30 minutes and can only be used once.

© 2024 SIP Brewery. All rights reserved.
        `.trim();
    }

    // Helper methods
    getEmailSubject(purpose) {
        const subjects = {
            EMAIL_VERIFICATION: 'SIP Brewery - Verify Your Email',
            TWO_FACTOR: 'SIP Brewery - Security Verification Required',
            PASSWORD_RESET: 'SIP Brewery - Password Reset Code'
        };
        return subjects[purpose] || 'SIP Brewery - Verification Code';
    }

    getMagicLinkSubject(purpose) {
        return 'SIP Brewery - Your Magic Link Login';
    }

    maskEmail(email) {
        const [username, domain] = email.split('@');
        const maskedUsername = username.length > 2 
            ? username[0] + '*'.repeat(username.length - 2) + username[username.length - 1]
            : username;
        return `${maskedUsername}@${domain}`;
    }

    // Cleanup expired entries
    cleanupExpired() {
        const now = new Date();
        
        // Cleanup expired OTPs
        for (const [otpId, otpData] of this.otpStorage.entries()) {
            if (now > otpData.expiresAt) {
                this.otpStorage.delete(otpId);
            }
        }

        // Cleanup expired magic links
        for (const [linkId, linkData] of this.magicLinkStorage.entries()) {
            if (now > linkData.expiresAt) {
                this.magicLinkStorage.delete(linkId);
            }
        }
    }

    // Get statistics
    getStats() {
        return {
            activeOTPs: this.otpStorage.size,
            activeMagicLinks: this.magicLinkStorage.size,
            provider: this.provider,
            uptime: process.uptime()
        };
    }
}

module.exports = new EmailAuthService();
