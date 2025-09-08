// Email Authentication and Weekly ASI Analysis Routes
const express = require('express');
const rateLimit = require('express-rate-limit');
const { body, validationResult } = require('express-validator');
const EmailAuthService = require('../services/EmailAuthService');
const cron = require('node-cron');
const nodemailer = require('nodemailer');

const router = express.Router();

// Rate limiting for email requests
const emailRateLimit = rateLimit({
    windowMs: 1 * 60 * 1000, // 1 minute
    max: 3, // 3 email requests per minute
    message: {
        success: false,
        message: 'Too many email requests. Please try again later.'
    }
});

// Rate limiting for verification
const verifyRateLimit = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 10, // 10 verification attempts
    message: {
        success: false,
        message: 'Too many verification attempts. Please try again later.'
    }
});

// Send Email OTP
router.post('/email/send-otp',
    emailRateLimit,
    [
        body('email')
            .isEmail()
            .normalizeEmail()
            .withMessage('Invalid email format'),
        body('purpose')
            .optional()
            .isIn(['EMAIL_VERIFICATION', 'TWO_FACTOR', 'PASSWORD_RESET'])
            .withMessage('Invalid purpose')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { email, purpose = 'EMAIL_VERIFICATION' } = req.body;
            const result = await EmailAuthService.sendEmailOTP(email, purpose);
            
            res.json(result);

        } catch (error) {
            console.error('Send Email OTP Error:', error);
            res.status(500).json({
                success: false,
                message: 'Failed to send email OTP'
            });
        }
    }
);

// Send Magic Link
router.post('/email/send-magic-link',
    emailRateLimit,
    [
        body('email')
            .isEmail()
            .normalizeEmail()
            .withMessage('Invalid email format'),
        body('purpose')
            .optional()
            .isIn(['EMAIL_VERIFICATION', 'LOGIN'])
            .withMessage('Invalid purpose')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { email, purpose = 'EMAIL_VERIFICATION' } = req.body;
            const result = await EmailAuthService.sendMagicLink(email, purpose);
            
            res.json(result);

        } catch (error) {
            console.error('Send Magic Link Error:', error);
            res.status(500).json({
                success: false,
                message: 'Failed to send magic link'
            });
        }
    }
);

// Verify Email OTP
router.post('/email/verify-otp',
    verifyRateLimit,
    [
        body('otpId')
            .isUUID()
            .withMessage('Invalid OTP ID format'),
        body('otp')
            .isLength({ min: 6, max: 6 })
            .isNumeric()
            .withMessage('OTP must be 6 digits'),
        body('email')
            .isEmail()
            .normalizeEmail()
            .withMessage('Invalid email format')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { otpId, otp, email } = req.body;
            const result = await EmailAuthService.verifyEmailOTP(otpId, otp, email);
            
            res.json(result);

        } catch (error) {
            console.error('Verify Email OTP Error:', error);
            res.status(500).json({
                success: false,
                message: 'Email OTP verification failed'
            });
        }
    }
);

// Verify Magic Link
router.post('/email/verify-magic-link',
    verifyRateLimit,
    [
        body('linkId')
            .isUUID()
            .withMessage('Invalid link ID format'),
        body('token')
            .isLength({ min: 64, max: 64 })
            .isHexadecimal()
            .withMessage('Invalid token format'),
        body('email')
            .isEmail()
            .normalizeEmail()
            .withMessage('Invalid email format')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { linkId, token, email } = req.body;
            const result = await EmailAuthService.verifyMagicLink(linkId, token, email);
            
            res.json(result);

        } catch (error) {
            console.error('Verify Magic Link Error:', error);
            res.status(500).json({
                success: false,
                message: 'Magic link verification failed'
            });
        }
    }
);

// Weekly ASI Analysis Cron Job Management
class WeeklyASIAnalysis {
    constructor() {
        this.cronJob = null;
        this.isRunning = false;
        this.lastRun = null;
        this.emailTransporter = this.initializeEmailTransporter();
    }

    initializeEmailTransporter() {
        if (process.env.SENDGRID_API_KEY) {
            return nodemailer.createTransporter({
                service: 'SendGrid',
                auth: {
                    user: 'apikey',
                    pass: process.env.SENDGRID_API_KEY
                }
            });
        }
        return null;
    }

    startWeeklyCron() {
        // Every Sunday at 9:00 AM IST
        this.cronJob = cron.schedule('0 9 * * 0', async () => {
            await this.runWeeklyAnalysis();
        }, {
            scheduled: true,
            timezone: "Asia/Kolkata"
        });

        console.log('📅 Weekly ASI Analysis cron job started - Every Sunday 9:00 AM IST');
    }

    async runWeeklyAnalysis() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.lastRun = new Date();

        try {
            console.log('🚀 Starting Weekly ASI Portfolio Analysis...');
            
            // Get all active users (replace with actual database query)
            const users = await this.getAllActiveUsers();
            
            let successCount = 0;
            let errorCount = 0;

            for (const user of users) {
                try {
                    await this.sendUserAnalysis(user);
                    successCount++;
                    await this.sleep(2000); // 2 second delay
                } catch (error) {
                    console.error(`Failed to send analysis to ${user.email}:`, error);
                    errorCount++;
                }
            }

            console.log(`✅ Weekly analysis completed: ${successCount} success, ${errorCount} errors`);

        } catch (error) {
            console.error('❌ Weekly analysis failed:', error);
        } finally {
            this.isRunning = false;
        }
    }

    async getAllActiveUsers() {
        // Mock data - replace with actual database query
        return [
            {
                id: 'user-1',
                name: 'Rajesh Kumar',
                email: 'rajesh@example.com',
                totalInvested: 250000,
                currentValue: 287500,
                returns: 15.0
            },
            {
                id: 'user-2',
                name: 'Priya Sharma', 
                email: 'priya@example.com',
                totalInvested: 500000,
                currentValue: 625000,
                returns: 25.0
            }
        ];
    }

    async sendUserAnalysis(user) {
        const emailContent = this.generateAnalysisEmail(user);
        
        const mailOptions = {
            from: `"SIP Brewery ASI" <${process.env.FROM_EMAIL}>`,
            to: user.email,
            subject: `📊 Your Weekly Portfolio Analysis - ${new Date().toLocaleDateString('en-IN')}`,
            html: emailContent
        };

        if (this.emailTransporter) {
            await this.emailTransporter.sendMail(mailOptions);
            console.log(`✅ Analysis sent to ${user.name}`);
        }
    }

    generateAnalysisEmail(user) {
        return `
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Weekly Portfolio Analysis</title>
</head>
<body style="font-family: Arial, sans-serif; background: #f4f4f4; margin: 0; padding: 20px;">
    <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
        
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center;">
            <h1 style="margin: 0; font-size: 24px;">🍺 SIP Brewery</h1>
            <p style="margin: 10px 0 0 0;">Weekly Portfolio Analysis</p>
            <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.8;">
                ${new Date().toLocaleDateString('en-IN', { 
                    weekday: 'long', 
                    year: 'numeric', 
                    month: 'long', 
                    day: 'numeric' 
                })}
            </p>
        </div>

        <!-- Content -->
        <div style="padding: 30px;">
            <h2 style="color: #2d3748; margin: 0 0 20px 0;">Hello ${user.name}! 👋</h2>
            
            <!-- Portfolio Summary -->
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 10px; padding: 20px; margin-bottom: 25px; color: white;">
                <h3 style="margin: 0 0 15px 0;">📊 Portfolio Summary</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; text-align: center;">
                    <div>
                        <div style="font-size: 20px; font-weight: bold;">₹${user.totalInvested.toLocaleString('en-IN')}</div>
                        <div style="opacity: 0.9; font-size: 12px;">Invested</div>
                    </div>
                    <div>
                        <div style="font-size: 20px; font-weight: bold;">₹${user.currentValue.toLocaleString('en-IN')}</div>
                        <div style="opacity: 0.9; font-size: 12px;">Current Value</div>
                    </div>
                    <div>
                        <div style="font-size: 20px; font-weight: bold;">+${user.returns}%</div>
                        <div style="opacity: 0.9; font-size: 12px;">Returns</div>
                    </div>
                </div>
            </div>

            <!-- ASI Analysis -->
            <div style="background: #f7fafc; border-radius: 10px; padding: 20px; margin-bottom: 25px;">
                <h3 style="color: #2d3748; margin: 0 0 15px 0;">🤖 ASI Analysis</h3>
                <div style="background: white; border-radius: 8px; padding: 15px;">
                    <p style="margin: 0; color: #4a5568;">
                        <strong>Performance:</strong> Your portfolio is performing excellently with ${user.returns}% returns, 
                        outperforming NIFTY 50 by ${(user.returns - 12.5).toFixed(1)}%.
                    </p>
                </div>
            </div>

            <!-- Recommendations -->
            <div style="background: #fff5f5; border-radius: 10px; padding: 20px; margin-bottom: 25px;">
                <h3 style="color: #2d3748; margin: 0 0 15px 0;">💡 This Week's Recommendations</h3>
                <div style="background: white; border-radius: 8px; padding: 15px;">
                    <ul style="margin: 0; padding-left: 20px; color: #4a5568;">
                        <li>Consider rebalancing your portfolio allocation</li>
                        <li>Increase SIP amount by 10% if possible</li>
                        <li>Add ELSS fund for tax saving benefits</li>
                    </ul>
                </div>
            </div>

            <!-- Market Outlook -->
            <div style="background: #f0fff4; border-radius: 10px; padding: 20px; margin-bottom: 25px;">
                <h3 style="color: #2d3748; margin: 0 0 15px 0;">🔮 Market Outlook</h3>
                <div style="background: white; border-radius: 8px; padding: 15px;">
                    <p style="margin: 0; color: #4a5568;">
                        <strong>Outlook:</strong> Positive | <strong>NIFTY Target:</strong> 26,500 by March 2025
                    </p>
                    <p style="margin: 10px 0 0 0; color: #4a5568; font-size: 14px;">
                        Key drivers: Strong earnings growth, stable policies, increasing retail participation
                    </p>
                </div>
            </div>

            <!-- CTA Button -->
            <div style="text-align: center; margin: 30px 0;">
                <a href="${process.env.FRONTEND_URL}/dashboard" 
                   style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; text-decoration: none; border-radius: 25px; font-weight: bold; display: inline-block;">
                    📱 View Full Analysis
                </a>
            </div>

            <!-- Footer -->
            <div style="text-align: center; padding: 20px 0; border-top: 1px solid #e2e8f0;">
                <p style="color: #718096; margin: 0; font-size: 14px;">
                    This analysis was generated by SIP Brewery's ASI system
                </p>
                <p style="color: #a0aec0; font-size: 12px; margin: 10px 0 0 0;">
                    © 2024 SIP Brewery. All rights reserved.
                </p>
            </div>
        </div>
    </div>
</body>
</html>
        `;
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    stopCron() {
        if (this.cronJob) {
            this.cronJob.stop();
            console.log('📅 Weekly ASI Analysis cron job stopped');
        }
    }

    getStatus() {
        return {
            isRunning: this.isRunning,
            lastRun: this.lastRun,
            cronActive: this.cronJob ? true : false,
            nextRun: this.getNextSunday()
        };
    }

    getNextSunday() {
        const now = new Date();
        const nextSunday = new Date(now);
        nextSunday.setDate(now.getDate() + (7 - now.getDay()));
        nextSunday.setHours(9, 0, 0, 0);
        return nextSunday.toLocaleString('en-IN');
    }
}

// Initialize Weekly ASI Analysis
const weeklyASI = new WeeklyASIAnalysis();

// Start Weekly ASI Analysis Cron
router.post('/asi/start-weekly-cron', async (req, res) => {
    try {
        weeklyASI.startWeeklyCron();
        res.json({
            success: true,
            message: 'Weekly ASI analysis cron job started',
            nextRun: weeklyASI.getNextSunday()
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            message: 'Failed to start cron job'
        });
    }
});

// Stop Weekly ASI Analysis Cron
router.post('/asi/stop-weekly-cron', async (req, res) => {
    try {
        weeklyASI.stopCron();
        res.json({
            success: true,
            message: 'Weekly ASI analysis cron job stopped'
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            message: 'Failed to stop cron job'
        });
    }
});

// Get Weekly ASI Analysis Status
router.get('/asi/cron-status', async (req, res) => {
    try {
        const status = weeklyASI.getStatus();
        res.json({
            success: true,
            data: status
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            message: 'Failed to get cron status'
        });
    }
});

// Trigger Weekly Analysis Manually (for testing)
router.post('/asi/trigger-analysis', async (req, res) => {
    try {
        // Run analysis in background
        weeklyASI.runWeeklyAnalysis().catch(console.error);
        
        res.json({
            success: true,
            message: 'Weekly analysis triggered manually'
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            message: 'Failed to trigger analysis'
        });
    }
});

// Email service health check
router.get('/email/health', async (req, res) => {
    try {
        const stats = EmailAuthService.getStats();
        res.json({
            success: true,
            message: 'Email authentication service is healthy',
            data: stats
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            message: 'Email service health check failed'
        });
    }
});

// Auto-start weekly cron on server startup
if (process.env.NODE_ENV === 'production' || process.env.AUTO_START_CRON === 'true') {
    weeklyASI.startWeeklyCron();
}

module.exports = router;
