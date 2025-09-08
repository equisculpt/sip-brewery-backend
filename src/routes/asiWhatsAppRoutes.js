const express = require('express');
const router = express.Router();
const ASIWhatsAppController = require('../controllers/ASIWhatsAppController');
const { authenticate, authorize } = require('../middleware/auth');
const { validateRequest } = require('../middleware/validation');
const rateLimit = require('express-rate-limit');

console.log('🛣️ SIP BREWERY ASI WHATSAPP ROUTES');
console.log('📱 Complete Platform Operations via WhatsApp');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

// Rate limiting for webhook
const webhookRateLimit = rateLimit({
    windowMs: 1 * 60 * 1000, // 1 minute
    max: 100, // Max 100 requests per minute
    message: {
        success: false,
        message: 'Too many webhook requests, please try again later'
    },
    standardHeaders: true,
    legacyHeaders: false
});

// Rate limiting for admin endpoints
const adminRateLimit = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // Max 100 requests per 15 minutes
    message: {
        success: false,
        message: 'Too many admin requests, please try again later'
    }
});

// Validation schemas
const sendMessageSchema = {
    type: 'object',
    required: ['phoneNumber', 'message'],
    properties: {
        phoneNumber: {
            type: 'string',
            pattern: '^\\d{10,15}$'
        },
        message: {
            type: 'string',
            minLength: 1,
            maxLength: 4096
        },
        type: {
            type: 'string',
            enum: ['text', 'image', 'document']
        }
    }
};

const testIntegrationSchema = {
    type: 'object',
    properties: {
        phoneNumber: {
            type: 'string',
            pattern: '^\\d{10,15}$'
        },
        testMessage: {
            type: 'string',
            minLength: 1,
            maxLength: 1000
        }
    }
};

const generateReportsSchema = {
    type: 'object',
    required: ['phoneNumber'],
    properties: {
        phoneNumber: {
            type: 'string',
            pattern: '^\\d{10,15}$'
        }
    }
};

/**
 * @swagger
 * /api/asi-whatsapp/webhook:
 *   get:
 *     summary: WhatsApp webhook verification
 *     description: Verify WhatsApp webhook with Facebook
 *     tags: [ASI WhatsApp]
 *     parameters:
 *       - in: query
 *         name: hub.mode
 *         required: true
 *         schema:
 *           type: string
 *       - in: query
 *         name: hub.verify_token
 *         required: true
 *         schema:
 *           type: string
 *       - in: query
 *         name: hub.challenge
 *         required: true
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: Webhook verified successfully
 *       403:
 *         description: Invalid verification token
 */
router.get('/webhook', webhookRateLimit, ASIWhatsAppController.handleWebhook);

/**
 * @swagger
 * /api/asi-whatsapp/webhook:
 *   post:
 *     summary: Handle incoming WhatsApp messages
 *     description: Process incoming WhatsApp messages with ASI integration
 *     tags: [ASI WhatsApp]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               object:
 *                 type: string
 *               entry:
 *                 type: array
 *                 items:
 *                   type: object
 *     responses:
 *       200:
 *         description: Messages processed successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 message:
 *                   type: string
 *                 processedCount:
 *                   type: integer
 *       400:
 *         description: Invalid webhook data
 *       500:
 *         description: Internal server error
 */
router.post('/webhook', webhookRateLimit, ASIWhatsAppController.handleWebhook);

/**
 * @swagger
 * /api/asi-whatsapp/send-message:
 *   post:
 *     summary: Send WhatsApp message
 *     description: Send a manual WhatsApp message (admin endpoint)
 *     tags: [ASI WhatsApp]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - phoneNumber
 *               - message
 *             properties:
 *               phoneNumber:
 *                 type: string
 *                 pattern: '^\\d{10,15}$'
 *                 example: '919876543210'
 *               message:
 *                 type: string
 *                 minLength: 1
 *                 maxLength: 4096
 *                 example: 'Hello from SIP Brewery!'
 *               type:
 *                 type: string
 *                 enum: [text, image, document]
 *                 default: text
 *     responses:
 *       200:
 *         description: Message sent successfully
 *       400:
 *         description: Invalid request data
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/send-message', 
    adminRateLimit,
    authenticate,
    authorize(['admin', 'support']),
    validateRequest(sendMessageSchema),
    ASIWhatsAppController.sendMessage
);

/**
 * @swagger
 * /api/asi-whatsapp/status:
 *   get:
 *     summary: Get ASI WhatsApp service status
 *     description: Get the current status of ASI WhatsApp service
 *     tags: [ASI WhatsApp]
 *     responses:
 *       200:
 *         description: Service status retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 data:
 *                   type: object
 *                   properties:
 *                     service:
 *                       type: string
 *                     status:
 *                       type: string
 *                     features:
 *                       type: object
 *                     integrations:
 *                       type: object
 */
router.get('/status', ASIWhatsAppController.getServiceStatus);

/**
 * @swagger
 * /api/asi-whatsapp/test-integration:
 *   post:
 *     summary: Test ASI integration
 *     description: Test the ASI WhatsApp integration with a sample message
 *     tags: [ASI WhatsApp]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: false
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               phoneNumber:
 *                 type: string
 *                 pattern: '^\\d{10,15}$'
 *                 default: '919876543210'
 *               testMessage:
 *                 type: string
 *                 default: 'Hello'
 *     responses:
 *       200:
 *         description: Integration test successful
 *       500:
 *         description: Integration test failed
 */
router.post('/test-integration',
    adminRateLimit,
    authenticate,
    authorize(['admin', 'developer']),
    validateRequest(testIntegrationSchema),
    ASIWhatsAppController.testASIIntegration
);

/**
 * @swagger
 * /api/asi-whatsapp/user-session/{phoneNumber}:
 *   get:
 *     summary: Get user session details
 *     description: Get WhatsApp session details for a specific user
 *     tags: [ASI WhatsApp]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: phoneNumber
 *         required: true
 *         schema:
 *           type: string
 *           pattern: '^\\d{10,15}$'
 *         example: '919876543210'
 *     responses:
 *       200:
 *         description: User session retrieved successfully
 *       400:
 *         description: Invalid phone number
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.get('/user-session/:phoneNumber',
    adminRateLimit,
    authenticate,
    authorize(['admin', 'support']),
    ASIWhatsAppController.getUserSession
);

/**
 * @swagger
 * /api/asi-whatsapp/generate-reports:
 *   post:
 *     summary: Generate reports for user
 *     description: Trigger report generation for a specific user
 *     tags: [ASI WhatsApp]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - phoneNumber
 *             properties:
 *               phoneNumber:
 *                 type: string
 *                 pattern: '^\\d{10,15}$'
 *                 example: '919876543210'
 *     responses:
 *       200:
 *         description: Report generation initiated
 *       400:
 *         description: Invalid request or user not registered
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/generate-reports',
    adminRateLimit,
    authenticate,
    authorize(['admin', 'support']),
    validateRequest(generateReportsSchema),
    ASIWhatsAppController.generateReportsForUser
);

/**
 * @swagger
 * /api/asi-whatsapp/platform-stats:
 *   get:
 *     summary: Get platform statistics
 *     description: Get comprehensive platform statistics for ASI WhatsApp service
 *     tags: [ASI WhatsApp]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Platform statistics retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 data:
 *                   type: object
 *                   properties:
 *                     totalUsers:
 *                       type: integer
 *                     activeUsers:
 *                       type: integer
 *                     totalInvestments:
 *                       type: string
 *                     activeSIPs:
 *                       type: integer
 *                     reportsGenerated:
 *                       type: integer
 *                     topIntents:
 *                       type: array
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.get('/platform-stats',
    adminRateLimit,
    authenticate,
    authorize(['admin', 'analyst']),
    ASIWhatsAppController.getPlatformStats
);

/**
 * @swagger
 * /api/asi-whatsapp/health:
 *   get:
 *     summary: Health check
 *     description: Check the health status of ASI WhatsApp service
 *     tags: [ASI WhatsApp]
 *     responses:
 *       200:
 *         description: Service is healthy
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 data:
 *                   type: object
 *                   properties:
 *                     status:
 *                       type: string
 *                     service:
 *                       type: string
 *                     uptime:
 *                       type: number
 *                     components:
 *                       type: object
 *       500:
 *         description: Service is unhealthy
 */
router.get('/health', ASIWhatsAppController.healthCheck);

// Error handling middleware for this router
router.use((error, req, res, next) => {
    console.error('❌ ASI WhatsApp route error:', error);
    
    if (error.type === 'validation') {
        return res.status(400).json({
            success: false,
            message: 'Validation error',
            errors: error.details
        });
    }
    
    if (error.type === 'rate_limit') {
        return res.status(429).json({
            success: false,
            message: 'Rate limit exceeded',
            retryAfter: error.retryAfter
        });
    }
    
    return res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
    });
});

module.exports = router;
