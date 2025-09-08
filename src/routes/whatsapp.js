const express = require('express');
const router = express.Router();
const whatsAppController = require('../controllers/whatsAppController');
const authenticateUser = require('../middleware/authenticateUser');
const { verifyToken } = require('../middleware/adminAuth');
const { validateRequest } = require('../middleware/validation');
const { whatsAppRateLimiter } = require('../middleware/rateLimiter');

// WhatsApp webhook (no auth required for Twilio)
router.get('/webhook', whatsAppController.handleWebhook);
router.post('/webhook', whatsAppController.handleWebhook);

// Add test endpoints for WhatsApp API with rate limiting
router.post('/send', whatsAppRateLimiter, whatsAppController.sendMessage);
router.post('/bulk-send', whatsAppRateLimiter, whatsAppController.sendBulkMessages);
router.get('/sessions', whatsAppController.getAllActiveSessions);
router.get('/sessions/:phoneNumber', whatsAppController.getSessionByPhoneNumber);
router.put('/sessions/:phoneNumber', whatsAppController.updateSession);
router.delete('/sessions/:phoneNumber', whatsAppController.deleteSession);
router.get('/messages/:phoneNumber', whatsAppController.getMessagesByPhoneNumber);

// Health check
router.get('/health', whatsAppController.healthCheck);

// Admin endpoints (require authentication)
router.use('/admin', verifyToken);

// Send test message
router.post('/admin/test-message', 
  validateRequest({
    body: {
      phoneNumber: { type: 'string', required: true },
      message: { type: 'string', required: true }
    }
  }),
  whatsAppController.sendTestMessage
);

// Get client status
router.get('/admin/client-status', whatsAppController.getClientStatus);

// Test connection
router.post('/admin/test-connection', whatsAppController.testConnection);

// Get AI status
router.get('/admin/ai-status', whatsAppController.getAiStatus);

// Test AI analysis
router.post('/admin/test-ai', 
  validateRequest({
    body: {
      fundName: { type: 'string', required: true }
    }
  }),
  whatsAppController.testAiAnalysis
);

// Disclaimer management
router.get('/admin/disclaimer-stats', whatsAppController.getDisclaimerStats);
router.delete('/admin/disclaimer-reset/:phoneNumber', whatsAppController.resetDisclaimerCounter);

// Get session statistics
router.get('/admin/stats', whatsAppController.getSessionStats);

// Get recent sessions
router.get('/admin/sessions', whatsAppController.getRecentSessions);

// Get session details
router.get('/admin/sessions/:phoneNumber', whatsAppController.getSessionDetails);

// Get message analytics
router.get('/admin/analytics', whatsAppController.getMessageAnalytics);

module.exports = router; 