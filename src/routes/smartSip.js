const express = require('express');
const router = express.Router();
const smartSipController = require('../controllers/smartSipController');
const { authenticateToken } = require('../middleware/auth');

// Apply authentication middleware to all routes
router.use(authenticateToken);

/**
 * @route   POST /api/sip/start
 * @desc    Start a new SIP (static or smart)
 * @access  Private
 */
router.post('/start', smartSipController.startSIP);

/**
 * @route   GET /api/sip/recommendation
 * @desc    Get current SIP recommendation based on market conditions
 * @access  Private
 */
router.get('/recommendation', smartSipController.getSIPRecommendation);

/**
 * @route   GET /api/sip/details
 * @desc    Get user's SIP details and current status
 * @access  Private
 */
router.get('/details', smartSipController.getSIPDetails);

/**
 * @route   PUT /api/sip/preferences
 * @desc    Update SIP preferences (risk tolerance, AI settings, etc.)
 * @access  Private
 */
router.put('/preferences', smartSipController.updateSIPPreferences);

/**
 * @route   PUT /api/sip/status
 * @desc    Update SIP status (pause/resume/stop)
 * @access  Private
 */
router.put('/status', smartSipController.updateSIPStatus);

/**
 * @route   POST /api/sip/execute
 * @desc    Execute SIP manually (for testing or immediate execution)
 * @access  Private
 */
router.post('/execute', smartSipController.executeSIP);

/**
 * @route   GET /api/sip/analytics
 * @desc    Get SIP analytics and performance metrics
 * @access  Private
 */
router.get('/analytics', smartSipController.getSIPAnalytics);

/**
 * @route   GET /api/sip/history
 * @desc    Get SIP history with optional limit parameter
 * @access  Private
 */
router.get('/history', smartSipController.getSIPHistory);

/**
 * @route   GET /api/sip/market-analysis
 * @desc    Get current market analysis for frontend display
 * @access  Private
 */
router.get('/market-analysis', smartSipController.getMarketAnalysis);

/**
 * @route   GET /api/sip/all
 * @desc    Get all active SIPs (admin endpoint)
 * @access  Private
 */
router.get('/all', smartSipController.getAllUserSIPs);
// Add alias for /all-users to match test expectation
router.get('/all-users', smartSipController.getAllUserSIPs);

module.exports = router; 