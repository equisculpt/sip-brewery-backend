const express = require('express');
const router = express.Router();
const dashboardController = require('../controllers/dashboardController');
const { authenticateToken } = require('../middleware/auth');

/**
 * @route   GET /api/dashboard
 * @desc    Get complete dashboard data
 * @access  Private
 */
router.get('/',
  authenticateToken,
  dashboardController.getDashboard
);

/**
 * @route   GET /api/dashboard/holdings
 * @desc    Get user holdings
 * @access  Private
 */
router.get('/holdings',
  authenticateToken,
  dashboardController.getHoldings
);

/**
 * @route   GET /api/dashboard/smart-sip
 * @desc    Get Smart SIP Center data
 * @access  Private
 */
router.get('/smart-sip',
  authenticateToken,
  dashboardController.getSmartSIPCenter
);

/**
 * @route   GET /api/dashboard/transactions
 * @desc    Get user transactions
 * @access  Private
 */
router.get('/transactions',
  authenticateToken,
  dashboardController.getTransactions
);

/**
 * @route   GET /api/dashboard/statements
 * @desc    Get statements data
 * @access  Private
 */
router.get('/statements',
  authenticateToken,
  dashboardController.getStatements
);

/**
 * @route   GET /api/dashboard/rewards
 * @desc    Get rewards data
 * @access  Private
 */
router.get('/rewards',
  authenticateToken,
  dashboardController.getRewards
);

/**
 * @route   GET /api/dashboard/referral
 * @desc    Get referral data
 * @access  Private
 */
router.get('/referral',
  authenticateToken,
  dashboardController.getReferralData
);

/**
 * @route   GET /api/dashboard/ai-analytics
 * @desc    Get AI analytics
 * @access  Private
 */
router.get('/ai-analytics',
  authenticateToken,
  dashboardController.getAIAnalytics
);

/**
 * @route   GET /api/dashboard/portfolio-analytics
 * @desc    Get portfolio analytics
 * @access  Private
 */
router.get('/portfolio-analytics',
  authenticateToken,
  dashboardController.getPortfolioAnalytics
);

/**
 * @route   GET /api/dashboard/performance-chart
 * @desc    Get performance chart data
 * @access  Private
 */
router.get('/performance-chart',
  authenticateToken,
  dashboardController.getPerformanceChart
);

/**
 * @route   GET /api/dashboard/profile
 * @desc    Get user profile
 * @access  Private
 */
router.get('/profile',
  authenticateToken,
  dashboardController.getProfile
);

/**
 * @route   GET /api/dashboard/ai-insights
 * @desc    Get AI-driven portfolio insights
 * @access  Private
 */
router.get('/ai-insights',
  authenticateToken,
  dashboardController.getAIInsights
);

/**
 * @route   GET /api/dashboard/predictive-analytics
 * @desc    Get predictive analytics (fund performance, market trends, behavioral insights)
 * @access  Private
 */
router.get('/predictive-analytics',
  authenticateToken,
  dashboardController.getPredictiveAnalytics
);

module.exports = router; 