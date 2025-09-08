const express = require('express');
const router = express.Router();
const analyticsController = require('../controllers/analyticsController');
const auth = require('../middleware/auth');

// Apply authentication middleware to all routes
router.use(auth);

/**
 * @route   GET /api/analytics/performance
 * @desc    Get comprehensive performance analytics with chart data
 * @access  Private
 */
router.get('/performance', analyticsController.getPerformanceAnalytics);

/**
 * @route   POST /api/analytics/chart-data
 * @desc    Get chart data for various chart types
 * @access  Private
 */
router.post('/chart-data', analyticsController.getChartData);

/**
 * @route   POST /api/analytics/sip-projections
 * @desc    Calculate SIP future value and projections
 * @access  Private
 */
router.post('/sip-projections', analyticsController.calculateSIPProjections);

/**
 * @route   POST /api/analytics/goal-based-investment
 * @desc    Calculate goal-based investment requirements
 * @access  Private
 */
router.post('/goal-based-investment', analyticsController.calculateGoalBasedInvestment);

/**
 * @route   GET /api/analytics/risk-profiling
 * @desc    Get risk profiling and assessment
 * @access  Private
 */
router.get('/risk-profiling', analyticsController.getRiskProfiling);

/**
 * @route   GET /api/analytics/nav-history/:fundCode
 * @desc    Get NAV history with calculations
 * @access  Private
 */
router.get('/nav-history/:fundCode', analyticsController.getNAVHistory);

/**
 * @route   GET /api/analytics/tax-calculations
 * @desc    Calculate tax implications and optimization
 * @access  Private
 */
router.get('/tax-calculations', analyticsController.getTaxCalculations);

/**
 * @route   GET /api/analytics/xirr
 * @desc    Get XIRR analytics
 * @access  Private
 */
router.get('/xirr', analyticsController.getXIRRAnalytics);

/**
 * @route   GET /api/analytics/portfolio-comparison
 * @desc    Get portfolio comparison analytics
 * @access  Private
 */
router.get('/portfolio-comparison', analyticsController.getPortfolioComparison);

/**
 * @route   GET /api/analytics/dashboard
 * @desc    Get comprehensive dashboard analytics
 * @access  Private
 */
router.get('/dashboard', analyticsController.getDashboardAnalytics);

/**
 * @route   GET /api/analytics/platform
 * @desc    Get platform analytics (admin)
 * @access  Private
 */
router.get('/platform', analyticsController.getPlatformAnalytics);

/**
 * @route   GET /api/analytics/regional
 * @desc    Get regional analytics
 * @access  Private
 */
router.get('/regional', analyticsController.getRegionalAnalytics);

/**
 * @route   GET /api/analytics/agent
 * @desc    Get agent analytics
 * @access  Private
 */
router.get('/agent', analyticsController.getAgentAnalytics);

/**
 * @route   POST /api/analytics/compare-funds
 * @desc    Comprehensive fund comparison with detailed analysis and ratings
 * @access  Private
 */
router.post('/compare-funds', analyticsController.compareFunds);

module.exports = router; 