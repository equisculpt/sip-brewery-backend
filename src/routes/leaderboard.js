const express = require('express');
const router = express.Router();
const leaderboardController = require('../controllers/leaderboardController');
const { authenticateToken } = require('../middleware/auth');

/**
 * @route   GET /api/leaderboard/:duration
 * @desc    Get leaderboard for specific duration (1M, 3M, 6M, 1Y, 3Y)
 * @access  Public
 */
router.get('/:duration', leaderboardController.getLeaderboard);

/**
 * @route   GET /api/leaderboard
 * @desc    Get all leaderboards
 * @access  Public
 */
router.get('/', leaderboardController.getAllLeaderboards);

/**
 * @route   GET /api/leaderboard/:duration/stats
 * @desc    Get leaderboard statistics for specific duration
 * @access  Public
 */
router.get('/:duration/stats', leaderboardController.getLeaderboardStats);

// Apply authentication middleware to protected routes
router.use(authenticateToken);

/**
 * @route   POST /api/leaderboard/portfolio/copy
 * @desc    Copy portfolio from leaderboard
 * @access  Private
 */
router.post('/portfolio/copy', leaderboardController.copyPortfolio);

/**
 * @route   GET /api/leaderboard/user/history
 * @desc    Get user's leaderboard history
 * @access  Private
 */
router.get('/user/history', leaderboardController.getUserLeaderboardHistory);

/**
 * @route   GET /api/leaderboard/portfolio/copy/history
 * @desc    Get user's portfolio copy history
 * @access  Private
 */
router.get('/portfolio/copy/history', leaderboardController.getPortfolioCopyHistory);

/**
 * @route   GET /api/leaderboard/:duration/user/rank
 * @desc    Get user's current rank in leaderboard
 * @access  Private
 */
router.get('/:duration/user/rank', leaderboardController.getUserRank);

/**
 * @route   POST /api/leaderboard/generate
 * @desc    Manually trigger leaderboard generation (admin)
 * @access  Private
 */
router.post('/generate', leaderboardController.generateLeaderboards);

/**
 * @route   POST /api/leaderboard/update-xirr
 * @desc    Update XIRR for all portfolios (admin)
 * @access  Private
 */
router.post('/update-xirr', leaderboardController.updateAllXIRR);

module.exports = router; 