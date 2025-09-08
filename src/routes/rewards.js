const express = require('express');
const router = express.Router();
const rewardsController = require('../controllers/rewardsController');
const authenticateUser = require('../middleware/authenticateUser');
const { verifyToken } = require('../middleware/adminAuth');

// User routes (require authentication)
router.use('/summary', authenticateUser);
router.use('/transactions', authenticateUser);
router.use('/simulate-sip-reward', authenticateUser);
router.use('/referral-code', authenticateUser);
router.use('/leaderboard', authenticateUser);

// Admin routes (require admin authentication)
router.use('/admin', verifyToken);

// User API endpoints
router.get('/summary', rewardsController.getRewardSummary);
router.get('/transactions', rewardsController.getRewardTransactions);
router.post('/simulate-sip-reward', rewardsController.simulateSipReward);
router.get('/referral-code', rewardsController.getReferralCode);
router.get('/leaderboard', rewardsController.getReferralLeaderboard);

// Internal API endpoints (called by transaction system)
router.post('/award-sip-points', rewardsController.awardSipPoints);
router.post('/validate-referral', rewardsController.validateReferral);
router.post('/revoke-referral', rewardsController.revokeReferralBonus);

// Admin API endpoints
router.post('/admin/mark-paid', rewardsController.markRewardAsPaid);
router.get('/admin/unpaid', rewardsController.getUnpaidRewards);
router.get('/admin/user/:userId', rewardsController.getUserRewardHistory);
router.get('/admin/export-csv', rewardsController.exportUnpaidRewardsCSV);

module.exports = router; 