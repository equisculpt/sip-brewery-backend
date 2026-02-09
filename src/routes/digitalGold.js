const express = require('express');
const router = express.Router();
const digitalGoldService = require('../services/digitalGoldService');
const goldMFHybridService = require('../services/goldMFHybridService');
const { authenticateToken } = require('../middleware/auth');
const logger = require('../utils/logger');

/**
 * Digital Gold API Routes
 * All routes require authentication
 */

// Get current gold price
router.get('/price/:goldType?', authenticateToken, async (req, res) => {
  try {
    const goldType = req.params.goldType || '24K';
    const priceData = await digitalGoldService.getCurrentGoldPrice(goldType);
    
    res.json({
      success: true,
      data: priceData
    });
  } catch (error) {
    logger.error('Failed to fetch gold price', { error: error.message });
    res.status(500).json({
      success: false,
      error: 'Failed to fetch gold price'
    });
  }
});

// Buy digital gold
router.post('/buy', authenticateToken, async (req, res) => {
  try {
    const { amount, goldType = '24K', paymentMethod = 'UPI' } = req.body;
    const userId = req.user.id;

    if (!amount || amount <= 0) {
      return res.status(400).json({
        success: false,
        error: 'Invalid amount'
      });
    }

    const result = await digitalGoldService.buyGold(
      userId,
      amount,
      goldType,
      paymentMethod
    );

    res.json(result);
  } catch (error) {
    logger.error('Gold purchase failed', { error: error.message });
    res.status(500).json({
      success: false,
      error: 'Gold purchase failed'
    });
  }
});

// Sell digital gold
router.post('/sell', authenticateToken, async (req, res) => {
  try {
    const { quantityGrams, goldType = '24K' } = req.body;
    const userId = req.user.id;

    if (!quantityGrams || quantityGrams <= 0) {
      return res.status(400).json({
        success: false,
        error: 'Invalid quantity'
      });
    }

    const result = await digitalGoldService.sellGold(
      userId,
      quantityGrams,
      goldType
    );

    res.json(result);
  } catch (error) {
    logger.error('Gold sale failed', { error: error.message });
    res.status(500).json({
      success: false,
      error: 'Gold sale failed'
    });
  }
});

// Create Gold SIP
router.post('/sip/create', authenticateToken, async (req, res) => {
  try {
    const { amount, frequency = 'MONTHLY', goldType = '24K', startDate } = req.body;
    const userId = req.user.id;

    if (!amount || amount <= 0) {
      return res.status(400).json({
        success: false,
        error: 'Invalid SIP amount'
      });
    }

    const result = await digitalGoldService.createGoldSIP(
      userId,
      amount,
      frequency,
      goldType,
      startDate
    );

    res.json(result);
  } catch (error) {
    logger.error('Gold SIP creation failed', { error: error.message });
    res.status(500).json({
      success: false,
      error: 'Gold SIP creation failed'
    });
  }
});

// Get user's gold holdings
router.get('/holdings', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const holdings = await digitalGoldService.getUserGoldHoldings(userId);

    res.json({
      success: true,
      data: holdings
    });
  } catch (error) {
    logger.error('Failed to fetch gold holdings', { error: error.message });
    res.status(500).json({
      success: false,
      error: 'Failed to fetch gold holdings'
    });
  }
});

// Get gold portfolio analytics
router.get('/analytics', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const analytics = await digitalGoldService.getGoldPortfolioAnalytics(userId);

    res.json({
      success: true,
      data: analytics
    });
  } catch (error) {
    logger.error('Failed to fetch gold analytics', { error: error.message });
    res.status(500).json({
      success: false,
      error: 'Failed to fetch gold analytics'
    });
  }
});

// Request physical gold delivery
router.post('/delivery/request', authenticateToken, async (req, res) => {
  try {
    const { quantityGrams, goldType = '24K', deliveryAddress } = req.body;
    const userId = req.user.id;

    if (!quantityGrams || !deliveryAddress) {
      return res.status(400).json({
        success: false,
        error: 'Quantity and delivery address are required'
      });
    }

    const result = await digitalGoldService.requestGoldDelivery(
      userId,
      quantityGrams,
      goldType,
      deliveryAddress
    );

    res.json(result);
  } catch (error) {
    logger.error('Gold delivery request failed', { error: error.message });
    res.status(500).json({
      success: false,
      error: 'Gold delivery request failed'
    });
  }
});

// Get gold-backed loan eligibility
router.get('/loan/eligibility', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const eligibility = await digitalGoldService.getGoldLoanEligibility(userId);

    res.json({
      success: true,
      data: eligibility
    });
  } catch (error) {
    logger.error('Failed to check loan eligibility', { error: error.message });
    res.status(500).json({
      success: false,
      error: 'Failed to check loan eligibility'
    });
  }
});

// HYBRID GOLD-MF ROUTES

// Get optimal gold-MF allocation
router.get('/hybrid/allocation', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const { riskProfile = 'moderate' } = req.query;

    const allocation = await goldMFHybridService.getOptimalAllocation(
      userId,
      riskProfile
    );

    res.json({
      success: true,
      data: allocation
    });
  } catch (error) {
    logger.error('Failed to calculate allocation', { error: error.message });
    res.status(500).json({
      success: false,
      error: 'Failed to calculate allocation'
    });
  }
});

// Auto-rebalance portfolio
router.post('/hybrid/rebalance', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const { riskProfile = 'moderate' } = req.body;

    const result = await goldMFHybridService.autoRebalance(userId, riskProfile);

    res.json(result);
  } catch (error) {
    logger.error('Auto-rebalancing failed', { error: error.message });
    res.status(500).json({
      success: false,
      error: 'Auto-rebalancing failed'
    });
  }
});

// Analyze gold as hedge
router.get('/hybrid/hedge-analysis', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const analysis = await goldMFHybridService.analyzeGoldHedge(userId);

    res.json({
      success: true,
      data: analysis
    });
  } catch (error) {
    logger.error('Hedge analysis failed', { error: error.message });
    res.status(500).json({
      success: false,
      error: 'Hedge analysis failed'
    });
  }
});

// Get tax-efficient allocation
router.post('/hybrid/tax-efficient', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const { investmentAmount, holdingPeriod = 36 } = req.body;

    if (!investmentAmount) {
      return res.status(400).json({
        success: false,
        error: 'Investment amount is required'
      });
    }

    const allocation = await goldMFHybridService.getTaxEfficientAllocation(
      userId,
      investmentAmount,
      holdingPeriod
    );

    res.json({
      success: true,
      data: allocation
    });
  } catch (error) {
    logger.error('Tax-efficient allocation failed', { error: error.message });
    res.status(500).json({
      success: false,
      error: 'Tax-efficient allocation failed'
    });
  }
});

// Get goal-based allocation
router.post('/hybrid/goal-based', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const goal = req.body;

    if (!goal.goalType || !goal.targetAmount || !goal.timeHorizon) {
      return res.status(400).json({
        success: false,
        error: 'Goal type, target amount, and time horizon are required'
      });
    }

    const allocation = await goldMFHybridService.getGoalBasedAllocation(
      userId,
      goal
    );

    res.json({
      success: true,
      data: allocation
    });
  } catch (error) {
    logger.error('Goal-based allocation failed', { error: error.message });
    res.status(500).json({
      success: false,
      error: 'Goal-based allocation failed'
    });
  }
});

module.exports = router;
