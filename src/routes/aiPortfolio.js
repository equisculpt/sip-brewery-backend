const express = require('express');
const router = express.Router();
const aiPortfolioController = require('../controllers/aiPortfolioController');
const { authenticateToken } = require('../middleware/auth');
const { apiRateLimiter } = require('../middleware/rateLimiter');

// Apply authentication and rate limiting to all routes
router.use(authenticateToken);
router.use(apiRateLimiter);

/**
 * @route POST /api/ai-portfolio/optimize
 * @desc Optimize user portfolio using AI
 * @access Private
 */
router.post('/optimize', aiPortfolioController.optimizePortfolio);

/**
 * @route GET /api/ai-portfolio/analysis
 * @desc Get detailed portfolio analysis
 * @access Private
 */
router.get('/analysis', aiPortfolioController.getPortfolioAnalysis);

/**
 * @route GET /api/ai-portfolio/recommendations
 * @desc Get AI-powered investment recommendations
 * @access Private
 */
router.get('/recommendations', aiPortfolioController.getRecommendations);

/**
 * @route POST /api/ai-portfolio/predict-performance
 * @desc Predict fund performance using AI
 * @access Private
 */
router.post('/predict-performance', aiPortfolioController.predictFundPerformance);

/**
 * @route POST /api/ai-portfolio/risk-assessment
 * @desc Get risk assessment for allocation
 * @access Private
 */
router.post('/risk-assessment', aiPortfolioController.getRiskAssessment);

/**
 * @route GET /api/ai-portfolio/tax-strategies
 * @desc Get tax optimization strategies
 * @access Private
 */
router.get('/tax-strategies', aiPortfolioController.getTaxStrategies);

/**
 * @route POST /api/ai-portfolio/expected-returns
 * @desc Calculate expected returns for allocation
 * @access Private
 */
router.post('/expected-returns', aiPortfolioController.getExpectedReturns);

module.exports = router; 