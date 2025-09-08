const express = require('express');
const router = express.Router();
const marketAnalyticsController = require('../controllers/marketAnalyticsController');
const { authenticateUser } = require('../middleware/auth');

/**
 * @swagger
 * /api/market-analytics/scrape-data:
 *   post:
 *     summary: Scrape NSE/BSE daily market data
 *     tags: [Market Analytics]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               date:
 *                 type: string
 *                 format: date
 *                 description: Date for scraping (optional, defaults to today)
 *     responses:
 *       200:
 *         description: Market data scraped successfully
 *       500:
 *         description: Failed to scrape market data
 */
router.post('/scrape-data', authenticateUser, marketAnalyticsController.scrapeMarketData);

/**
 * @swagger
 * /api/market-analytics/sentiment:
 *   post:
 *     summary: Analyze market sentiment
 *     tags: [Market Analytics]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               period:
 *                 type: string
 *                 enum: [1d, 1w, 1m, 3m, 1y]
 *                 description: Analysis period
 *     responses:
 *       200:
 *         description: Market sentiment analyzed successfully
 *       500:
 *         description: Failed to analyze market sentiment
 */
router.post('/sentiment', authenticateUser, marketAnalyticsController.analyzeMarketSentiment);

/**
 * @swagger
 * /api/market-analytics/macro-data:
 *   get:
 *     summary: Fetch macroeconomic data
 *     tags: [Market Analytics]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Macro data fetched successfully
 *       500:
 *         description: Failed to fetch macro data
 */
router.get('/macro-data', authenticateUser, marketAnalyticsController.fetchMacroData);

/**
 * @swagger
 * /api/market-analytics/sector-correlations:
 *   get:
 *     summary: Analyze sector correlations
 *     tags: [Market Analytics]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Sector correlations analyzed successfully
 *       500:
 *         description: Failed to analyze sector correlations
 */
router.get('/sector-correlations', authenticateUser, marketAnalyticsController.analyzeSectorCorrelations);

/**
 * @swagger
 * /api/market-analytics/high-risk-funds:
 *   get:
 *     summary: Predict high-risk funds
 *     tags: [Market Analytics]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: High-risk funds predicted successfully
 *       500:
 *         description: Failed to predict high-risk funds
 */
router.get('/high-risk-funds', authenticateUser, marketAnalyticsController.predictHighRiskFunds);

/**
 * @swagger
 * /api/market-analytics/comprehensive-analysis:
 *   get:
 *     summary: Perform comprehensive market analysis
 *     tags: [Market Analytics]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Comprehensive analysis completed successfully
 *       500:
 *         description: Failed to perform comprehensive analysis
 */
router.get('/comprehensive-analysis', authenticateUser, marketAnalyticsController.performComprehensiveAnalysis);

/**
 * @swagger
 * /api/market-analytics/dashboard:
 *   get:
 *     summary: Get market analytics dashboard data
 *     tags: [Market Analytics]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Dashboard data retrieved successfully
 *       500:
 *         description: Failed to get dashboard data
 */
router.get('/dashboard', authenticateUser, marketAnalyticsController.getMarketAnalyticsDashboard);

/**
 * @swagger
 * /api/market-analytics/portfolio-insights/:portfolioId:
 *   get:
 *     summary: Get market insights for user portfolio
 *     tags: [Market Analytics]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: portfolioId
 *         required: true
 *         schema:
 *           type: string
 *         description: Portfolio ID
 *     responses:
 *       200:
 *         description: Market insights retrieved successfully
 *       500:
 *         description: Failed to get market insights
 */
router.get('/portfolio-insights/:portfolioId', authenticateUser, marketAnalyticsController.getMarketInsightsForPortfolio);

/**
 * @swagger
 * /api/market-analytics/sector-trends:
 *   get:
 *     summary: Get sector performance trends
 *     tags: [Market Analytics]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: period
 *         schema:
 *           type: string
 *           enum: [1d, 1w, 1m, 3m, 1y]
 *         description: Analysis period
 *     responses:
 *       200:
 *         description: Sector trends retrieved successfully
 *       500:
 *         description: Failed to get sector trends
 */
router.get('/sector-trends', authenticateUser, marketAnalyticsController.getSectorPerformanceTrends);

/**
 * @swagger
 * /api/market-analytics/sentiment-trends:
 *   get:
 *     summary: Get market sentiment trends
 *     tags: [Market Analytics]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: period
 *         schema:
 *           type: string
 *           enum: [1d, 1w, 1m, 3m, 1y]
 *         description: Analysis period
 *     responses:
 *       200:
 *         description: Sentiment trends retrieved successfully
 *       500:
 *         description: Failed to get sentiment trends
 */
router.get('/sentiment-trends', authenticateUser, marketAnalyticsController.getMarketSentimentTrends);

/**
 * @swagger
 * /api/market-analytics/macro-indicators:
 *   get:
 *     summary: Get macroeconomic indicators
 *     tags: [Market Analytics]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Macro indicators retrieved successfully
 *       500:
 *         description: Failed to get macro indicators
 */
router.get('/macro-indicators', authenticateUser, marketAnalyticsController.getMacroeconomicIndicators);

/**
 * @swagger
 * /api/market-analytics/risk-assessment:
 *   get:
 *     summary: Get risk assessment summary
 *     tags: [Market Analytics]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Risk assessment retrieved successfully
 *       500:
 *         description: Failed to get risk assessment
 */
router.get('/risk-assessment', authenticateUser, marketAnalyticsController.getRiskAssessmentSummary);

module.exports = router; 