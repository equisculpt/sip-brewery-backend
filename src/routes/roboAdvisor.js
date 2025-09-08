const express = require('express');
const router = express.Router();
const roboAdvisorController = require('../controllers/roboAdvisorController');
const { authenticateUser } = require('../middleware/auth');

/**
 * @swagger
 * /api/robo-advisor/portfolio-review:
 *   post:
 *     summary: Perform portfolio review
 *     tags: [Robo Advisor]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               reviewType:
 *                 type: string
 *                 enum: [daily, weekly, monthly, quarterly]
 *                 description: Review frequency
 *     responses:
 *       200:
 *         description: Portfolio review completed successfully
 *       500:
 *         description: Failed to perform portfolio review
 */
router.post('/portfolio-review', authenticateUser, roboAdvisorController.performPortfolioReview);

/**
 * @swagger
 * /api/robo-advisor/switch-recommendations:
 *   post:
 *     summary: Generate switch recommendations
 *     tags: [Robo Advisor]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               fundName:
 *                 type: string
 *                 description: Specific fund name (optional)
 *     responses:
 *       200:
 *         description: Switch recommendations generated successfully
 *       500:
 *         description: Failed to generate switch recommendations
 */
router.post('/switch-recommendations', authenticateUser, roboAdvisorController.generateSwitchRecommendations);

/**
 * @swagger
 * /api/robo-advisor/tax-harvesting:
 *   get:
 *     summary: Check tax harvesting opportunities
 *     tags: [Robo Advisor]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Tax harvesting opportunities retrieved successfully
 *       500:
 *         description: Failed to check tax harvesting opportunities
 */
router.get('/tax-harvesting', authenticateUser, roboAdvisorController.checkTaxHarvestingOpportunities);

/**
 * @swagger
 * /api/robo-advisor/stpswp-plan:
 *   get:
 *     summary: Generate STP/SWP plans
 *     tags: [Robo Advisor]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: STP/SWP plans generated successfully
 *       500:
 *         description: Failed to generate STP/SWP plans
 */
router.get('/stpswp-plan', authenticateUser, roboAdvisorController.generateSTPSWPPlan);

/**
 * @swagger
 * /api/robo-advisor/sip-deviations:
 *   get:
 *     summary: Check SIP goal deviations
 *     tags: [Robo Advisor]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: SIP goal deviations checked successfully
 *       500:
 *         description: Failed to check SIP goal deviations
 */
router.get('/sip-deviations', authenticateUser, roboAdvisorController.checkSIPGoalDeviations);

/**
 * @swagger
 * /api/robo-advisor/suggestions:
 *   get:
 *     summary: Get robo-advisory suggestions
 *     tags: [Robo Advisor]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: suggestionType
 *         schema:
 *           type: string
 *           enum: [all, portfolio, switches, tax, stpswp, sip]
 *         description: Type of suggestions to retrieve
 *     responses:
 *       200:
 *         description: Robo-advisory suggestions retrieved successfully
 *       500:
 *         description: Failed to get robo-advisory suggestions
 */
router.get('/suggestions', authenticateUser, roboAdvisorController.getRoboAdvisorySuggestions);

/**
 * @swagger
 * /api/robo-advisor/portfolio-health:
 *   get:
 *     summary: Get portfolio health summary
 *     tags: [Robo Advisor]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Portfolio health summary retrieved successfully
 *       500:
 *         description: Failed to get portfolio health summary
 */
router.get('/portfolio-health', authenticateUser, roboAdvisorController.getPortfolioHealthSummary);

/**
 * @swagger
 * /api/robo-advisor/rebalancing:
 *   get:
 *     summary: Get rebalancing recommendations
 *     tags: [Robo Advisor]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Rebalancing recommendations retrieved successfully
 *       500:
 *         description: Failed to get rebalancing recommendations
 */
router.get('/rebalancing', authenticateUser, roboAdvisorController.getRebalancingRecommendations);

/**
 * @swagger
 * /api/robo-advisor/goal-progress:
 *   get:
 *     summary: Get goal progress summary
 *     tags: [Robo Advisor]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Goal progress summary retrieved successfully
 *       500:
 *         description: Failed to get goal progress summary
 */
router.get('/goal-progress', authenticateUser, roboAdvisorController.getGoalProgressSummary);

/**
 * @swagger
 * /api/robo-advisor/risk-assessment:
 *   get:
 *     summary: Get risk assessment summary
 *     tags: [Robo Advisor]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Risk assessment summary retrieved successfully
 *       500:
 *         description: Failed to get risk assessment summary
 */
router.get('/risk-assessment', authenticateUser, roboAdvisorController.getRiskAssessmentSummary);

/**
 * @swagger
 * /api/robo-advisor/market-opportunities:
 *   get:
 *     summary: Get market opportunities
 *     tags: [Robo Advisor]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Market opportunities retrieved successfully
 *       500:
 *         description: Failed to get market opportunities
 */
router.get('/market-opportunities', authenticateUser, roboAdvisorController.getMarketOpportunities);

/**
 * @swagger
 * /api/robo-advisor/dashboard:
 *   get:
 *     summary: Get robo-advisory dashboard
 *     tags: [Robo Advisor]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Robo-advisory dashboard data retrieved successfully
 *       500:
 *         description: Failed to get robo-advisory dashboard data
 */
router.get('/dashboard', authenticateUser, roboAdvisorController.getRoboAdvisoryDashboard);

/**
 * @swagger
 * /api/robo-advisor/execute-action:
 *   post:
 *     summary: Execute robo-advisory action
 *     tags: [Robo Advisor]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               actionType:
 *                 type: string
 *                 enum: [PORTFOLIO_REVIEW, SWITCH_RECOMMENDATION, TAX_HARVESTING, STP_SWP_PLAN, SIP_DEVIATION_CHECK]
 *                 description: Type of action to execute
 *               actionData:
 *                 type: object
 *                 description: Additional data for the action
 *     responses:
 *       200:
 *         description: Robo-advisory action executed successfully
 *       400:
 *         description: Invalid action type
 *       500:
 *         description: Failed to execute robo-advisory action
 */
router.post('/execute-action', authenticateUser, roboAdvisorController.executeRoboAdvisoryAction);

module.exports = router; 