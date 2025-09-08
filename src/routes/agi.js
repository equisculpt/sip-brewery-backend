const express = require('express');
const router = express.Router();
const agiController = require('../controllers/agiController');
const agiInsightsController = require('../controllers/agiInsightsController');
const agiMacroController = require('../controllers/agiMacroController');
const agiBehaviorController = require('../controllers/agiBehaviorController');
const agiRecommendationsController = require('../controllers/agiRecommendationsController');
const agiAnalyticsController = require('../controllers/agiAnalyticsController');
const agiFeedbackController = require('../controllers/agiFeedbackController');
const agiStatusController = require('../controllers/agiStatusController');
const agiScenarioController = require('../controllers/agiScenarioController');
const agiExplainController = require('../controllers/agiExplainController');
const agiAutonomousController = require('../controllers/agiAutonomousController');
const agiMarketController = require('../controllers/agiMarketController');
const agiRiskController = require('../controllers/agiRiskController');
const agiBehavioralController = require('../controllers/agiBehavioralController');
const agiActionsController = require('../controllers/agiActionsController');
const { authenticateUser } = require('../middleware/auth');

/**
 * @swagger
 * tags:
 *   name: AGI
 *   description: Artificial General Intelligence endpoints for SipBrewery
 */

/**
 * @swagger
 * components:
 *   schemas:
 *     AGIInsight:
 *       type: object
 *       properties:
 *         type:
 *           type: string
 *           enum: [FUND_SWITCH, SIP_UPDATE, REBALANCING, TAX_OPTIMIZATION, GOAL_ADJUSTMENT]
 *         priority:
 *           type: string
 *           enum: [HIGH, MEDIUM, LOW]
 *         action:
 *           type: string
 *         reasoning:
 *           type: string
 *         expectedImpact:
 *           type: string
 *         timeline:
 *           type: string
 *         riskLevel:
 *           type: string
 *           enum: [LOW, MEDIUM, HIGH]
 *         confidence:
 *           type: number
 *           minimum: 0
 *           maximum: 1
 *     AGIRecommendation:
 *       type: object
 *       properties:
 *         type:
 *           type: string
 *         action:
 *           type: string
 *         reasoning:
 *           type: string
 *         priority:
 *           type: string
 *     MacroeconomicImpact:
 *       type: object
 *       properties:
 *         inflation:
 *           type: object
 *         interestRates:
 *           type: object
 *         fiscalPolicy:
 *           type: object
 *         sectorRotation:
 *           type: object
 *         overallImpact:
 *           type: string
 *     UserBehavior:
 *       type: object
 *       properties:
 *         userId:
 *           type: string
 *         action:
 *           type: string
 *         context:
 *           type: object
 *     AGIFeedback:
 *       type: object
 *       properties:
 *         userId:
 *           type: string
 *         insightId:
 *           type: string
 *         feedback:
 *           type: string
 *           enum: [accepted, rejected, implemented, ignored]
 *         rating:
 *           type: number
 *           minimum: 1
 *           maximum: 5
 *         comments:
 *           type: string
 */

// Initialize AGI for user
router.post('/initialize', authenticateUser, agiAutonomousController.initializeAGI);

// Perform autonomous portfolio management
router.post('/autonomous-management', authenticateUser, agiAutonomousController.autonomousPortfolioManagement);

// Get market predictions
router.get('/predictions', authenticateUser, agiMarketController.getMarketPredictions);

// Perform intelligent risk management
router.post('/risk-management', authenticateUser, agiRiskController.intelligentRiskManagement);

// Behavioral nudges
router.post('/behavioral-nudges', authenticateUser, agiBehavioralController.generateBehavioralNudges);

// Execute autonomous actions
router.post('/execute-actions', authenticateUser, agiActionsController.executeAutonomousActions);

// Toggle autonomous mode
router.post('/toggle-autonomous', authenticateUser, agiActionsController.toggleAutonomousMode);

// Get AGI insights
router.get('/insights', authenticateUser, agiActionsController.getAGIInsights);

// Get weekly AGI insights for user
router.get('/insights/:userId', authenticateUser, agiInsightsController.getWeeklyInsights);

// Get personalized investment recommendations
router.get('/recommendations/:userId', authenticateUser, agiRecommendationsController.getPersonalizedRecommendations);

// Analyze macroeconomic impact on portfolio
router.get('/macroeconomic/:userId', authenticateUser, agiMacroController.analyzeMacroeconomicImpact);

// Track user behavior for AGI learning
router.post('/behavior', authenticateUser, agiBehaviorController.trackUserBehavior);

// Learn from market events
router.post('/learn', authenticateUser, agiBehaviorController.learnFromMarketEvents);

// Get AGI capabilities
router.get('/capabilities', agiStatusController.getAGICapabilities);

// Submit feedback for AGI improvement
router.post('/feedback', authenticateUser, agiFeedbackController.submitFeedback);

// Get AGI system status
router.get('/status', agiStatusController.getAGIStatus);

// Dynamic SIP/portfolio allocation
router.post('/dynamic-allocation', authenticateUser, agiController.getDynamicAllocation);

// Scenario simulation (bull/bear/sideways)
router.post('/scenario-simulation', authenticateUser, agiScenarioController.simulateScenario);

// Behavioral nudges
router.post('/behavioral-nudges', authenticateUser, agiController.generateBehavioralNudges);

// Explainability endpoint
router.post('/explain', authenticateUser, agiExplainController.explain);

// Portfolio upload/analytics
router.post('/portfolio/analytics', authenticateUser, agiAnalyticsController.portfolioAnalytics);

// Mutual fund analytics
router.post('/mutual-fund/analytics', authenticateUser, agiAnalyticsController.mutualFundAnalytics);

// Stock vs fund comparison
router.post('/compare/stock-vs-fund', authenticateUser, agiAnalyticsController.compareStockVsFund);

// Scenario simulation & explainability
router.post('/scenario-simulation', authenticateUser, agiController.simulateScenario);
router.post('/explain', authenticateUser, agiController.explain);

// Advanced scenario analytics endpoint
router.post('/advanced-scenario-analytics', authenticateUser, agiScenarioController.advancedScenarioAnalyticsEndpoint);

module.exports = router;