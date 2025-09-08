const express = require('express');
const router = express.Router();
const tierOutreachController = require('../controllers/tierOutreachController');
const { authenticateUser } = require('../middleware/auth');

/**
 * @swagger
 * components:
 *   schemas:
 *     Location:
 *       type: object
 *       properties:
 *         city:
 *           type: string
 *         state:
 *           type: string
 *         country:
 *           type: string
 *     SimplifiedOnboarding:
 *       type: object
 *       properties:
 *         languageCode:
 *           type: string
 *     VernacularContent:
 *       type: object
 *       properties:
 *         contentType:
 *           type: string
 *           enum: [onboarding, investment_guide, educational, community]
 *         languageCode:
 *           type: string
 */

/**
 * @swagger
 * /api/tier/determine:
 *   post:
 *     summary: Determine user tier based on location and profile
 *     tags: [Tier Outreach]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               location:
 *                 $ref: '#/components/schemas/Location'
 *     responses:
 *       200:
 *         description: User tier determined successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 */
router.post('/determine', authenticateUser, tierOutreachController.determineUserTier);

/**
 * @swagger
 * /api/tier/simplified-onboarding:
 *   post:
 *     summary: Create simplified onboarding flow
 *     tags: [Tier Outreach]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/SimplifiedOnboarding'
 *     responses:
 *       200:
 *         description: Simplified onboarding created successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 */
router.post('/simplified-onboarding', authenticateUser, tierOutreachController.createSimplifiedOnboarding);

/**
 * @swagger
 * /api/tier/vernacular-content:
 *   get:
 *     summary: Get vernacular content for user
 *     tags: [Tier Outreach]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: contentType
 *         required: true
 *         schema:
 *           type: string
 *         description: Type of content
 *       - in: query
 *         name: languageCode
 *         required: true
 *         schema:
 *           type: string
 *         description: Language code
 *     responses:
 *       200:
 *         description: Vernacular content retrieved successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 */
router.get('/vernacular-content', authenticateUser, tierOutreachController.getVernacularContent);

/**
 * @swagger
 * /api/tier/micro-investments:
 *   post:
 *     summary: Create micro-investment options
 *     tags: [Tier Outreach]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Micro-investment options created successfully
 *       401:
 *         description: Unauthorized
 */
router.post('/micro-investments', authenticateUser, tierOutreachController.createMicroInvestmentOptions);

/**
 * @swagger
 * /api/tier/community-features:
 *   post:
 *     summary: Create community features
 *     tags: [Tier Outreach]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               location:
 *                 $ref: '#/components/schemas/Location'
 *     responses:
 *       200:
 *         description: Community features created successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 */
router.post('/community-features', authenticateUser, tierOutreachController.createCommunityFeatures);

/**
 * @swagger
 * /api/tier/financial-literacy:
 *   post:
 *     summary: Create financial literacy program
 *     tags: [Tier Outreach]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Financial literacy program created successfully
 *       401:
 *         description: Unauthorized
 */
router.post('/financial-literacy', authenticateUser, tierOutreachController.createFinancialLiteracyProgram);

/**
 * @swagger
 * /api/tier/features:
 *   get:
 *     summary: Get tier-specific features
 *     tags: [Tier Outreach]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Tier-specific features retrieved successfully
 *       401:
 *         description: Unauthorized
 */
router.get('/features', authenticateUser, tierOutreachController.getTierSpecificFeatures);

/**
 * @swagger
 * /api/tier/categories:
 *   get:
 *     summary: Get tier categories
 *     tags: [Tier Outreach]
 *     responses:
 *       200:
 *         description: Tier categories retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 data:
 *                   type: object
 *                   properties:
 *                     tierCategories:
 *                       type: object
 *                       properties:
 *                         TIER_1:
 *                           type: object
 *                           properties:
 *                             name:
 *                               type: string
 *                             description:
 *                               type: string
 *                             cities:
 *                               type: array
 *                               items:
 *                                 type: string
 *                             features:
 *                               type: array
 *                               items:
 *                                 type: string
 *                             minInvestment:
 *                               type: number
 *                             literacyLevel:
 *                               type: string
 *                         TIER_2:
 *                           type: object
 *                         TIER_3:
 *                           type: object
 *                         RURAL:
 *                           type: object
 *                     totalTiers:
 *                       type: number
 *                 message:
 *                   type: string
 */
router.get('/categories', tierOutreachController.getTierCategories);

/**
 * @swagger
 * /api/tier/micro-investment-options:
 *   get:
 *     summary: Get micro-investment options
 *     tags: [Tier Outreach]
 *     responses:
 *       200:
 *         description: Micro-investment options retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 data:
 *                   type: object
 *                   properties:
 *                     microInvestmentOptions:
 *                       type: object
 *                       properties:
 *                         DAILY_SIP:
 *                           type: object
 *                           properties:
 *                             name:
 *                               type: string
 *                             minAmount:
 *                               type: number
 *                             maxAmount:
 *                               type: number
 *                             frequency:
 *                               type: string
 *                             description:
 *                               type: string
 *                         WEEKLY_SIP:
 *                           type: object
 *                         GOAL_BASED:
 *                           type: object
 *                         FESTIVAL_SAVINGS:
 *                           type: object
 *                     totalOptions:
 *                       type: number
 *                 message:
 *                   type: string
 */
router.get('/micro-investment-options', tierOutreachController.getMicroInvestmentOptions);

/**
 * @swagger
 * /api/tier/financial-literacy-modules:
 *   get:
 *     summary: Get financial literacy modules
 *     tags: [Tier Outreach]
 *     responses:
 *       200:
 *         description: Financial literacy modules retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 data:
 *                   type: object
 *                   properties:
 *                     financialLiteracyModules:
 *                       type: object
 *                       properties:
 *                         BASIC:
 *                           type: object
 *                           properties:
 *                             title:
 *                               type: string
 *                             modules:
 *                               type: array
 *                               items:
 *                                 type: string
 *                             duration:
 *                               type: string
 *                             difficulty:
 *                               type: string
 *                         INTERMEDIATE:
 *                           type: object
 *                         ADVANCED:
 *                           type: object
 *                     totalLevels:
 *                       type: number
 *                 message:
 *                   type: string
 */
router.get('/financial-literacy-modules', tierOutreachController.getFinancialLiteracyModules);

module.exports = router; 