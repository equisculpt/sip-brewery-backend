const express = require('express');
const router = express.Router();
const socialInvestingController = require('../controllers/socialInvestingController');
const { authenticateUser } = require('../middleware/auth');

/**
 * @swagger
 * components:
 *   schemas:
 *     FollowRequest:
 *       type: object
 *       properties:
 *         followingId:
 *           type: string
 *           description: ID of user to follow
 *     SharePortfolioRequest:
 *       type: object
 *       properties:
 *         shareType:
 *           type: string
 *           enum: [public, private, followers]
 *           description: Type of sharing
 *     ChallengeRequest:
 *       type: object
 *       properties:
 *         challengeType:
 *           type: string
 *           enum: [sip_challenge, diversification_challenge, goal_achievement]
 *           description: Type of challenge
 *     AwardPointsRequest:
 *       type: object
 *       properties:
 *         points:
 *           type: number
 *           description: Points to award
 *         reason:
 *           type: string
 *           description: Reason for awarding points
 *     GameRequest:
 *       type: object
 *       properties:
 *         gameType:
 *           type: string
 *           enum: [investment_quiz, risk_assessment, portfolio_simulation]
 *           description: Type of educational game
 *     GameAnswerRequest:
 *       type: object
 *       properties:
 *         gameId:
 *           type: string
 *           description: Game ID
 *         answer:
 *           type: number
 *           description: Answer index
 *     CommunityEventRequest:
 *       type: object
 *       properties:
 *         eventData:
 *           type: object
 *           properties:
 *             title:
 *               type: string
 *             description:
 *               type: string
 *             type:
 *               type: string
 *             category:
 *               type: string
 *             startDate:
 *               type: string
 *               format: date-time
 *             endDate:
 *               type: string
 *               format: date-time
 *             location:
 *               type: string
 *             maxParticipants:
 *               type: number
 *             rewardPoints:
 *               type: number
 *             rewardBadges:
 *               type: array
 *               items:
 *                 type: string
 */

/**
 * @swagger
 * /api/social/profile:
 *   post:
 *     summary: Create social profile for user
 *     tags: [Social Investing]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Social profile created successfully
 *       401:
 *         description: Unauthorized
 */
router.post('/profile', authenticateUser, socialInvestingController.createSocialProfile);

/**
 * @swagger
 * /api/social/follow:
 *   post:
 *     summary: Follow another investor
 *     tags: [Social Investing]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/FollowRequest'
 *     responses:
 *       200:
 *         description: Successfully followed investor
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 */
router.post('/follow', authenticateUser, socialInvestingController.followInvestor);

/**
 * @swagger
 * /api/social/share-portfolio:
 *   post:
 *     summary: Share portfolio performance
 *     tags: [Social Investing]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/SharePortfolioRequest'
 *     responses:
 *       200:
 *         description: Portfolio performance shared successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 */
router.post('/share-portfolio', authenticateUser, socialInvestingController.sharePortfolioPerformance);

/**
 * @swagger
 * /api/social/challenge:
 *   post:
 *     summary: Create investment challenge
 *     tags: [Social Investing]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/ChallengeRequest'
 *     responses:
 *       200:
 *         description: Investment challenge created successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 */
router.post('/challenge', authenticateUser, socialInvestingController.createInvestmentChallenge);

/**
 * @swagger
 * /api/social/award-points:
 *   post:
 *     summary: Award points to user
 *     tags: [Social Investing]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/AwardPointsRequest'
 *     responses:
 *       200:
 *         description: Points awarded successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 */
router.post('/award-points', authenticateUser, socialInvestingController.awardPoints);

/**
 * @swagger
 * /api/social/leaderboard:
 *   get:
 *     summary: Get leaderboard
 *     tags: [Social Investing]
 *     parameters:
 *       - in: query
 *         name: category
 *         schema:
 *           type: string
 *           enum: [overall, points, performance, consistency, community]
 *         description: Leaderboard category
 *       - in: query
 *         name: timeFrame
 *         schema:
 *           type: string
 *           enum: [daily, weekly, monthly, yearly]
 *         description: Time frame for leaderboard
 *     responses:
 *       200:
 *         description: Leaderboard retrieved successfully
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
 *                     leaderboard:
 *                       type: array
 *                       items:
 *                         type: object
 *                         properties:
 *                           rank:
 *                             type: number
 *                           userId:
 *                             type: string
 *                           name:
 *                             type: string
 *                           username:
 *                             type: string
 *                           avatar:
 *                             type: string
 *                           score:
 *                             type: number
 *                           level:
 *                             type: number
 *                           achievements:
 *                             type: number
 *                     category:
 *                       type: string
 *                     timeFrame:
 *                       type: string
 *                     totalParticipants:
 *                       type: number
 *                 message:
 *                   type: string
 */
router.get('/leaderboard', socialInvestingController.getLeaderboard);

/**
 * @swagger
 * /api/social/educational-game:
 *   post:
 *     summary: Create educational game
 *     tags: [Social Investing]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/GameRequest'
 *     responses:
 *       200:
 *         description: Educational game created successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 */
router.post('/educational-game', authenticateUser, socialInvestingController.createEducationalGame);

/**
 * @swagger
 * /api/social/game-answer:
 *   post:
 *     summary: Submit game answer
 *     tags: [Social Investing]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/GameAnswerRequest'
 *     responses:
 *       200:
 *         description: Game answer submitted successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 */
router.post('/game-answer', authenticateUser, socialInvestingController.submitGameAnswer);

/**
 * @swagger
 * /api/social/achievements:
 *   get:
 *     summary: Get user achievements
 *     tags: [Social Investing]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: User achievements retrieved successfully
 *       401:
 *         description: Unauthorized
 */
router.get('/achievements', authenticateUser, socialInvestingController.getUserAchievements);

/**
 * @swagger
 * /api/social/community-event:
 *   post:
 *     summary: Create community event
 *     tags: [Social Investing]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/CommunityEventRequest'
 *     responses:
 *       200:
 *         description: Community event created successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 */
router.post('/community-event', authenticateUser, socialInvestingController.createCommunityEvent);

/**
 * @swagger
 * /api/social/features:
 *   get:
 *     summary: Get social features
 *     tags: [Social Investing]
 *     responses:
 *       200:
 *         description: Social features retrieved successfully
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
 *                     socialFeatures:
 *                       type: object
 *                       properties:
 *                         SOCIAL_TRADING:
 *                           type: object
 *                           properties:
 *                             name:
 *                               type: string
 *                             description:
 *                               type: string
 *                             features:
 *                               type: array
 *                               items:
 *                                 type: string
 *                         COMMUNITY_FEATURES:
 *                           type: object
 *                         GAMIFICATION:
 *                           type: object
 *                         EDUCATIONAL_GAMES:
 *                           type: object
 *                     totalFeatures:
 *                       type: number
 *                 message:
 *                   type: string
 */
router.get('/features', socialInvestingController.getSocialFeatures);

/**
 * @swagger
 * /api/social/achievement-types:
 *   get:
 *     summary: Get achievement types
 *     tags: [Social Investing]
 *     responses:
 *       200:
 *         description: Achievement types retrieved successfully
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
 *                     achievementTypes:
 *                       type: object
 *                       properties:
 *                         INVESTMENT_ACHIEVEMENTS:
 *                           type: object
 *                           properties:
 *                             first_investment:
 *                               type: object
 *                               properties:
 *                                 name:
 *                                   type: string
 *                                 description:
 *                                   type: string
 *                                 points:
 *                                   type: number
 *                                 icon:
 *                                   type: string
 *                                 category:
 *                                   type: string
 *                             sip_streak:
 *                               type: object
 *                             portfolio_growth:
 *                               type: object
 *                             diversification:
 *                               type: object
 *                         LEARNING_ACHIEVEMENTS:
 *                           type: object
 *                         SOCIAL_ACHIEVEMENTS:
 *                           type: object
 *                     totalCategories:
 *                       type: number
 *                 message:
 *                   type: string
 */
router.get('/achievement-types', socialInvestingController.getAchievementTypes);

/**
 * @swagger
 * /api/social/challenge-types:
 *   get:
 *     summary: Get challenge types
 *     tags: [Social Investing]
 *     responses:
 *       200:
 *         description: Challenge types retrieved successfully
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
 *                     challengeTypes:
 *                       type: object
 *                       properties:
 *                         INVESTMENT_CHALLENGES:
 *                           type: object
 *                           properties:
 *                             sip_challenge:
 *                               type: object
 *                               properties:
 *                                 name:
 *                                   type: string
 *                                 description:
 *                                   type: string
 *                                 duration:
 *                                   type: string
 *                                 reward:
 *                                   type: number
 *                                 difficulty:
 *                                   type: string
 *                                 category:
 *                                   type: string
 *                             diversification_challenge:
 *                               type: object
 *                             goal_achievement:
 *                               type: object
 *                         LEARNING_CHALLENGES:
 *                           type: object
 *                         SOCIAL_CHALLENGES:
 *                           type: object
 *                     totalCategories:
 *                       type: number
 *                 message:
 *                   type: string
 */
router.get('/challenge-types', socialInvestingController.getChallengeTypes);

module.exports = router; 