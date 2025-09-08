const express = require('express');
const router = express.Router();
const learningController = require('../controllers/learningController');
const { authenticateUser } = require('../middleware/auth');

/**
 * @swagger
 * tags:
 *   name: Learning
 *   description: Investor education and learning endpoints for SipBrewery
 */

/**
 * @swagger
 * components:
 *   schemas:
 *     LearningPath:
 *       type: object
 *       properties:
 *         currentLevel:
 *           type: string
 *           enum: [BEGINNER, INTERMEDIATE, ADVANCED, EXPERT]
 *         targetLevel:
 *           type: string
 *         topics:
 *           type: array
 *           items:
 *             type: string
 *         progress:
 *           type: object
 *           properties:
 *             completedLessons:
 *               type: number
 *             totalLessons:
 *               type: number
 *             completedTopics:
 *               type: number
 *             totalTopics:
 *               type: number
 *             currentStreak:
 *               type: number
 *             longestStreak:
 *               type: number
 *             totalPoints:
 *               type: number
 *     Lesson:
 *       type: object
 *       properties:
 *         title:
 *           type: string
 *         content:
 *           type: string
 *         example:
 *           type: string
 *         personalizedTip:
 *           type: string
 *         keyTakeaways:
 *           type: array
 *           items:
 *             type: string
 *         nextSteps:
 *           type: string
 *         duration:
 *           type: string
 *     Quiz:
 *       type: object
 *       properties:
 *         questions:
 *           type: array
 *           items:
 *             type: object
 *             properties:
 *               id:
 *                 type: number
 *               question:
 *                 type: string
 *               options:
 *                 type: array
 *                 items:
 *                   type: string
 *               type:
 *                 type: string
 *         timeLimit:
 *           type: number
 *         totalQuestions:
 *           type: number
 *     QuizResult:
 *       type: object
 *       properties:
 *         score:
 *           type: number
 *         correctAnswers:
 *           type: number
 *         totalQuestions:
 *           type: number
 *         results:
 *           type: array
 *           items:
 *             type: object
 *     DailyNudge:
 *       type: object
 *       properties:
 *         title:
 *           type: string
 *         message:
 *           type: string
 *         action:
 *           type: string
 *         estimatedTime:
 *           type: string
 *         topic:
 *           type: string
 *     LearningAnalytics:
 *       type: object
 *       properties:
 *         progress:
 *           type: object
 *         performance:
 *           type: object
 *         achievements:
 *           type: object
 *         recommendations:
 *           type: array
 *           items:
 *             type: string
 */

// Initialize learning path for user
router.post('/initialize/:userId', authenticateUser, learningController.initializeLearningPath);

// Get personalized lesson
router.get('/lesson/:userId/:topic/:lessonIndex', authenticateUser, learningController.getPersonalizedLesson);

// Start quiz
router.post('/quiz/start/:userId', authenticateUser, learningController.startQuiz);

// Submit quiz answers
router.post('/quiz/submit/:userId/:quizId', authenticateUser, learningController.submitQuiz);

// Get daily learning nudge
router.get('/nudge/:userId', authenticateUser, learningController.getDailyNudge);

// Track learning progress
router.post('/progress/:userId', authenticateUser, learningController.trackLearningProgress);

// Get learning analytics
router.get('/analytics/:userId', authenticateUser, learningController.getLearningAnalytics);

// Get learning topics
router.get('/topics', learningController.getLearningTopics);

// Get learning levels
router.get('/levels', learningController.getLearningLevels);

// Get user learning profile
router.get('/profile/:userId', authenticateUser, learningController.getUserLearningProfile);

module.exports = router; 