const logger = require('../utils/logger');
const learningModule = require('../services/learningModule');
const { validateObjectId } = require('../middleware/validation');

class LearningController {
  /**
   * @swagger
   * /api/learning/initialize/{userId}:
   *   post:
   *     summary: Initialize learning path for user
   *     description: Create personalized learning path based on user's current knowledge level
   *     tags: [Learning]
   *     parameters:
   *       - in: path
   *         name: userId
   *         required: true
   *         schema:
   *           type: string
   *         description: User ID
   *     responses:
   *       200:
   *         description: Learning path initialized successfully
   *         content:
   *           application/json:
   *             schema:
   *               type: object
   *               properties:
   *                 success:
   *                   type: boolean
   *                 data:
   *                   $ref: '#/components/schemas/LearningPath'
   *       400:
   *         description: Invalid user ID
   *       404:
   *         description: User not found
   *       500:
   *         description: Internal server error
   */
  async initializeLearningPath(req, res) {
    try {
      const { userId } = req.params;

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      const result = await learningModule.initializeLearningPath(userId);

      if (!result.success) {
        return res.status(404).json(result);
      }

      res.json(result);
    } catch (error) {
      logger.error('Error in initializeLearningPath', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to initialize learning path',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/learning/lesson/{userId}/{topic}/{lessonIndex}:
   *   get:
   *     summary: Get personalized lesson
   *     description: Get AI-generated personalized lesson content
   *     tags: [Learning]
   *     parameters:
   *       - in: path
   *         name: userId
   *         required: true
   *         schema:
   *           type: string
   *         description: User ID
   *       - in: path
   *         name: topic
   *         required: true
   *         schema:
   *           type: string
   *         description: Learning topic
   *       - in: path
   *         name: lessonIndex
   *         required: true
   *         schema:
   *           type: integer
   *         description: Lesson index
   *     responses:
   *       200:
   *         description: Lesson retrieved successfully
   *         content:
   *           application/json:
   *             schema:
   *               type: object
   *               properties:
   *                 success:
   *                   type: boolean
   *                 data:
   *                   $ref: '#/components/schemas/Lesson'
   *       400:
   *         description: Invalid parameters
   *       404:
   *         description: Lesson not found
   *       500:
   *         description: Internal server error
   */
  async getPersonalizedLesson(req, res) {
    try {
      const { userId, topic, lessonIndex } = req.params;

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      const lessonIndexNum = parseInt(lessonIndex);
      if (isNaN(lessonIndexNum) || lessonIndexNum < 0) {
        return res.status(400).json({
          success: false,
          message: 'Invalid lesson index'
        });
      }

      const result = await learningModule.getPersonalizedLesson(userId, topic, lessonIndexNum);

      if (!result.success) {
        return res.status(404).json(result);
      }

      res.json(result);
    } catch (error) {
      logger.error('Error in getPersonalizedLesson', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get personalized lesson',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/learning/quiz/start/{userId}:
   *   post:
   *     summary: Start quiz
   *     description: Start a quiz for a specific topic
   *     tags: [Learning]
   *     parameters:
   *       - in: path
   *         name: userId
   *         required: true
   *         schema:
   *           type: string
   *         description: User ID
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             required:
   *               - topic
   *             properties:
   *               topic:
   *                 type: string
   *                 description: Quiz topic
   *     responses:
   *       200:
   *         description: Quiz started successfully
   *         content:
   *           application/json:
   *             schema:
   *               type: object
   *               properties:
   *                 success:
   *                   type: boolean
   *                 data:
   *                   $ref: '#/components/schemas/Quiz'
   *       400:
   *         description: Invalid request data
   *       500:
   *         description: Internal server error
   */
  async startQuiz(req, res) {
    try {
      const { userId } = req.params;
      const { topic } = req.body;

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      if (!topic) {
        return res.status(400).json({
          success: false,
          message: 'Topic is required'
        });
      }

      const result = await learningModule.startQuiz(userId, topic);

      if (!result.success) {
        return res.status(404).json(result);
      }

      res.json(result);
    } catch (error) {
      logger.error('Error in startQuiz', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to start quiz',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/learning/quiz/submit/{userId}/{quizId}:
   *   post:
   *     summary: Submit quiz answers
   *     description: Submit quiz answers and get results
   *     tags: [Learning]
   *     parameters:
   *       - in: path
   *         name: userId
   *         required: true
   *         schema:
   *           type: string
   *         description: User ID
   *       - in: path
   *         name: quizId
   *         required: true
   *         schema:
   *           type: string
   *         description: Quiz ID
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             required:
   *               - answers
   *             properties:
   *               answers:
   *                 type: array
   *                 items:
   *                   type: number
   *                 description: Array of answer indices
   *     responses:
   *       200:
   *         description: Quiz submitted successfully
   *         content:
   *           application/json:
   *             schema:
   *               type: object
   *               properties:
   *                 success:
   *                   type: boolean
   *                 data:
   *                   $ref: '#/components/schemas/QuizResult'
   *       400:
   *         description: Invalid request data
   *       500:
   *         description: Internal server error
   */
  async submitQuiz(req, res) {
    try {
      const { userId, quizId } = req.params;
      const { answers } = req.body;

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      if (!validateObjectId(quizId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid quiz ID'
        });
      }

      if (!answers || !Array.isArray(answers)) {
        return res.status(400).json({
          success: false,
          message: 'Answers array is required'
        });
      }

      const result = await learningModule.submitQuiz(userId, quizId, answers);

      if (!result.success) {
        return res.status(404).json(result);
      }

      res.json(result);
    } catch (error) {
      logger.error('Error in submitQuiz', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to submit quiz',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/learning/nudge/{userId}:
   *   get:
   *     summary: Get daily learning nudge
   *     description: Get personalized daily learning reminder
   *     tags: [Learning]
   *     parameters:
   *       - in: path
   *         name: userId
   *         required: true
   *         schema:
   *           type: string
   *         description: User ID
   *     responses:
   *       200:
   *         description: Daily nudge retrieved successfully
   *         content:
   *           application/json:
   *             schema:
   *               type: object
   *               properties:
   *                 success:
   *                   type: boolean
   *                 data:
   *                   $ref: '#/components/schemas/DailyNudge'
   *       400:
   *         description: Invalid user ID
   *       404:
   *         description: User not found
   *       500:
   *         description: Internal server error
   */
  async getDailyNudge(req, res) {
    try {
      const { userId } = req.params;

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      const result = await learningModule.getDailyNudge(userId);

      if (!result.success) {
        return res.status(404).json(result);
      }

      res.json(result);
    } catch (error) {
      logger.error('Error in getDailyNudge', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get daily nudge',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/learning/progress/{userId}:
   *   post:
   *     summary: Track learning progress
   *     description: Track user's learning progress and achievements
   *     tags: [Learning]
   *     parameters:
   *       - in: path
   *         name: userId
   *         required: true
   *         schema:
   *           type: string
   *         description: User ID
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             required:
   *               - action
   *             properties:
   *               action:
   *                 type: string
   *                 enum: [lesson_completed, quiz_completed, topic_completed, daily_login]
   *               details:
   *                 type: object
   *     responses:
   *       200:
   *         description: Progress tracked successfully
   *         content:
   *           application/json:
   *             schema:
   *               type: object
   *               properties:
   *                 success:
   *                   type: boolean
   *                 data:
   *                   type: object
   *       400:
   *         description: Invalid request data
   *       500:
   *         description: Internal server error
   */
  async trackLearningProgress(req, res) {
    try {
      const { userId } = req.params;
      const { action, details = {} } = req.body;

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      if (!action) {
        return res.status(400).json({
          success: false,
          message: 'Action is required'
        });
      }

      const result = await learningModule.trackLearningProgress(userId, action, details);

      res.json(result);
    } catch (error) {
      logger.error('Error in trackLearningProgress', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to track learning progress',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/learning/analytics/{userId}:
   *   get:
   *     summary: Get learning analytics
   *     description: Get comprehensive learning analytics and insights
   *     tags: [Learning]
   *     parameters:
   *       - in: path
   *         name: userId
   *         required: true
   *         schema:
   *           type: string
   *         description: User ID
   *     responses:
   *       200:
   *         description: Learning analytics retrieved successfully
   *         content:
   *           application/json:
   *             schema:
   *               type: object
   *               properties:
   *                 success:
   *                   type: boolean
   *                 data:
   *                   $ref: '#/components/schemas/LearningAnalytics'
   *       400:
   *         description: Invalid user ID
   *       404:
   *         description: User not found
   *       500:
   *         description: Internal server error
   */
  async getLearningAnalytics(req, res) {
    try {
      const { userId } = req.params;

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      const result = await learningModule.getLearningAnalytics(userId);

      if (!result.success) {
        return res.status(404).json(result);
      }

      res.json(result);
    } catch (error) {
      logger.error('Error in getLearningAnalytics', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get learning analytics',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/learning/topics:
   *   get:
   *     summary: Get learning topics
   *     description: Get all available learning topics
   *     tags: [Learning]
   *     responses:
   *       200:
   *         description: Learning topics retrieved successfully
   *         content:
   *           application/json:
   *             schema:
   *               type: object
   *               properties:
   *                 success:
   *                   type: boolean
   *                 data:
   *                   type: object
   *       500:
   *         description: Internal server error
   */
  async getLearningTopics(req, res) {
    try {
      const topics = learningModule.learningTopics;

      res.json({
        success: true,
        data: topics
      });
    } catch (error) {
      logger.error('Error in getLearningTopics', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get learning topics',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/learning/levels:
   *   get:
   *     summary: Get learning levels
   *     description: Get all available learning levels
   *     tags: [Learning]
   *     responses:
   *       200:
   *         description: Learning levels retrieved successfully
   *         content:
   *           application/json:
   *             schema:
   *               type: object
   *               properties:
   *                 success:
   *                   type: boolean
   *                 data:
   *                   type: object
   *       500:
   *         description: Internal server error
   */
  async getLearningLevels(req, res) {
    try {
      const levels = learningModule.learningLevels;

      res.json({
        success: true,
        data: levels
      });
    } catch (error) {
      logger.error('Error in getLearningLevels', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get learning levels',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/learning/profile/{userId}:
   *   get:
   *     summary: Get user learning profile
   *     description: Get user's learning profile and progress
   *     tags: [Learning]
   *     parameters:
   *       - in: path
   *         name: userId
   *         required: true
   *         schema:
   *           type: string
   *         description: User ID
   *     responses:
   *       200:
   *         description: User learning profile retrieved successfully
   *         content:
   *           application/json:
   *             schema:
   *               type: object
   *               properties:
   *                 success:
   *                   type: boolean
   *                 data:
   *                   type: object
   *       400:
   *         description: Invalid user ID
   *       404:
   *         description: User not found
   *       500:
   *         description: Internal server error
   */
  async getUserLearningProfile(req, res) {
    try {
      const { userId } = req.params;

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      // TODO: Implement user learning profile retrieval
      const profile = {
        currentLevel: 'BEGINNER',
        progress: {
          completedLessons: 0,
          totalLessons: 12,
          completedTopics: 0,
          totalTopics: 4,
          currentStreak: 0,
          longestStreak: 0,
          totalPoints: 0
        },
        achievements: [],
        currentTopic: 'BASICS',
        currentLesson: 0
      };

      res.json({
        success: true,
        data: profile
      });
    } catch (error) {
      logger.error('Error in getUserLearningProfile', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get user learning profile',
        error: error.message
      });
    }
  }
}

module.exports = new LearningController(); 