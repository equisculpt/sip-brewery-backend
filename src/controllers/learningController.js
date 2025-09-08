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
}

module.exports = new LearningController();
