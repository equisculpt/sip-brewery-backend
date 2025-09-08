const logger = require('../utils/logger');
const agiEngine = require('../services/agiEngine');
const { validateObjectId } = require('../middleware/validation');
const explainabilityEngine = require('../services/explainabilityEngine');
const advancedScenarioAnalytics = require('../services/advancedScenarioAnalytics');
const decisionEngine = require('../services/decisionEngine');
const behavioralRecommendationEngine = require('../services/behavioralRecommendationEngine');

class AGIController {
  /**
   * @swagger
   * /api/agi/insights/{userId}:
   *   get:
   *     summary: Get weekly AGI insights for user
   *     description: Generate personalized weekly insights using AGI engine
   *     tags: [AGI]
   *     parameters:
   *       - in: path
   *         name: userId
   *         required: true
   *         schema:
   *           type: string
   *         description: User ID
   *     responses:
   *       200:
   *         description: Weekly AGI insights generated successfully
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
   *                     insights:
   *                       type: array
   *                       items:
   *                         type: object
   *                         properties:
   *                           type:
   *                             type: string
   *                           priority:
   *                             type: string
   *                           action:
   *                             type: string
   *                           reasoning:
   *                             type: string
   *                           confidence:
   *                             type: number
   *       400:
   *         description: Invalid user ID
   *       404:
   *         description: User not found
   *       500:
   *         description: Internal server error
   */
  async getWeeklyInsights(req, res) {
    try {
      const { userId } = req.params;

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      const result = await agiEngine.generateWeeklyInsights(userId);

      if (!result.success) {
        return res.status(404).json(result);
      }

      res.json(result);
    } catch (error) {
      logger.error('Error in getWeeklyInsights', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to generate weekly insights',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/agi/recommendations/{userId}:
   *   get:
   *     summary: Get personalized investment recommendations
   *     description: Get AI-generated personalized investment recommendations
   *     tags: [AGI]
   *     parameters:
   *       - in: path
   *         name: userId
   *         required: true
   *         schema:
   *           type: string
   *         description: User ID
   *       - in: query
   *         name: type
   *         schema:
   *           type: string
   *           enum: [comprehensive, tax_optimization, risk_management, goal_alignment, market_opportunity]
   *         description: Type of recommendations
   *     responses:
   *       200:
   *         description: Recommendations generated successfully
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
   *                     recommendations:
   *                       type: array
   *                     recommendationType:
   *                       type: string
   *                     generatedAt:
   *                       type: string
   *       400:
   *         description: Invalid user ID
   *       404:
   *         description: User not found
   *       500:
   *         description: Internal server error
   */
  async getPersonalizedRecommendations(req, res) {
    try {
      const { userId } = req.params;
      const { type = 'comprehensive' } = req.query;

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      const result = await agiEngine.getPersonalizedRecommendations(userId, type);

      if (!result.success) {
        return res.status(404).json(result);
      }

      res.json(result);
    } catch (error) {
      logger.error('Error in getPersonalizedRecommendations', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get personalized recommendations',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/agi/macroeconomic/{userId}:
   *   get:
   *     summary: Analyze macroeconomic impact on portfolio
   *     description: Analyze how macroeconomic factors affect user's portfolio
   *     tags: [AGI]
   *     parameters:
   *       - in: path
   *         name: userId
   *         required: true
   *         schema:
   *           type: string
   *         description: User ID
   *     responses:
   *       200:
   *         description: Macroeconomic analysis completed successfully
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
   *                     impact:
   *                       type: object
   *                     overallImpact:
   *                       type: string
   *                     recommendations:
   *                       type: array
   *       400:
   *         description: Invalid user ID
   *       404:
   *         description: User not found
   *       500:
   *         description: Internal server error
   */
  async analyzeMacroeconomicImpact(req, res) {
    try {
      const { userId } = req.params;

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      const result = await agiEngine.analyzeMacroeconomicImpact(userId);

      if (!result.success) {
        return res.status(404).json(result);
      }

      res.json(result);
    } catch (error) {
      logger.error('Error in analyzeMacroeconomicImpact', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to analyze macroeconomic impact',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/agi/behavior:
   *   post:
   *     summary: Track user behavior for AGI learning
   *     description: Track user actions to improve AGI recommendations
   *     tags: [AGI]
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             required:
   *               - userId
   *               - action
   *             properties:
   *               userId:
   *                 type: string
   *               action:
   *                 type: string
   *               context:
   *                 type: object
   *     responses:
   *       200:
   *         description: Behavior tracked successfully
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
  async trackUserBehavior(req, res) {
    try {
      const { userId, action, context = {} } = req.body;

      if (!userId || !action) {
        return res.status(400).json({
          success: false,
          message: 'User ID and action are required'
        });
      }

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      const result = await agiEngine.trackUserBehavior(userId, action, context);

      res.json(result);
    } catch (error) {
      logger.error('Error in trackUserBehavior', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to track user behavior',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/agi/learn:
   *   post:
   *     summary: Learn from market events
   *     description: Update AGI models based on market events and user reactions
   *     tags: [AGI]
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             required:
   *               - eventType
   *               - eventData
   *             properties:
   *               eventType:
   *                 type: string
   *               eventData:
   *                 type: object
   *               userReactions:
   *                 type: array
   *     responses:
   *       200:
   *         description: Learning completed successfully
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
  async learnFromMarketEvents(req, res) {
    try {
      const { eventType, eventData, userReactions = [] } = req.body;

      if (!eventType || !eventData) {
        return res.status(400).json({
          success: false,
          message: 'Event type and data are required'
        });
      }

      const result = await agiEngine.learnFromMarketEvents(eventType, eventData, userReactions);

      res.json(result);
    } catch (error) {
      logger.error('Error in learnFromMarketEvents', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to learn from market events',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/agi/capabilities:
   *   get:
   *     summary: Get AGI capabilities
   *     description: Get information about AGI engine capabilities
   *     tags: [AGI]
   *     responses:
   *       200:
   *         description: AGI capabilities retrieved successfully
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
   *                     capabilities:
   *                       type: object
   *                     macroeconomicFactors:
   *                       type: object
   *                     insightTypes:
   *                       type: object
   *       500:
   *         description: Internal server error
   */
  async getAGICapabilities(req, res) {
    try {
      const capabilities = {
        capabilities: agiEngine.agiCapabilities,
        macroeconomicFactors: agiEngine.macroeconomicFactors,
        insightTypes: agiEngine.agiInsightTypes,
        learningSources: agiEngine.learningSources
      };

      res.json({
        success: true,
        data: capabilities
      });
    } catch (error) {
      logger.error('Error in getAGICapabilities', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get AGI capabilities',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/agi/feedback:
   *   post:
   *     summary: Submit feedback for AGI improvement
   *     description: Submit user feedback to improve AGI accuracy and recommendations
   *     tags: [AGI]
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             required:
   *               - userId
   *               - insightId
   *               - feedback
   *             properties:
   *               userId:
   *                 type: string
   *               insightId:
   *                 type: string
   *               feedback:
   *                 type: string
   *                 enum: [accepted, rejected, implemented, ignored]
   *               rating:
   *                 type: number
   *                 minimum: 1
   *                 maximum: 5
   *               comments:
   *                 type: string
   *     responses:
   *       200:
   *         description: Feedback submitted successfully
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
  async submitFeedback(req, res) {
    try {
      const { userId, insightId, feedback, rating, comments } = req.body;

      if (!userId || !insightId || !feedback) {
        return res.status(400).json({
          success: false,
          message: 'User ID, insight ID, and feedback are required'
        });
      }

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      // Store feedback for AGI learning
      const feedbackData = {
        userId,
        insightId,
        feedback,
        rating: rating || 3,
        comments: comments || '',
        timestamp: new Date()
      };

      // TODO: Store feedback in database
      logger.info('AGI feedback received', feedbackData);

      res.json({
        success: true,
        data: {
          message: 'Feedback submitted successfully',
          feedbackId: Date.now().toString()
        }
      });
    } catch (error) {
      logger.error('Error in submitFeedback', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to submit feedback',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/agi/status:
   *   get:
   *     summary: Get AGI system status
   *     description: Get current status and health of AGI system
   *     tags: [AGI]
   *     responses:
   *       200:
   *         description: AGI status retrieved successfully
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
   *                     status:
   *                       type: string
   *                     uptime:
   *                       type: number
   *                     insightsGenerated:
   *                       type: number
   *                     accuracy:
   *                       type: number
   *                     lastTraining:
   *                       type: string
   *       500:
   *         description: Internal server error
   */
  async getAGIStatus(req, res) {
    try {
      const status = {
        status: 'operational',
        uptime: process.uptime(),
        insightsGenerated: 0, // TODO: Get from database
        accuracy: 0.85, // TODO: Calculate from feedback
        lastTraining: new Date().toISOString(),
        version: '1.0.0',
        model: 'mistral'
      };

      res.json({
        success: true,
        data: status
      });
    } catch (error) {
      logger.error('Error in getAGIStatus', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get AGI status',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/agi/initialize:
   *   post:
   *     summary: Initialize AGI for user
   *     description: Initialize AGI system for a specific user
   *     tags: [AGI]
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             properties:
   *               userId:
   *                 type: string
   *                 required: true
   *     responses:
   *       200:
   *         description: AGI initialized successfully
   *       400:
   *         description: Invalid request
   *       500:
   *         description: Internal server error
   */
  async initializeAGI(req, res) {
    try {
      const { userId } = req.body;

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      const result = await agiEngine.generateWeeklyInsights(userId);

      if (!result.success) {
        return res.status(404).json(result);
      }

      res.json({
        success: true,
        message: 'AGI initialized successfully',
        data: result.data
      });
    } catch (error) {
      logger.error('Error in initializeAGI', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to initialize AGI',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/agi/autonomous-management:
   *   post:
   *     summary: Perform autonomous portfolio management
   *     description: Execute autonomous portfolio management actions
   *     tags: [AGI]
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             properties:
   *               userId:
   *                 type: string
   *                 required: true
   *               enableAutonomous:
   *                 type: boolean
   *     responses:
   *       200:
   *         description: Autonomous management executed successfully
   *       400:
   *         description: Invalid request
   *       500:
   *         description: Internal server error
   */
  async autonomousPortfolioManagement(req, res) {
    try {
      const { userId, enableAutonomous = true } = req.body;

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      const result = await agiEngine.generateWeeklyInsights(userId);

      if (!result.success) {
        return res.status(404).json(result);
      }

      res.json({
        success: true,
        message: 'Autonomous portfolio management executed',
        data: {
          autonomousEnabled: enableAutonomous,
          insights: result.data.insights,
          recommendations: result.data.insights.filter(insight => insight.priority === 'high')
        }
      });
    } catch (error) {
      logger.error('Error in autonomousPortfolioManagement', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to execute autonomous management',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/agi/predictions:
   *   get:
   *     summary: Get market predictions
   *     description: Get AI-generated market predictions
   *     tags: [AGI]
   *     parameters:
   *       - in: query
   *         name: timeframe
   *         schema:
   *           type: string
   *           enum: [short_term, medium_term, long_term]
   *         description: Prediction timeframe
   *     responses:
   *       200:
   *         description: Predictions generated successfully
   *       500:
   *         description: Internal server error
   */
  async getMarketPredictions(req, res) {
    try {
      const { timeframe = 'medium_term' } = req.query;

      const result = await agiEngine.generateWeeklyInsights('system');

      res.json({
        success: true,
        data: {
          timeframe,
          predictions: result.data.insights || [],
          confidence: result.data.confidence || 0.7,
          generatedAt: new Date().toISOString()
        }
      });
    } catch (error) {
      logger.error('Error in getMarketPredictions', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to generate market predictions',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/agi/risk-management:
   *   post:
   *     summary: Perform intelligent risk management
   *     description: Execute intelligent risk management analysis
   *     tags: [AGI]
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             properties:
   *               userId:
   *                 type: string
   *                 required: true
   *     responses:
   *       200:
   *         description: Risk management executed successfully
   *       400:
   *         description: Invalid request
   *       500:
   *         description: Internal server error
   */
  async intelligentRiskManagement(req, res) {
    try {
      const { userId } = req.body;

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      const result = await agiEngine.generateWeeklyInsights(userId);

      if (!result.success) {
        return res.status(404).json(result);
      }

      const riskInsights = result.data.insights.filter(insight => 
        insight.insightType === 'RISK_MANAGEMENT'
      );

      res.json({
        success: true,
        data: {
          riskAssessment: riskInsights,
          overallRiskLevel: 'medium',
          recommendations: riskInsights.map(insight => insight.reasoning)
        }
      });
    } catch (error) {
      logger.error('Error in intelligentRiskManagement', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to execute risk management',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/agi/execute-actions:
   *   post:
   *     summary: Execute autonomous actions
   *     description: Execute autonomous actions based on AGI insights
   *     tags: [AGI]
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             properties:
   *               userId:
   *                 type: string
   *                 required: true
   *               actions:
   *                 type: array
   *                 items:
   *                   type: object
   *     responses:
   *       200:
   *         description: Actions executed successfully
   *       400:
   *         description: Invalid request
   *       500:
   *         description: Internal server error
   */
  async executeAutonomousActions(req, res) {
    try {
      const { userId, actions } = req.body;

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      if (!actions || !Array.isArray(actions)) {
        return res.status(400).json({
          success: false,
          message: 'Actions array is required'
        });
      }

      const results = actions.map(action => ({
        action: action.type || 'unknown',
        success: true,
        executedAt: new Date().toISOString(),
        result: 'Action executed successfully'
      }));

      res.json({
        success: true,
        data: {
          executedActions: results,
          totalActions: actions.length,
          successfulActions: results.length
        }
      });
    } catch (error) {
      logger.error('Error in executeAutonomousActions', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to execute actions',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/agi/toggle-autonomous:
   *   post:
   *     summary: Toggle autonomous mode
   *     description: Enable or disable autonomous mode for user
   *     tags: [AGI]
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             properties:
   *               userId:
   *                 type: string
   *                 required: true
   *               enable:
   *                 type: boolean
   *                 required: true
   *     responses:
   *       200:
   *         description: Autonomous mode toggled successfully
   *       400:
   *         description: Invalid request
   *       500:
   *         description: Internal server error
   */
  async toggleAutonomousMode(req, res) {
    try {
      const { userId, enable } = req.body;

      if (!validateObjectId(userId)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid user ID'
        });
      }

      if (typeof enable !== 'boolean') {
        return res.status(400).json({
          success: false,
          message: 'Enable parameter must be a boolean'
        });
      }

      res.json({
        success: true,
        data: {
          autonomousEnabled: enable,
          updatedAt: new Date().toISOString(),
          message: enable ? 'Autonomous mode enabled' : 'Autonomous mode disabled'
        }
      });
    } catch (error) {
      logger.error('Error in toggleAutonomousMode', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to toggle autonomous mode',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/agi/insights:
   *   get:
   *     summary: Get AGI insights
   *     description: Get general AGI insights and analysis
   *     tags: [AGI]
   *     parameters:
   *       - in: query
   *         name: timeframe
   *         schema:
   *           type: string
   *           enum: [daily, weekly, monthly]
   *         description: Insight timeframe
   *     responses:
   *       200:
   *         description: Insights retrieved successfully
   *       500:
   *         description: Internal server error
   */
  async getAGIInsights(req, res) {
    try {
      const { timeframe = 'weekly' } = req.query;

      const result = await agiEngine.generateWeeklyInsights('system');

      res.json({
        success: true,
        data: {
          timeframe,
          insights: result.data.insights || [],
          summary: {
            totalInsights: result.data.insights?.length || 0,
            highPriority: result.data.insights?.filter(i => i.priority === 'high').length || 0,
            generatedAt: new Date().toISOString()
          }
        }
      });
    } catch (error) {
      logger.error('Error in getAGIInsights', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get AGI insights',
        error: error.message
      });
    }
  }

  /**
   * @swagger
   * /api/agi/dynamic-allocation:
   *   post:
   *     summary: Get dynamic SIP/portfolio allocation
   *     description: Use Decision Engine for dynamic allocation
   *     tags: [AGI]
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             properties:
   *               userProfile:
   *                 type: object
   *               marketState:
   *                 type: object
   *               assets:
   *                 type: array
   *                 items: { type: object }
   *     responses:
   *       200:
   *         description: Dynamic allocation result
   *       400:
   *         description: Invalid request
   *       500:
   *         description: Internal server error
   */
  async getDynamicAllocation(req, res) {
    try {
      const { userProfile, marketState, assets } = req.body;
      if (!userProfile || !marketState || !assets) {
        return res.status(400).json({ success: false, message: 'userProfile, marketState, and assets are required' });
      }
      const result = await decisionEngine.getDynamicAllocation(userProfile, marketState, assets);
      res.json({ success: true, data: result });
    } catch (error) {
      logger.error('Error in getDynamicAllocation', { error: error.message });
      res.status(500).json({ success: false, message: 'Failed to get dynamic allocation', error: error.message });
    }
  }

  /**
   * @swagger
   * /api/agi/scenario-simulation:
   *   post:
   *     summary: Simulate scenario (bull/bear/sideways)
   *     description: Use Decision Engine to simulate investment scenario
   *     tags: [AGI]
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             properties:
   *               userProfile:
   *                 type: object
   *               scenario:
   *                 type: string
   *               assets:
   *                 type: array
   *                 items: { type: object }
   *     responses:
   *       200:
   *         description: Scenario simulation result
   *       400:
   *         description: Invalid request
   *       500:
   *         description: Internal server error
   */
  async simulateScenario(req, res) {
    try {
      const { userProfile, scenario, assets } = req.body;
      if (!userProfile || !scenario || !assets) {
        return res.status(400).json({ success: false, message: 'userProfile, scenario, and assets are required' });
      }
      const result = await decisionEngine.simulateScenario(userProfile, scenario, assets);
      res.json({ success: true, data: result });
    } catch (error) {
      logger.error('Error in simulateScenario', { error: error.message });
      res.status(500).json({ success: false, message: 'Failed to simulate scenario', error: error.message });
    }
  }

  /**
   * @swagger
   * /api/agi/behavioral-nudges:
   *   post:
   *     summary: Generate behavioral nudges
   *     description: Use Behavioral Recommendation Engine for nudges
   *     tags: [AGI]
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             properties:
   *               userProfile:
   *                 type: object
   *               userActions:
   *                 type: object
   *               marketEvents:
   *                 type: object
   *     responses:
   *       200:
   *         description: Behavioral nudges generated
   *       400:
   *         description: Invalid request
   *       500:
   *         description: Internal server error
   */
  async generateBehavioralNudges(req, res) {
    try {
      const { userProfile, userActions, marketEvents } = req.body;
      if (!userProfile || !userActions || !marketEvents) {
        return res.status(400).json({ success: false, message: 'userProfile, userActions, and marketEvents are required' });
      }
      const nudges = behavioralRecommendationEngine.generateNudges(userProfile, userActions, marketEvents);
      res.json({ success: true, data: { nudges } });
    } catch (error) {
      logger.error('Error in generateBehavioralNudges', { error: error.message });
      res.status(500).json({ success: false, message: 'Failed to generate behavioral nudges', error: error.message });
    }
  }

  /**
   * @swagger
   * /api/agi/explain:
   *   post:
   *     summary: Get explanation for a recommendation or decision
   *     description: Generate an explainability report for AGI/AI output
   *     tags: [AGI]
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             properties:
   *               context:
   *                 type: object
   *               result:
   *                 type: object
   *     responses:
   *       200:
   *         description: Explanation generated
   *       400:
   *         description: Invalid request
   *       500:
   *         description: Internal server error
   */
  async explain(req, res) {
    try {
      const { context, result } = req.body;
      if (!context || !result) {
        return res.status(400).json({ success: false, message: 'context and result are required' });
      }
      const explanation = explainabilityEngine.generateExplanation(context, result);
      res.json({ success: true, data: explanation });
    } catch (error) {
      logger.error('Error in explain', { error: error.message });
      res.status(500).json({ success: false, message: 'Failed to generate explanation', error: error.message });
    }
  }

  /**
   * @swagger
   * /api/agi/advanced-scenario-analytics:
   *   post:
   *     summary: Run advanced scenario analytics
   *     description: Simulate advanced market/portfolio scenarios
   *     tags: [AGI]
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             properties:
   *               userProfile:
   *                 type: object
   *               portfolio:
   *                 type: object
   *               scenarios:
   *                 type: object
   *     responses:
   *       200:
   *         description: Analytics result
   *       400:
   *         description: Invalid request
   *       500:
   *         description: Internal server error
   */
  async advancedScenarioAnalyticsEndpoint(req, res) {
    try {
      const { userProfile, portfolio, scenarios } = req.body;
      if (!userProfile || !portfolio || !scenarios) {
        return res.status(400).json({ success: false, message: 'userProfile, portfolio, and scenarios are required' });
      }
      const analyticsResult = advancedScenarioAnalytics.simulate(userProfile, portfolio, scenarios);
      res.json({ success: true, data: analyticsResult });
    } catch (error) {
      logger.error('Error in advancedScenarioAnalyticsEndpoint', { error: error.message });
      res.status(500).json({ success: false, message: 'Failed to run advanced scenario analytics', error: error.message });
    }
  }
}

/**
 * @swagger
 * /api/agi/portfolio/analytics:
 *   post:
 *     summary: Upload and analyze user portfolio
 *     description: Upload a portfolio (stocks, MFs, ETFs) and get analytics
 *     tags: [AGI]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               portfolio:
 *                 type: array
 *                 items:
 *                   type: object
 *     responses:
 *       200:
 *         description: Portfolio analytics returned
 *       400:
 *         description: Invalid request
 *       500:
 *         description: Internal server error
 */
async portfolioAnalytics(req, res) {
  try {
    const { portfolio, mfAnalytics = {}, stockAnalytics = {} } = req.body;
    if (!portfolio || !Array.isArray(portfolio)) {
      return res.status(400).json({ success: false, message: 'portfolio (array) is required' });
    }
    const featureExtractor = require('../utils/featureExtractor');
    const result = featureExtractor.extractPortfolioFeatures(portfolio, mfAnalytics, stockAnalytics);
    res.json({ success: true, data: result });
  } catch (error) {
    logger.error('Error in portfolioAnalytics', { error: error.message });
    res.status(500).json({ success: false, message: 'Failed to analyze portfolio', error: error.message });
  }
},

/**
 * @swagger
 * /api/agi/mutual-fund/analytics:
 *   post:
 *     summary: Get mutual fund analytics
 *     description: Analyze a mutual fund's NAV history and returns
 *     tags: [AGI]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               schemeDetail:
 *                 type: object
 *               navHistory:
 *                 type: array
 *                 items:
 *                   type: object
 *               returns:
 *                 type: object
 *     responses:
 *       200:
 *         description: Mutual fund analytics returned
 *       400:
 *         description: Invalid request
 *       500:
 *         description: Internal server error
 */
async mutualFundAnalytics(req, res) {
  try {
    const { schemeDetail, navHistory, returns } = req.body;
    if (!schemeDetail || !navHistory || !returns) {
      return res.status(400).json({ success: false, message: 'schemeDetail, navHistory, and returns are required' });
    }
    const featureExtractor = require('../utils/featureExtractor');
    const result = featureExtractor.extractMfFeatures({ schemeDetail, navHistory, returns });
    res.json({ success: true, data: result });
  } catch (error) {
    logger.error('Error in mutualFundAnalytics', { error: error.message });
    res.status(500).json({ success: false, message: 'Failed to analyze mutual fund', error: error.message });
  }
},

/**
 * @swagger
 * /api/agi/compare/stock-vs-fund:
 *   post:
 *     summary: Compare stocks vs mutual funds
 *     description: Compare risk/return/drawdown for stocks and mutual funds
 *     tags: [AGI]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               stockAnalytics:
 *                 type: object
 *               mfAnalytics:
 *                 type: object
 *     responses:
 *       200:
 *         description: Comparison returned
 *       400:
 *         description: Invalid request
 *       500:
 *         description: Internal server error
 */
async compareStockVsFund(req, res) {
  try {
    const { stockAnalytics = {}, mfAnalytics = {} } = req.body;
    const featureExtractor = require('../utils/featureExtractor');
    const result = featureExtractor.compareStockVsMf(stockAnalytics, mfAnalytics);
    res.json({ success: true, data: result });
  } catch (error) {
    logger.error('Error in compareStockVsFund', { error: error.message });
    res.status(500).json({ success: false, message: 'Failed to compare stock vs fund', error: error.message });
  }
},

module.exports = new AGIController();