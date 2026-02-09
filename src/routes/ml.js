const express = require('express');
const router = express.Router();
const { authenticateToken } = require('../middleware/unifiedAuth');
const { validateRequest } = require('../middleware/validation');
const { body, query } = require('express-validator');
const mlInferenceService = require('../services/mlInferenceService');
const logger = require('../utils/logger');

const portfolioOptimizationSchema = [
  body('portfolioData').isObject().withMessage('Portfolio data is required'),
  body('portfolioData.holdings').isArray().withMessage('Holdings array is required'),
  body('userProfile').optional().isObject()
];

const fundPredictionSchema = [
  body('fundIds').isArray({ min: 1 }).withMessage('At least one fund ID is required'),
  body('timeHorizon').optional().isIn(['1M', '3M', '6M', '1Y', '3Y']).withMessage('Invalid time horizon')
];

const riskPredictionSchema = [
  body('portfolioData').isObject().withMessage('Portfolio data is required'),
  body('timeHorizon').optional().isIn(['1D', '1W', '1M']).withMessage('Invalid time horizon')
];

router.post(
  '/portfolio/optimize',
  authenticateToken,
  portfolioOptimizationSchema,
  validateRequest,
  async (req, res) => {
    try {
      const { portfolioData, userProfile } = req.body;
      const userId = req.userId;

      logger.info('Portfolio optimization requested', { userId });

      const result = await mlInferenceService.optimizePortfolio(
        userId,
        portfolioData,
        userProfile || {}
      );

      if (result.success) {
        res.json({
          success: true,
          data: result.data,
          message: 'Portfolio optimized successfully'
        });
      } else {
        res.status(500).json({
          success: false,
          message: 'Portfolio optimization failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Portfolio optimization endpoint error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }
);

router.post(
  '/funds/predict',
  authenticateToken,
  fundPredictionSchema,
  validateRequest,
  async (req, res) => {
    try {
      const { fundIds, timeHorizon = '1Y' } = req.body;

      logger.info('Fund performance prediction requested', { fundIds, timeHorizon });

      const result = await mlInferenceService.predictFundPerformance(fundIds, timeHorizon);

      if (result.success) {
        res.json({
          success: true,
          data: result.data,
          message: 'Fund performance predicted successfully'
        });
      } else {
        res.status(500).json({
          success: false,
          message: 'Fund prediction failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Fund prediction endpoint error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }
);

router.post(
  '/risk/predict',
  authenticateToken,
  riskPredictionSchema,
  validateRequest,
  async (req, res) => {
    try {
      const { portfolioData, timeHorizon = '1M' } = req.body;
      const userId = req.userId;

      logger.info('Risk prediction requested', { userId, timeHorizon });

      const result = await mlInferenceService.predictRisk(userId, portfolioData, timeHorizon);

      if (result.success) {
        res.json({
          success: true,
          data: result.data,
          message: 'Risk predicted successfully'
        });
      } else {
        res.status(500).json({
          success: false,
          message: 'Risk prediction failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Risk prediction endpoint error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }
);

router.post(
  '/behavior/predict',
  authenticateToken,
  async (req, res) => {
    try {
      const { userHistory, currentContext } = req.body;
      const userId = req.userId;

      logger.info('Behavioral prediction requested', { userId });

      const result = await mlInferenceService.predictUserBehavior(
        userId,
        userHistory || {},
        currentContext || {}
      );

      if (result.success) {
        res.json({
          success: true,
          data: result.data,
          message: 'Behavior predicted successfully'
        });
      } else {
        res.status(500).json({
          success: false,
          message: 'Behavioral prediction failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Behavioral prediction endpoint error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }
);

router.get(
  '/market/regime',
  authenticateToken,
  async (req, res) => {
    try {
      logger.info('Market regime detection requested');

      const result = await mlInferenceService.detectMarketRegime();

      if (result.success) {
        res.json({
          success: true,
          data: result.data,
          message: 'Market regime detected successfully'
        });
      } else {
        res.status(500).json({
          success: false,
          message: 'Market regime detection failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Market regime detection endpoint error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }
);

router.delete(
  '/cache/clear',
  authenticateToken,
  async (req, res) => {
    try {
      const { pattern = 'ml:*' } = req.query;

      if (req.user.role !== 'ADMIN') {
        return res.status(403).json({
          success: false,
          message: 'Admin access required'
        });
      }

      await mlInferenceService.clearCache(pattern);

      res.json({
        success: true,
        message: 'ML cache cleared successfully'
      });
    } catch (error) {
      logger.error('Cache clear endpoint error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to clear cache',
        error: error.message
      });
    }
  }
);

router.get(
  '/health',
  async (req, res) => {
    try {
      res.json({
        success: true,
        message: 'ML service is healthy',
        models: {
          portfolio_optimizer: 'ready',
          fund_predictor: 'ready',
          risk_predictor: 'ready',
          behavioral_predictor: 'ready',
          regime_detector: 'ready'
        }
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: 'ML service health check failed'
      });
    }
  }
);

module.exports = router;
