const express = require('express');
const router = express.Router();
const SIPCalculatorService = require('../services/SIPCalculatorService');
const { body, query, validationResult } = require('express-validator');
const response = require('../utils/response');
const logger = require('../utils/logger');

const sipCalculatorService = new SIPCalculatorService();

// Validation middleware
const validateRequest = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return response.error(res, 'Validation failed', errors.array());
  }
  next();
};

/**
 * @route POST /api/sip-calculator/regular
 * @desc Calculate regular SIP returns
 * @access Public
 */
router.post('/regular', [
  body('monthlyInvestment').isFloat({ min: 500, max: 1000000 }).withMessage('Monthly investment must be between ₹500 and ₹10,00,000'),
  body('expectedReturn').isFloat({ min: 1, max: 30 }).withMessage('Expected return must be between 1% and 30%'),
  body('timePeriod').isInt({ min: 1, max: 50 }).withMessage('Time period must be between 1 and 50 years'),
  validateRequest
], async (req, res) => {
  try {
    const { monthlyInvestment, expectedReturn, timePeriod } = req.body;
    
    logger.info(`Calculating regular SIP: ₹${monthlyInvestment}/month, ${expectedReturn}% return, ${timePeriod} years`);
    
    const calculation = sipCalculatorService.calculateRegularSIP(
      parseFloat(monthlyInvestment),
      parseFloat(expectedReturn),
      parseInt(timePeriod)
    );
    
    return response.success(res, 'Regular SIP calculation completed successfully', calculation);
    
  } catch (error) {
    logger.error('Error in regular SIP calculation:', error);
    return response.error(res, 'Failed to calculate regular SIP', error.message);
  }
});

/**
 * @route POST /api/sip-calculator/stepup
 * @desc Calculate step-up SIP returns
 * @access Public
 */
router.post('/stepup', [
  body('monthlyInvestment').isFloat({ min: 500, max: 1000000 }).withMessage('Monthly investment must be between ₹500 and ₹10,00,000'),
  body('expectedReturn').isFloat({ min: 1, max: 30 }).withMessage('Expected return must be between 1% and 30%'),
  body('timePeriod').isInt({ min: 1, max: 50 }).withMessage('Time period must be between 1 and 50 years'),
  body('stepUpPercentage').isFloat({ min: 1, max: 50 }).withMessage('Step-up percentage must be between 1% and 50%'),
  validateRequest
], async (req, res) => {
  try {
    const { monthlyInvestment, expectedReturn, timePeriod, stepUpPercentage } = req.body;
    
    logger.info(`Calculating step-up SIP: ₹${monthlyInvestment}/month, ${expectedReturn}% return, ${timePeriod} years, ${stepUpPercentage}% step-up`);
    
    const calculation = sipCalculatorService.calculateStepUpSIP(
      parseFloat(monthlyInvestment),
      parseFloat(expectedReturn),
      parseInt(timePeriod),
      parseFloat(stepUpPercentage)
    );
    
    return response.success(res, 'Step-up SIP calculation completed successfully', calculation);
    
  } catch (error) {
    logger.error('Error in step-up SIP calculation:', error);
    return response.error(res, 'Failed to calculate step-up SIP', error.message);
  }
});

/**
 * @route POST /api/sip-calculator/dynamic
 * @desc Calculate dynamic SIP returns with AI adjustments
 * @access Public
 */
router.post('/dynamic', [
  body('monthlyInvestment').isFloat({ min: 500, max: 1000000 }).withMessage('Monthly investment must be between ₹500 and ₹10,00,000'),
  body('expectedReturn').isFloat({ min: 1, max: 30 }).withMessage('Expected return must be between 1% and 30%'),
  body('timePeriod').isInt({ min: 1, max: 50 }).withMessage('Time period must be between 1 and 50 years'),
  body('dynamicAdjustmentRange').optional().isFloat({ min: 5, max: 50 }).withMessage('Dynamic adjustment range must be between 5% and 50%'),
  validateRequest
], async (req, res) => {
  try {
    const { monthlyInvestment, expectedReturn, timePeriod, dynamicAdjustmentRange = 15 } = req.body;
    
    logger.info(`Calculating dynamic SIP: ₹${monthlyInvestment}/month, ${expectedReturn}% return, ${timePeriod} years, ±${dynamicAdjustmentRange}% AI adjustment`);
    
    const calculation = sipCalculatorService.calculateDynamicSIP(
      parseFloat(monthlyInvestment),
      parseFloat(expectedReturn),
      parseInt(timePeriod),
      parseFloat(dynamicAdjustmentRange)
    );
    
    return response.success(res, 'Dynamic SIP calculation completed successfully', calculation);
    
  } catch (error) {
    logger.error('Error in dynamic SIP calculation:', error);
    return response.error(res, 'Failed to calculate dynamic SIP', error.message);
  }
});

/**
 * @route POST /api/sip-calculator/compare
 * @desc Compare all SIP calculation types
 * @access Public
 */
router.post('/compare', [
  body('monthlyInvestment').isFloat({ min: 500, max: 1000000 }).withMessage('Monthly investment must be between ₹500 and ₹10,00,000'),
  body('expectedReturn').isFloat({ min: 1, max: 30 }).withMessage('Expected return must be between 1% and 30%'),
  body('timePeriod').isInt({ min: 1, max: 50 }).withMessage('Time period must be between 1 and 50 years'),
  body('stepUpPercentage').optional().isFloat({ min: 1, max: 50 }).withMessage('Step-up percentage must be between 1% and 50%'),
  body('dynamicAdjustmentRange').optional().isFloat({ min: 5, max: 50 }).withMessage('Dynamic adjustment range must be between 5% and 50%'),
  validateRequest
], async (req, res) => {
  try {
    const params = {
      monthlyInvestment: parseFloat(req.body.monthlyInvestment),
      expectedReturn: parseFloat(req.body.expectedReturn),
      timePeriod: parseInt(req.body.timePeriod),
      stepUpPercentage: parseFloat(req.body.stepUpPercentage || 10),
      dynamicAdjustmentRange: parseFloat(req.body.dynamicAdjustmentRange || 15)
    };
    
    logger.info(`Comparing SIP types: ₹${params.monthlyInvestment}/month, ${params.expectedReturn}% return, ${params.timePeriod} years`);
    
    const comparison = sipCalculatorService.getSIPComparison(params);
    
    return response.success(res, 'SIP comparison completed successfully', comparison);
    
  } catch (error) {
    logger.error('Error in SIP comparison:', error);
    return response.error(res, 'Failed to compare SIP calculations', error.message);
  }
});

/**
 * @route POST /api/sip-calculator/goal-based
 * @desc Calculate required SIP for a target goal
 * @access Public
 */
router.post('/goal-based', [
  body('targetAmount').isFloat({ min: 100000, max: 100000000 }).withMessage('Target amount must be between ₹1,00,000 and ₹10,00,00,000'),
  body('timePeriod').isInt({ min: 1, max: 50 }).withMessage('Time period must be between 1 and 50 years'),
  body('expectedReturn').isFloat({ min: 1, max: 30 }).withMessage('Expected return must be between 1% and 30%'),
  validateRequest
], async (req, res) => {
  try {
    const { targetAmount, timePeriod, expectedReturn } = req.body;
    
    logger.info(`Calculating goal-based SIP: Target ₹${targetAmount}, ${timePeriod} years, ${expectedReturn}% return`);
    
    const calculation = sipCalculatorService.calculateGoalBasedSIP(
      parseFloat(targetAmount),
      parseInt(timePeriod),
      parseFloat(expectedReturn)
    );
    
    return response.success(res, 'Goal-based SIP calculation completed successfully', calculation);
    
  } catch (error) {
    logger.error('Error in goal-based SIP calculation:', error);
    return response.error(res, 'Failed to calculate goal-based SIP', error.message);
  }
});

/**
 * @route GET /api/sip-calculator/quick-calculate
 * @desc Quick SIP calculation with query parameters
 * @access Public
 */
router.get('/quick-calculate', [
  query('monthlyInvestment').isFloat({ min: 500, max: 1000000 }).withMessage('Monthly investment must be between ₹500 and ₹10,00,000'),
  query('expectedReturn').isFloat({ min: 1, max: 30 }).withMessage('Expected return must be between 1% and 30%'),
  query('timePeriod').isInt({ min: 1, max: 50 }).withMessage('Time period must be between 1 and 50 years'),
  query('type').optional().isIn(['regular', 'stepup', 'dynamic']).withMessage('Type must be regular, stepup, or dynamic'),
  validateRequest
], async (req, res) => {
  try {
    const { monthlyInvestment, expectedReturn, timePeriod, type = 'regular' } = req.query;
    
    let calculation;
    
    switch (type) {
      case 'stepup':
        calculation = sipCalculatorService.calculateStepUpSIP(
          parseFloat(monthlyInvestment),
          parseFloat(expectedReturn),
          parseInt(timePeriod),
          10 // Default 10% step-up
        );
        break;
      case 'dynamic':
        calculation = sipCalculatorService.calculateDynamicSIP(
          parseFloat(monthlyInvestment),
          parseFloat(expectedReturn),
          parseInt(timePeriod),
          15 // Default 15% adjustment range
        );
        break;
      default:
        calculation = sipCalculatorService.calculateRegularSIP(
          parseFloat(monthlyInvestment),
          parseFloat(expectedReturn),
          parseInt(timePeriod)
        );
    }
    
    return response.success(res, `Quick ${type} SIP calculation completed successfully`, calculation);
    
  } catch (error) {
    logger.error('Error in quick SIP calculation:', error);
    return response.error(res, 'Failed to perform quick SIP calculation', error.message);
  }
});

/**
 * @route GET /api/sip-calculator/health
 * @desc Health check for SIP calculator service
 * @access Public
 */
router.get('/health', (req, res) => {
  try {
    return response.success(res, 'SIP Calculator Service is healthy', {
      service: 'SIP Calculator',
      status: 'operational',
      version: '1.0.0',
      features: [
        'Regular SIP Calculation',
        'Step-up SIP Calculation',
        'Dynamic SIP with AI',
        'Goal-based SIP Planning',
        'SIP Comparison Analysis'
      ],
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    return response.error(res, 'SIP Calculator Service health check failed', error.message);
  }
});

module.exports = router;
