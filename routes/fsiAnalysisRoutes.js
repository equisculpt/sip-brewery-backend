const express = require('express');
const router = express.Router();
const { authenticateToken } = require('../middleware/auth');
const { validateRequest } = require('../middleware/validation');
const { param, query } = require('express-validator');
const response = require('../utils/response');
const logger = require('../utils/logger');

// Import services
const RealMutualFundDataService = require('../services/RealMutualFundDataService');
const FSIAnalysisService = require('../services/FSIAnalysisService');
const MarketSentimentService = require('../services/MarketSentimentService');

const realMutualFundService = new RealMutualFundDataService();
const fsiAnalysisService = new FSIAnalysisService();
const marketSentimentService = new MarketSentimentService();

console.log('🚀 FSI Analysis Routes Initialized - Real-time AI Analysis');

/**
 * @route GET /api/fsi-analysis/:fundId/basic-info
 * @desc Get basic fund information
 * @access Private
 */
router.get('/:fundId/basic-info', [
  authenticateToken,
  param('fundId').notEmpty().withMessage('Fund ID is required'),
  validateRequest
], async (req, res) => {
  try {
    const { fundId } = req.params;
    
    logger.info(`🚀 Fetching basic info for fund: ${fundId}`);
    
    const fundInfo = await realMutualFundService.getFundBasicInfo(fundId);
    
    return response.success(res, 'Fund basic info retrieved successfully', fundInfo);
  } catch (error) {
    logger.error('Error fetching fund basic info:', error);
    return response.error(res, 'Failed to fetch fund basic info', error.message);
  }
});

/**
 * @route GET /api/fsi-analysis/:fundId/fsi-analysis
 * @desc Get comprehensive FSI analysis for a fund
 * @access Private
 */
router.get('/:fundId/fsi-analysis', [
  authenticateToken,
  param('fundId').notEmpty().withMessage('Fund ID is required'),
  validateRequest
], async (req, res) => {
  try {
    const { fundId } = req.params;
    
    logger.info(`🚀 Performing FSI analysis for fund: ${fundId}`);
    
    const fsiAnalysis = await fsiAnalysisService.performComprehensiveAnalysis(fundId);
    
    return response.success(res, 'FSI analysis completed successfully', fsiAnalysis);
  } catch (error) {
    logger.error('Error performing FSI analysis:', error);
    return response.error(res, 'Failed to perform FSI analysis', error.message);
  }
});

/**
 * @route GET /api/fsi-analysis/market/sentiment-analysis
 * @desc Get current market sentiment analysis
 * @access Private
 */
router.get('/market/sentiment-analysis', [
  authenticateToken,
  validateRequest
], async (req, res) => {
  try {
    logger.info('🚀 Fetching market sentiment analysis');
    
    const marketSentiment = await marketSentimentService.getCurrentMarketSentiment();
    
    return response.success(res, 'Market sentiment analysis retrieved successfully', marketSentiment);
  } catch (error) {
    logger.error('Error fetching market sentiment:', error);
    return response.error(res, 'Failed to fetch market sentiment', error.message);
  }
});

/**
 * @route GET /api/fsi-analysis/:fundId/holdings
 * @desc Get fund holdings with FSI analysis
 * @access Private
 */
router.get('/:fundId/holdings', [
  authenticateToken,
  param('fundId').notEmpty().withMessage('Fund ID is required'),
  query('limit').optional().isInt({ min: 1, max: 100 }).withMessage('Limit must be between 1 and 100'),
  validateRequest
], async (req, res) => {
  try {
    const { fundId } = req.params;
    const { limit = 50 } = req.query;
    
    logger.info(`🚀 Fetching holdings for fund: ${fundId}`);
    
    const holdings = await fsiAnalysisService.getFundHoldingsWithAnalysis(fundId, parseInt(limit));
    
    return response.success(res, 'Fund holdings retrieved successfully', holdings);
  } catch (error) {
    logger.error('Error fetching fund holdings:', error);
    return response.error(res, 'Failed to fetch fund holdings', error.message);
  }
});

/**
 * @route GET /api/fsi-analysis/:fundId/sector-allocation
 * @desc Get sector allocation with future outlook
 * @access Private
 */
router.get('/:fundId/sector-allocation', [
  authenticateToken,
  param('fundId').notEmpty().withMessage('Fund ID is required'),
  validateRequest
], async (req, res) => {
  try {
    const { fundId } = req.params;
    
    logger.info(`🚀 Fetching sector allocation for fund: ${fundId}`);
    
    const sectorAllocation = await fsiAnalysisService.getSectorAllocationWithOutlook(fundId);
    
    return response.success(res, 'Sector allocation retrieved successfully', sectorAllocation);
  } catch (error) {
    logger.error('Error fetching sector allocation:', error);
    return response.error(res, 'Failed to fetch sector allocation', error.message);
  }
});

/**
 * @route GET /api/fsi-analysis/:fundId/performance
 * @desc Get fund performance data
 * @access Private
 */
router.get('/:fundId/performance', [
  authenticateToken,
  param('fundId').notEmpty().withMessage('Fund ID is required'),
  validateRequest
], async (req, res) => {
  try {
    const { fundId } = req.params;
    
    logger.info(`🚀 Fetching performance data for fund: ${fundId}`);
    
    const performance = await fsiAnalysisService.getFundPerformanceData(fundId);
    
    return response.success(res, 'Performance data retrieved successfully', performance);
  } catch (error) {
    logger.error('Error fetching performance data:', error);
    return response.error(res, 'Failed to fetch performance data', error.message);
  }
});

/**
 * @route GET /api/fsi-analysis/:fundId/risk-metrics
 * @desc Get fund risk metrics
 * @access Private
 */
router.get('/:fundId/risk-metrics', [
  authenticateToken,
  param('fundId').notEmpty().withMessage('Fund ID is required'),
  validateRequest
], async (req, res) => {
  try {
    const { fundId } = req.params;
    
    logger.info(`🚀 Fetching risk metrics for fund: ${fundId}`);
    
    const riskMetrics = await fsiAnalysisService.getFundRiskMetrics(fundId);
    
    return response.success(res, 'Risk metrics retrieved successfully', riskMetrics);
  } catch (error) {
    logger.error('Error fetching risk metrics:', error);
    return response.error(res, 'Failed to fetch risk metrics', error.message);
  }
});

/**
 * @route GET /api/fsi-analysis/stocks/:stockId/analysis
 * @desc Get detailed stock analysis
 * @access Private
 */
router.get('/stocks/:stockId/analysis', [
  authenticateToken,
  param('stockId').notEmpty().withMessage('Stock ID is required'),
  validateRequest
], async (req, res) => {
  try {
    const { stockId } = req.params;
    
    logger.info(`🚀 Fetching detailed analysis for stock: ${stockId}`);
    
    const stockAnalysis = await fsiAnalysisService.getDetailedStockAnalysis(stockId);
    
    return response.success(res, 'Stock analysis retrieved successfully', stockAnalysis);
  } catch (error) {
    logger.error('Error fetching stock analysis:', error);
    return response.error(res, 'Failed to fetch stock analysis', error.message);
  }
});

module.exports = router;
