const express = require('express');
const router = express.Router();
const { authenticateToken } = require('../middleware/unifiedAuth');
const kafkaProducer = require('../services/kafkaProducerService');
const knowledgeGraph = require('../services/knowledgeGraphService');
const vectorDatabase = require('../services/vectorDatabaseService');
const realTimeFeatures = require('../services/realTimeFeatureService');
const logger = require('../utils/logger');

// Knowledge Graph endpoints

router.post('/graph/fund', authenticateToken, async (req, res) => {
  try {
    const { fundData } = req.body;
    
    const result = await knowledgeGraph.createFund(fundData);
    
    res.json({
      success: true,
      data: result,
      message: 'Fund added to knowledge graph'
    });
  } catch (error) {
    logger.error('Failed to create fund in graph', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

router.get('/graph/similar-funds/:fundId', authenticateToken, async (req, res) => {
  try {
    const { fundId } = req.params;
    const { limit = 10 } = req.query;
    
    const similarFunds = await knowledgeGraph.findSimilarFunds(fundId, parseInt(limit));
    
    res.json({
      success: true,
      data: similarFunds
    });
  } catch (error) {
    logger.error('Failed to find similar funds', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

router.get('/graph/portfolio/:userId', authenticateToken, async (req, res) => {
  try {
    const { userId } = req.params;
    
    const graph = await knowledgeGraph.getUserPortfolioGraph(userId);
    
    res.json({
      success: true,
      data: graph
    });
  } catch (error) {
    logger.error('Failed to get portfolio graph', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

router.get('/graph/concentration-risk/:userId', authenticateToken, async (req, res) => {
  try {
    const { userId } = req.params;
    const { threshold = 0.35 } = req.query;
    
    const risks = await knowledgeGraph.detectConcentrationRisk(userId, parseFloat(threshold));
    
    res.json({
      success: true,
      data: risks,
      hasRisk: risks.length > 0
    });
  } catch (error) {
    logger.error('Failed to detect concentration risk', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

router.get('/graph/stats', authenticateToken, async (req, res) => {
  try {
    const stats = await knowledgeGraph.getGraphStatistics();
    
    res.json({
      success: true,
      data: stats
    });
  } catch (error) {
    logger.error('Failed to get graph stats', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Vector Database endpoints

router.post('/vector/fund/index', authenticateToken, async (req, res) => {
  try {
    const { fundId, description, metadata } = req.body;
    
    await vectorDatabase.indexFund(fundId, description, metadata);
    
    res.json({
      success: true,
      message: 'Fund indexed successfully'
    });
  } catch (error) {
    logger.error('Failed to index fund', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

router.get('/vector/fund/similar/:fundId', authenticateToken, async (req, res) => {
  try {
    const { fundId } = req.params;
    const { topK = 10 } = req.query;
    
    const similarFunds = await vectorDatabase.findSimilarFunds(fundId, parseInt(topK));
    
    res.json({
      success: true,
      data: similarFunds
    });
  } catch (error) {
    logger.error('Failed to find similar funds', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

router.post('/vector/fund/search', authenticateToken, async (req, res) => {
  try {
    const { query, topK = 10, filter = {} } = req.body;
    
    const results = await vectorDatabase.searchFundsByQuery(query, topK, filter);
    
    res.json({
      success: true,
      data: results
    });
  } catch (error) {
    logger.error('Failed to search funds', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

router.get('/vector/user/similar/:userId', authenticateToken, async (req, res) => {
  try {
    const { userId } = req.params;
    const { topK = 10 } = req.query;
    
    const similarUsers = await vectorDatabase.findSimilarUsers(userId, parseInt(topK));
    
    res.json({
      success: true,
      data: similarUsers
    });
  } catch (error) {
    logger.error('Failed to find similar users', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Real-time Features endpoints

router.get('/features/user/:userId', authenticateToken, async (req, res) => {
  try {
    const { userId } = req.params;
    
    const features = await realTimeFeatures.getUserFeatures(userId);
    
    res.json({
      success: true,
      data: features
    });
  } catch (error) {
    logger.error('Failed to get user features', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

router.get('/features/portfolio/:userId', authenticateToken, async (req, res) => {
  try {
    const { userId } = req.params;
    
    const features = await realTimeFeatures.getPortfolioFeatures(userId);
    
    res.json({
      success: true,
      data: features
    });
  } catch (error) {
    logger.error('Failed to get portfolio features', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

router.get('/features/market', authenticateToken, async (req, res) => {
  try {
    const features = await realTimeFeatures.getMarketFeatures();
    
    res.json({
      success: true,
      data: features
    });
  } catch (error) {
    logger.error('Failed to get market features', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

router.get('/features/all/:userId', authenticateToken, async (req, res) => {
  try {
    const { userId } = req.params;
    
    const features = await realTimeFeatures.getAllFeatures(userId);
    
    res.json({
      success: true,
      data: features
    });
  } catch (error) {
    logger.error('Failed to get all features', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Event publishing endpoints

router.post('/events/user', authenticateToken, async (req, res) => {
  try {
    const { eventType, eventData } = req.body;
    const userId = req.userId;
    
    await kafkaProducer.publishUserEvent(userId, eventType, eventData);
    
    res.json({
      success: true,
      message: 'User event published'
    });
  } catch (error) {
    logger.error('Failed to publish user event', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

router.post('/events/portfolio', authenticateToken, async (req, res) => {
  try {
    const { changeType, changeData } = req.body;
    const userId = req.userId;
    
    await kafkaProducer.publishPortfolioChange(userId, changeType, changeData);
    
    res.json({
      success: true,
      message: 'Portfolio change event published'
    });
  } catch (error) {
    logger.error('Failed to publish portfolio event', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Health check
router.get('/health', async (req, res) => {
  try {
    const kafkaMetrics = kafkaProducer.getMetrics();
    const graphMetrics = knowledgeGraph.getMetrics();
    const vectorMetrics = vectorDatabase.getMetrics();
    const featureMetrics = await realTimeFeatures.getMetrics();
    
    res.json({
      success: true,
      services: {
        kafka: kafkaMetrics,
        knowledge_graph: graphMetrics,
        vector_database: vectorMetrics,
        real_time_features: featureMetrics
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

module.exports = router;
