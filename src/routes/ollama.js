const express = require('express');
const router = express.Router();
const OllamaController = require('../controllers/ollamaController');
const { authenticateToken } = require('../middleware/auth');

// Initialize controller
const ollamaController = new OllamaController();

// Initialize controller on startup
ollamaController.initialize().catch(error => {
  console.error('Failed to initialize Ollama controller:', error);
});

/**
 * @route   POST /api/ollama/ask
 * @desc    Ask a question using RAG + Ollama
 * @access  Private
 */
router.post('/ask', authenticateToken, async (req, res) => {
  await ollamaController.askQuestion(req, res);
});

/**
 * @route   POST /api/ollama/training/generate
 * @desc    Generate training dataset for fine-tuning
 * @access  Private (Admin only)
 */
router.post('/training/generate', authenticateToken, async (req, res) => {
  await ollamaController.generateTrainingData(req, res);
});

/**
 * @route   GET /api/ollama/status
 * @desc    Get Ollama model status and RAG statistics
 * @access  Private
 */
router.get('/status', authenticateToken, async (req, res) => {
  await ollamaController.getModelStatus(req, res);
});

/**
 * @route   POST /api/ollama/compliance/audit
 * @desc    Run compliance audit
 * @access  Private
 */
router.post('/compliance/audit', authenticateToken, async (req, res) => {
  await ollamaController.runComplianceAudit(req, res);
});

/**
 * @route   POST /api/ollama/portfolio/analytics
 * @desc    Get comprehensive portfolio analytics
 * @access  Private
 */
router.post('/portfolio/analytics', authenticateToken, async (req, res) => {
  await ollamaController.getPortfolioAnalytics(req, res);
});

/**
 * @route   POST /api/ollama/rag/document
 * @desc    Add document to RAG vector database
 * @access  Private
 */
router.post('/rag/document', authenticateToken, async (req, res) => {
  await ollamaController.addDocument(req, res);
});

/**
 * @route   GET /api/ollama/rag/search
 * @desc    Search documents in RAG
 * @access  Private
 */
router.get('/rag/search', authenticateToken, async (req, res) => {
  await ollamaController.searchDocuments(req, res);
});

/**
 * @route   GET /api/ollama/health
 * @desc    Get system health status
 * @access  Private
 */
router.get('/health', authenticateToken, async (req, res) => {
  await ollamaController.getSystemHealth(req, res);
});

/**
 * @route   GET /api/ollama/rag/stats
 * @desc    Get RAG statistics
 * @access  Private
 */
router.get('/rag/stats', authenticateToken, async (req, res) => {
  await ollamaController.getRAGStats(req, res);
});

/**
 * @route   GET /api/ollama/test
 * @desc    Test Ollama connection
 * @access  Private
 */
router.get('/test', authenticateToken, async (req, res) => {
  await ollamaController.testOllamaConnection(req, res);
});

/**
 * @route   GET /api/ollama/models
 * @desc    List available Ollama models
 * @access  Private
 */
router.get('/models', authenticateToken, async (req, res) => {
  try {
    const response = await ollamaController.getModelStatus(req, res);
    // This will be handled by the controller
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Error getting models',
      error: error.message
    });
  }
});

/**
 * @route   POST /api/ollama/chat
 * @desc    Chat endpoint for conversational AI
 * @access  Private
 */
router.post('/chat', authenticateToken, async (req, res) => {
  try {
    const { message, userId, conversationId } = req.body;

    if (!message) {
      return res.status(400).json({
        success: false,
        message: 'Message is required'
      });
    }

    // Add conversation context if available
    const context = conversationId ? `Conversation ID: ${conversationId}. ` : '';
    
    // Use the ask endpoint logic
    req.body.question = message;
    req.body.context = context;
    
    await ollamaController.askQuestion(req, res);

  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Error processing chat message',
      error: error.message
    });
  }
});

/**
 * @route   POST /api/ollama/financial-advice
 * @desc    Get financial advice with compliance checks
 * @access  Private
 */
router.post('/financial-advice', authenticateToken, async (req, res) => {
  try {
    const { query, portfolioData, userId } = req.body;

    if (!query) {
      return res.status(400).json({
        success: false,
        message: 'Query is required'
      });
    }

    // Add portfolio context if available
    let context = '';
    if (portfolioData) {
      const analytics = await ollamaController.portfolioAnalyticsService.calculatePortfolioAnalytics(portfolioData);
      context = `Portfolio Context: Total Value: ₹${analytics.basic.totalCurrentValue}, XIRR: ${analytics.basic.xirr}%, Risk Level: ${analytics.risk.riskLevel}. `;
    }

    // Use the ask endpoint with enhanced context
    req.body.question = query;
    req.body.context = context;
    
    await ollamaController.askQuestion(req, res);

  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Error processing financial advice request',
      error: error.message
    });
  }
});

/**
 * @route   POST /api/ollama/sip-recommendation
 * @desc    Get SIP recommendation with market analysis
 * @access  Private
 */
router.post('/sip-recommendation', authenticateToken, async (req, res) => {
  try {
    const { userId, riskProfile, investmentGoal, currentSIP } = req.body;

    // Build context for SIP recommendation
    const context = `User Profile: Risk Profile: ${riskProfile}, Investment Goal: ${investmentGoal}, Current SIP: ₹${currentSIP || 0}. `;

    const question = `Provide a SIP recommendation for a ${riskProfile} risk profile investor with ${investmentGoal} investment goal. Consider current market conditions and suggest optimal fund allocation.`;

    req.body.question = question;
    req.body.context = context;
    req.body.userId = userId;

    await ollamaController.askQuestion(req, res);

  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Error processing SIP recommendation request',
      error: error.message
    });
  }
});

/**
 * @route   POST /api/ollama/fund-comparison
 * @desc    Compare mutual funds
 * @access  Private
 */
router.post('/fund-comparison', authenticateToken, async (req, res) => {
  try {
    const { fund1, fund2, comparisonCriteria } = req.body;

    if (!fund1 || !fund2) {
      return res.status(400).json({
        success: false,
        message: 'Two funds are required for comparison'
      });
    }

    const question = `Compare ${fund1} and ${fund2} mutual funds based on ${comparisonCriteria || 'performance, expense ratio, and risk metrics'}. Provide a detailed analysis suitable for Indian investors.`;

    req.body.question = question;
    req.body.userId = req.user.id;

    await ollamaController.askQuestion(req, res);

  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Error processing fund comparison request',
      error: error.message
    });
  }
});

/**
 * @route   POST /api/ollama/tax-optimization
 * @desc    Get tax optimization advice
 * @access  Private
 */
router.post('/tax-optimization', authenticateToken, async (req, res) => {
  try {
    const { portfolioData, taxSlab, userId } = req.body;

    if (!portfolioData) {
      return res.status(400).json({
        success: false,
        message: 'Portfolio data is required for tax optimization'
      });
    }

    // Calculate tax metrics first
    const taxMetrics = await ollamaController.portfolioAnalyticsService.calculateTaxMetrics(portfolioData);
    
    const context = `Tax Context: Tax Slab: ${taxSlab}, Total Tax Liability: ₹${taxMetrics.taxLiability.total}, Unrealized Gains: ₹${Object.values(taxMetrics.unrealizedGains).reduce((sum, gain) => sum + gain.amount, 0)}. `;

    const question = `Provide tax optimization advice for my mutual fund portfolio. Consider my tax slab (${taxSlab}) and suggest strategies to minimize tax liability while maintaining investment goals.`;

    req.body.question = question;
    req.body.context = context;
    req.body.userId = userId;

    await ollamaController.askQuestion(req, res);

  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Error processing tax optimization request',
      error: error.message
    });
  }
});

/**
 * @route   GET /api/ollama/status/public
 * @desc    Public status endpoint (no auth required)
 * @access  Public
 */
router.get('/status/public', async (req, res) => {
  try {
    const health = {
      success: true,
      status: 'operational',
      services: {
        ollama: false,
        rag: false
      },
      timestamp: new Date().toISOString()
    };

    // Check basic services without full initialization
    try {
      const axios = require('axios');
      await axios.get('http://localhost:11434/api/tags');
      health.services.ollama = true;
    } catch (error) {
      health.status = 'degraded';
    }

    res.json(health);

  } catch (error) {
    res.status(500).json({
      success: false,
      status: 'error',
      message: 'Service unavailable'
    });
  }
});

module.exports = router; 