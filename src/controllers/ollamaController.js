const RAGService = require('../services/ragService');
const TrainingDataService = require('../services/trainingDataService');
const ComplianceService = require('../services/complianceService');
const PortfolioAnalyticsService = require('../services/portfolioAnalyticsService');
const logger = require('../utils/logger');
const axios = require('axios');

class OllamaController {
  constructor() {
    this.ragService = new RAGService();
    this.trainingDataService = new TrainingDataService();
    this.complianceService = new ComplianceService();
    this.portfolioAnalyticsService = new PortfolioAnalyticsService();
    this.ollamaBaseUrl = 'http://localhost:11434';
  }

  /**
   * Initialize all services
   */
  async initialize() {
    try {
      await this.ragService.initialize();
      await this.complianceService.initialize();
      // Add default documents to RAG
      await this.ragService.addSEBIDocuments();
      await this.ragService.addMFEducationalContent();
      logger.info('Ollama Controller initialized successfully');
      return true;
    } catch (error) {
      logger.error('Error initializing Ollama Controller:', error);
      return false;
    }
  }

  /**
   * Ask question using RAG + Ollama
   */
  async askQuestion(req, res) {
    try {
      const { question, userId, context } = req.body;
      if (!question) {
        return res.status(400).json({
          success: false,
          message: 'Question is required'
        });
      }
      logger.info(`Processing question from user ${userId}: ${question}`);
      // Generate answer using RAG
      const ragResponse = await this.ragService.generateAnswer(question, context);
      // Check compliance
      const complianceCheck = await this.complianceService.runComplianceAudit({
        aiResponse: ragResponse.answer,
        userQuery: question,
        userId: userId
      });
      res.json({
        success: true,
        answer: ragResponse.answer,
        compliance: complianceCheck
      });
    } catch (error) {
      logger.error('Error in askQuestion:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to process question',
        error: error.message
      });
    }
  }
}

module.exports = new OllamaController();
