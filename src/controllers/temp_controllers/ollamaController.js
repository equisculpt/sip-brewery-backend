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

      // Add compliance disclaimer if needed
      let finalAnswer = ragResponse.answer;
      if (!complianceCheck.overallCompliant) {
        finalAnswer += '\n\n*This information is for educational purposes only. Please consult your financial advisor before making investment decisions.*';
      }

      const response = {
        success: true,
        answer: finalAnswer,
        sources: ragResponse.sources,
        compliance: {
          compliant: complianceCheck.overallCompliant,
          issues: complianceCheck.summary.criticalIssues
        },
        metadata: {
          timestamp: new Date().toISOString(),
          userId: userId,
          questionLength: question.length,
          answerLength: finalAnswer.length
        }
      };

      logger.info(`Question answered successfully for user ${userId}`);
      res.json(response);

    } catch (error) {
      logger.error('Error processing question:', error);
      res.status(500).json({
        success: false,
        message: 'Error processing your question. Please try again.',
        error: error.message
      });
    }
  }

  /**
   * Generate training dataset
   */
  async generateTrainingData(req, res) {
    try {
      logger.info('Starting training data generation...');

      const result = await this.trainingDataService.generateCompleteDataset();

      // Validate the generated data
      const validationResults = {};
      for (const file of result.files) {
        validationResults[file] = await this.trainingDataService.validateTrainingData(file);
      }

      const response = {
        success: true,
        message: 'Training dataset generated successfully',
        data: result,
        validation: validationResults,
        metadata: {
          timestamp: new Date().toISOString(),
          totalPairs: result.totalPairs,
          categories: result.categories
        }
      };

      logger.info(`Training data generated: ${result.totalPairs} Q&A pairs`);
      res.json(response);

    } catch (error) {
      logger.error('Error generating training data:', error);
      res.status(500).json({
        success: false,
        message: 'Error generating training data',
        error: error.message
      });
    }
  }

  /**
   * Get Ollama model status
   */
  async getModelStatus(req, res) {
    try {
      const response = await axios.get(`${this.ollamaBaseUrl}/api/tags`);
      const models = response.data.models || [];

      const status = {
        success: true,
        ollamaRunning: true,
        models: models.map(model => ({
          name: model.name,
          size: model.size,
          modified: model.modified_at
        })),
        ragStats: this.ragService.getRAGStats(),
        metadata: {
          timestamp: new Date().toISOString(),
          totalModels: models.length
        }
      };

      res.json(status);

    } catch (error) {
      logger.error('Error getting model status:', error);
      res.status(500).json({
        success: false,
        ollamaRunning: false,
        message: 'Ollama service not available',
        error: error.message
      });
    }
  }

  /**
   * Run compliance audit
   */
  async runComplianceAudit(req, res) {
    try {
      const { data } = req.body;

      logger.info('Running compliance audit...');

      const auditResult = await this.complianceService.runComplianceAudit(data);
      const report = await this.complianceService.generateComplianceReport();

      // Save report
      const reportPath = await this.complianceService.saveComplianceReport(report);

      const response = {
        success: true,
        audit: auditResult,
        report: report,
        reportPath: reportPath,
        metadata: {
          timestamp: new Date().toISOString(),
          totalChecks: auditResult.summary.total,
          compliantChecks: auditResult.summary.compliant,
          criticalIssues: auditResult.summary.criticalIssues
        }
      };

      logger.info(`Compliance audit completed: ${auditResult.summary.compliant}/${auditResult.summary.total} checks passed`);
      res.json(response);

    } catch (error) {
      logger.error('Error running compliance audit:', error);
      res.status(500).json({
        success: false,
        message: 'Error running compliance audit',
        error: error.message
      });
    }
  }

  /**
   * Get portfolio analytics
   */
  async getPortfolioAnalytics(req, res) {
    try {
      const { portfolioData } = req.body;

      if (!portfolioData) {
        return res.status(400).json({
          success: false,
          message: 'Portfolio data is required'
        });
      }

      logger.info('Calculating portfolio analytics...');

      const analytics = await this.portfolioAnalyticsService.calculatePortfolioAnalytics(portfolioData);

      const response = {
        success: true,
        analytics: analytics,
        metadata: {
          timestamp: new Date().toISOString(),
          totalFunds: analytics.basic.numberOfFunds,
          totalValue: analytics.basic.totalCurrentValue,
          xirr: analytics.basic.xirr
        }
      };

      logger.info('Portfolio analytics calculated successfully');
      res.json(response);

    } catch (error) {
      logger.error('Error calculating portfolio analytics:', error);
      res.status(500).json({
        success: false,
        message: 'Error calculating portfolio analytics',
        error: error.message
      });
    }
  }

  /**
   * Add document to RAG
   */
  async addDocument(req, res) {
    try {
      const { document } = req.body;

      if (!document || !document.content) {
        return res.status(400).json({
          success: false,
          message: 'Document with content is required'
        });
      }

      logger.info(`Adding document to RAG: ${document.id}`);

      const success = await this.ragService.addDocument(document);

      if (success) {
        // Save documents to storage
        await this.ragService.saveDocuments();

        const response = {
          success: true,
          message: 'Document added successfully',
          documentId: document.id,
          ragStats: this.ragService.getRAGStats(),
          metadata: {
            timestamp: new Date().toISOString(),
            contentLength: document.content.length
          }
        };

        logger.info(`Document added successfully: ${document.id}`);
        res.json(response);
      } else {
        res.status(500).json({
          success: false,
          message: 'Failed to add document'
        });
      }

    } catch (error) {
      logger.error('Error adding document:', error);
      res.status(500).json({
        success: false,
        message: 'Error adding document',
        error: error.message
      });
    }
  }

  /**
   * Search documents in RAG
   */
  async searchDocuments(req, res) {
    try {
      const { query, category, limit = 5 } = req.query;

      if (!query) {
        return res.status(400).json({
          success: false,
          message: 'Search query is required'
        });
      }

      logger.info(`Searching documents: ${query}`);

      const results = await this.ragService.searchDocuments(query, category, parseInt(limit));

      const response = {
        success: true,
        query: query,
        results: results,
        metadata: {
          timestamp: new Date().toISOString(),
          resultsCount: results.length,
          category: category || 'all'
        }
      };

      logger.info(`Search completed: ${results.length} results found`);
      res.json(response);

    } catch (error) {
      logger.error('Error searching documents:', error);
      res.status(500).json({
        success: false,
        message: 'Error searching documents',
        error: error.message
      });
    }
  }

  /**
   * Get system health
   */
  async getSystemHealth(req, res) {
    try {
      const health = {
        success: true,
        status: 'healthy',
        services: {
          ollama: false,
          rag: false,
          compliance: false,
          training: false
        },
        metadata: {
          timestamp: new Date().toISOString(),
          uptime: process.uptime()
        }
      };

      // Check Ollama
      try {
        await axios.get(`${this.ollamaBaseUrl}/api/tags`);
        health.services.ollama = true;
      } catch (error) {
        health.status = 'degraded';
      }

      // Check RAG
      try {
        const ragStats = this.ragService.getRAGStats();
        health.services.rag = ragStats.totalDocuments > 0;
      } catch (error) {
        health.status = 'degraded';
      }

      // Check compliance
      try {
        health.services.compliance = this.complianceService.sebiGuidelines.size > 0;
      } catch (error) {
        health.status = 'degraded';
      }

      // Check training
      try {
        health.services.training = true; // Training service is always available
      } catch (error) {
        health.status = 'degraded';
      }

      // Overall status
      const healthyServices = Object.values(health.services).filter(Boolean).length;
      if (healthyServices < 2) {
        health.status = 'unhealthy';
      }

      res.json(health);

    } catch (error) {
      logger.error('Error getting system health:', error);
      res.status(500).json({
        success: false,
        status: 'error',
        message: 'Error checking system health',
        error: error.message
      });
    }
  }

  /**
   * Get RAG statistics
   */
  async getRAGStats(req, res) {
    try {
      const stats = this.ragService.getRAGStats();

      const response = {
        success: true,
        stats: stats,
        metadata: {
          timestamp: new Date().toISOString()
        }
      };

      res.json(response);

    } catch (error) {
      logger.error('Error getting RAG stats:', error);
      res.status(500).json({
        success: false,
        message: 'Error getting RAG statistics',
        error: error.message
      });
    }
  }

  /**
   * Test Ollama connection
   */
  async testOllamaConnection(req, res) {
    try {
      const response = await axios.get(`${this.ollamaBaseUrl}/api/tags`);
      
      const testResult = {
        success: true,
        connected: true,
        models: response.data.models || [],
        metadata: {
          timestamp: new Date().toISOString(),
          ollamaVersion: '0.9.6' // You might want to get this dynamically
        }
      };

      res.json(testResult);

    } catch (error) {
      logger.error('Ollama connection test failed:', error);
      res.status(500).json({
        success: false,
        connected: false,
        message: 'Ollama service not available',
        error: error.message
      });
    }
  }
}

module.exports = OllamaController; 