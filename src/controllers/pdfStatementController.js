const pdfStatementService = require('../services/pdfStatementService');
const response = require('../utils/response');
const logger = require('../utils/logger');

// PDF rendering is disabled in test mode
const renderPDFStatement = null;

class PDFStatementController {
  /**
   * Generate PDF statement
   */
  async generateStatement(req, res) {
    try {
      if (!req.userId) {
        return response.error(res, 'Authentication required', 401);
      }

      const { statementType = 'comprehensive', dateRange } = req.body;
      // Validate statement type
      const validTypes = [
        'comprehensive', 'holdings', 'transactions', 'pnl', 
        'capital-gain', 'tax', 'rewards', 'smart-sip'
      ];
      if (!validTypes.includes(statementType)) {
        return response.error(res, 'Invalid statement type', 400);
      }
      logger.info('Generating PDF statement', { 
        userId: req.userId, 
        statementType,
        dateRange 
      });
      // Get user data
      const user = await this.getUserData(req.userId);
      if (!user) {
        return response.error(res, 'User not found', 404);
      }
      // Get portfolio data
      const portfolio = await this.getPortfolioData(req.userId);
      if (!portfolio) {
        return response.error(res, 'Portfolio not found', 404);
      }
      // Get transactions
      const transactions = await this.getTransactionData(req.userId, dateRange);
      // Get rewards data
      const rewards = await this.getRewardsData(req.userId);
      // Process data for PDF
      const processedData = pdfStatementService.processUserData(
        user, 
        portfolio, 
        transactions, 
        rewards
      );
      // Generate PDF
      const pdfBuffer = await pdfStatementService.generatePDF(processedData, statementType);
      res.setHeader('Content-Type', 'application/pdf');
      res.setHeader('Content-Disposition', 'attachment; filename=statement.pdf');
      res.send(pdfBuffer);
    } catch (error) {
      logger.error('Error generating PDF statement', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to generate PDF statement',
        error: error.message
      });
    }
  }
  // Placeholder for data fetching methods
  async getUserData(userId) { return {}; }
  async getPortfolioData(userId) { return {}; }
  async getTransactionData(userId, dateRange) { return []; }
  async getRewardsData(userId) { return []; }
}

module.exports = new PDFStatementController();
