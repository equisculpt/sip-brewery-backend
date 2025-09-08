const logger = require('../utils/logger');
const complianceEngine = require('../services/complianceEngine');
const { authenticateUser, adminAuth } = require('../middleware/auth');

class ComplianceController {
  /**
   * Generate SEBI report
   */
  async generateSEBIReport(req, res) {
    try {
      logger.info('SEBI report generation request received');

      const { userId } = req.user;
      const { period } = req.body;

      const result = await complianceEngine.generateSEBIReport(userId, period);

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'SEBI report generated successfully',
          data: result.data
        });
      } else {
        res.status(500).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('SEBI report generation controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to generate SEBI report',
        error: error.message
      });
    }
  }

  /**
   * Generate AMFI report
   */
  async generateAMFIReport(req, res) {
    try {
      logger.info('AMFI report generation request received');

      const { userId } = req.user;
      const { quarter } = req.body;

      const result = await complianceEngine.generateAMFIReport(userId, quarter);

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'AMFI report generated successfully',
          data: result.data
        });
      } else {
        res.status(500).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('AMFI report generation controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to generate AMFI report',
        error: error.message
      });
    }
  }
}

module.exports = new ComplianceController();
