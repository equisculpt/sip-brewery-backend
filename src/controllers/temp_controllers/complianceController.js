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

  /**
   * Check regulatory violations
   */
  async checkRegulatoryViolations(req, res) {
    try {
      logger.info('Regulatory violations check request received');

      const { userId } = req.user;

      const result = await complianceEngine.checkRegulatoryViolations(userId);

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'Regulatory violations checked successfully',
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
      logger.error('Regulatory violations check controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to check regulatory violations',
        error: error.message
      });
    }
  }

  /**
   * Generate admin reports (Admin only)
   */
  async generateAdminReports(req, res) {
    try {
      logger.info('Admin reports generation request received');

      const { adminId } = req.user;
      const { reportType } = req.body;

      const result = await complianceEngine.generateAdminReports(adminId, reportType);

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'Admin reports generated successfully',
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
      logger.error('Admin reports generation controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to generate admin reports',
        error: error.message
      });
    }
  }

  /**
   * Get compliance metrics (Admin only)
   */
  async getComplianceMetrics(req, res) {
    try {
      logger.info('Compliance metrics request received');

      const result = await complianceEngine.getComplianceMetrics();

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'Compliance metrics retrieved successfully',
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
      logger.error('Compliance metrics controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get compliance metrics',
        error: error.message
      });
    }
  }

  /**
   * Monitor real-time compliance (Admin only)
   */
  async monitorRealTimeCompliance(req, res) {
    try {
      logger.info('Real-time compliance monitoring request received');

      const result = await complianceEngine.monitorRealTimeCompliance();

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'Real-time compliance monitoring completed',
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
      logger.error('Real-time compliance monitoring controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to monitor real-time compliance',
        error: error.message
      });
    }
  }

  /**
   * Get user compliance status
   */
  async getUserComplianceStatus(req, res) {
    try {
      logger.info('User compliance status request received');

      const { userId } = req.user;

      // Check violations for the user
      const violations = await complianceEngine.checkRegulatoryViolations(userId);

      if (!violations.success) {
        return res.status(500).json({
          success: false,
          message: 'Failed to get user compliance status',
          error: violations.error
        });
      }

      const complianceStatus = {
        totalViolations: violations.data.totalViolations,
        highPriorityViolations: violations.data.highPriorityViolations,
        violations: violations.data.violations,
        alerts: violations.data.alerts,
        complianceScore: this.calculateComplianceScore(violations.data.violations),
        status: this.getComplianceStatus(violations.data.violations),
        lastChecked: new Date().toISOString()
      };

      res.status(200).json({
        success: true,
        message: 'User compliance status retrieved successfully',
        data: complianceStatus
      });
    } catch (error) {
      logger.error('User compliance status controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get user compliance status',
        error: error.message
      });
    }
  }

  /**
   * Get compliance dashboard (Admin only)
   */
  async getComplianceDashboard(req, res) {
    try {
      logger.info('Compliance dashboard request received');

      // Get all compliance data
      const [
        metrics,
        monitoring,
        violations
      ] = await Promise.all([
        complianceEngine.getComplianceMetrics(),
        complianceEngine.monitorRealTimeCompliance(),
        this.getSystemWideViolations()
      ]);

      const dashboardData = {
        metrics: metrics.success ? metrics.data : null,
        monitoring: monitoring.success ? monitoring.data : null,
        violations: violations,
        lastUpdated: new Date().toISOString()
      };

      res.status(200).json({
        success: true,
        message: 'Compliance dashboard data retrieved successfully',
        data: dashboardData
      });
    } catch (error) {
      logger.error('Compliance dashboard controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get compliance dashboard data',
        error: error.message
      });
    }
  }

  /**
   * Get violation trends (Admin only)
   */
  async getViolationTrends(req, res) {
    try {
      logger.info('Violation trends request received');

      const { period } = req.query;
      const analysisPeriod = period || '1m';

      const metrics = await complianceEngine.getComplianceMetrics();

      if (!metrics.success) {
        return res.status(500).json({
          success: false,
          message: 'Failed to get violation trends',
          error: metrics.error
        });
      }

      const trends = {
        overall: metrics.data.overall,
        byCategory: metrics.data.byCategory,
        trends: metrics.data.trends,
        period: analysisPeriod,
        timestamp: new Date().toISOString()
      };

      res.status(200).json({
        success: true,
        message: 'Violation trends retrieved successfully',
        data: trends
      });
    } catch (error) {
      logger.error('Violation trends controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get violation trends',
        error: error.message
      });
    }
  }

  /**
   * Get compliance alerts (Admin only)
   */
  async getComplianceAlerts(req, res) {
    try {
      logger.info('Compliance alerts request received');

      const metrics = await complianceEngine.getComplianceMetrics();

      if (!metrics.success) {
        return res.status(500).json({
          success: false,
          message: 'Failed to get compliance alerts',
          error: metrics.error
        });
      }

      const alerts = {
        activeAlerts: metrics.data.alerts,
        highPriorityAlerts: metrics.data.alerts.filter(a => a.priority === 'HIGH'),
        totalAlerts: metrics.data.alerts.length,
        lastUpdated: new Date().toISOString()
      };

      res.status(200).json({
        success: true,
        message: 'Compliance alerts retrieved successfully',
        data: alerts
      });
    } catch (error) {
      logger.error('Compliance alerts controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get compliance alerts',
        error: error.message
      });
    }
  }

  /**
   * Download compliance report
   */
  async downloadComplianceReport(req, res) {
    try {
      logger.info('Compliance report download request received');

      const { userId } = req.user;
      const { reportType, period } = req.params;

      let result;
      if (reportType === 'SEBI') {
        result = await complianceEngine.generateSEBIReport(userId, period);
      } else if (reportType === 'AMFI') {
        result = await complianceEngine.generateAMFIReport(userId, period);
      } else {
        return res.status(400).json({
          success: false,
          message: 'Invalid report type',
          error: 'Supported types: SEBI, AMFI'
        });
      }

      if (!result.success) {
        return res.status(500).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }

      // Send PDF file
      const fs = require('fs');
      const path = require('path');

      if (fs.existsSync(result.data.pdfPath)) {
        res.download(result.data.pdfPath, `${reportType}_Report_${period}.pdf`, (err) => {
          if (err) {
            logger.error('PDF download failed', { error: err.message });
            res.status(500).json({
              success: false,
              message: 'Failed to download report',
              error: err.message
            });
          }
        });
      } else {
        res.status(404).json({
          success: false,
          message: 'Report file not found',
          error: 'PDF file does not exist'
        });
      }
    } catch (error) {
      logger.error('Compliance report download controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to download compliance report',
        error: error.message
      });
    }
  }

  /**
   * Get compliance summary for user
   */
  async getComplianceSummary(req, res) {
    try {
      logger.info('Compliance summary request received');

      const { userId } = req.user;

      // Get all compliance data for user
      const [
        violations,
        sebiReport,
        amfiReport
      ] = await Promise.all([
        complianceEngine.checkRegulatoryViolations(userId),
        complianceEngine.generateSEBIReport(userId, 'monthly'),
        complianceEngine.generateAMFIReport(userId, 'Q1')
      ]);

      const summary = {
        violations: violations.success ? violations.data : null,
        sebiReport: sebiReport.success ? sebiReport.data : null,
        amfiReport: amfiReport.success ? amfiReport.data : null,
        complianceScore: this.calculateComplianceScore(violations.success ? violations.data.violations : []),
        status: this.getComplianceStatus(violations.success ? violations.data.violations : []),
        lastUpdated: new Date().toISOString()
      };

      res.status(200).json({
        success: true,
        message: 'Compliance summary retrieved successfully',
        data: summary
      });
    } catch (error) {
      logger.error('Compliance summary controller error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Failed to get compliance summary',
        error: error.message
      });
    }
  }

  // Helper methods
  calculateComplianceScore(violations) {
    try {
      if (violations.length === 0) return 100;

      const highPriorityViolations = violations.filter(v => v.priority === 'HIGH').length;
      const mediumPriorityViolations = violations.filter(v => v.priority === 'MEDIUM').length;
      const lowPriorityViolations = violations.filter(v => v.priority === 'LOW').length;

      // Calculate score based on violation severity
      const score = 100 - (highPriorityViolations * 20) - (mediumPriorityViolations * 10) - (lowPriorityViolations * 5);
      return Math.max(0, score);
    } catch (error) {
      logger.error('Compliance score calculation failed', { error: error.message });
      return 0;
    }
  }

  getComplianceStatus(violations) {
    try {
      const highPriorityViolations = violations.filter(v => v.priority === 'HIGH').length;
      const totalViolations = violations.length;

      if (highPriorityViolations > 0) return 'NON_COMPLIANT';
      if (totalViolations > 5) return 'NEEDS_ATTENTION';
      if (totalViolations > 0) return 'MINOR_VIOLATIONS';
      return 'COMPLIANT';
    } catch (error) {
      logger.error('Compliance status determination failed', { error: error.message });
      return 'UNKNOWN';
    }
  }

  async getSystemWideViolations() {
    try {
      // Mock system-wide violations data
      return {
        totalViolations: 150,
        highPriorityViolations: 25,
        mediumPriorityViolations: 75,
        lowPriorityViolations: 50,
        resolvedViolations: 100,
        complianceRate: 0.85
      };
    } catch (error) {
      logger.error('System-wide violations retrieval failed', { error: error.message });
      return {
        totalViolations: 0,
        highPriorityViolations: 0,
        mediumPriorityViolations: 0,
        lowPriorityViolations: 0,
        resolvedViolations: 0,
        complianceRate: 0
      };
    }
  }
}

module.exports = new ComplianceController(); 