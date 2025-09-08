const digioService = require('../services/digioService');
const demoDigioService = require('../services/demoDigioService');
const logger = require('../utils/logger');
const appConfig = require('../config/app');

class DigioController {
  constructor() {
    // Use demo service for development, real service for production
    this.service = appConfig.DEMO_MODE ? demoDigioService : digioService;
  }

  /**
   * 1. KYC Verification API
   */
  async initiateKYC(req, res) {
    try {
      const { kycData } = req.body;
      const userId = req.user.id;

      logger.info('Digio KYC initiation request', { userId, kycData });

      const result = await this.service.initiateKYC({
        ...kycData,
        userId
      });

      if (result.success) {
        res.status(201).json({
          success: true,
          message: 'KYC initiated successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'KYC initiation failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Digio KYC initiation error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  async getKYCStatus(req, res) {
    try {
      const { kycId } = req.params;
      const userId = req.user.id;

      logger.info('Digio KYC status request', { userId, kycId });

      const result = await this.service.getKYCStatus(kycId);

      if (result.success) {
        res.json({
          success: true,
          message: 'KYC status fetched successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'Failed to fetch KYC status',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Digio KYC status error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }
}

module.exports = new DigioController();
