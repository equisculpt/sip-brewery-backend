const bseStarMFService = require('../services/bseStarMFService');
const demoBSEStarMFService = require('../services/demoBSEStarMFService');
const logger = require('../utils/logger');
const appConfig = require('../config/app');

class BSEStarMFController {
  constructor() {
    // Use demo service for development, real service for production
    this.service = appConfig.DEMO_MODE ? demoBSEStarMFService : bseStarMFService;
  }

  /**
   * 1. Client Creation API (AddClient/ModifyClient)
   */
  async createClient(req, res) {
    try {
      const { clientData } = req.body;
      const userId = req.user.id;

      logger.info('BSE Star MF client creation request', { userId, clientData });

      const result = await this.service.createClient({
        ...clientData,
        userId
      });

      if (result.success) {
        res.status(201).json({
          success: true,
          message: 'Client created successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'Client creation failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('BSE Star MF client creation error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  async modifyClient(req, res) {
    try {
      const { clientId } = req.params;
      const { clientData } = req.body;
      const userId = req.user.id;

      logger.info('BSE Star MF client modification request', { userId, clientId, clientData });

      const result = await this.service.modifyClient(clientId, {
        ...clientData,
        userId
      });

      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'Client modified successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'Client modification failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('BSE Star MF client modification error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }
}

module.exports = new BSEStarMFController();
