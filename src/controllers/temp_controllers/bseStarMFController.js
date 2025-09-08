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
        res.json({
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

  /**
   * 2. Scheme Master Data API
   */
  async getSchemeMasterData(req, res) {
    try {
      const filters = req.query;
      const userId = req.user.id;

      logger.info('BSE Star MF scheme master data request', { userId, filters });

      const result = await this.service.getSchemeMasterData(filters);

      if (result.success) {
        res.json({
          success: true,
          message: 'Scheme master data retrieved successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'Failed to retrieve scheme master data',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('BSE Star MF scheme master data error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  async getSchemeDetails(req, res) {
    try {
      const { schemeCode } = req.params;
      const userId = req.user.id;

      logger.info('BSE Star MF scheme details request', { userId, schemeCode });

      const result = await this.service.getSchemeDetails(schemeCode);

      if (result.success) {
        res.json({
          success: true,
          message: 'Scheme details retrieved successfully',
          data: result.data
        });
      } else {
        res.status(404).json({
          success: false,
          message: 'Scheme not found',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('BSE Star MF scheme details error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * 3. Lumpsum Order Placement API
   */
  async placeLumpsumOrder(req, res) {
    try {
      const { orderData } = req.body;
      const userId = req.user.id;

      logger.info('BSE Star MF lumpsum order request', { userId, orderData });

      const result = await this.service.placeLumpsumOrder({
        ...orderData,
        userId
      });

      if (result.success) {
        res.status(201).json({
          success: true,
          message: 'Order placed successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'Order placement failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('BSE Star MF lumpsum order error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * 4. Order Status API
   */
  async getOrderStatus(req, res) {
    try {
      const { orderId } = req.params;
      const userId = req.user.id;

      logger.info('BSE Star MF order status request', { userId, orderId });

      const result = await this.service.getOrderStatus(orderId);

      if (result.success) {
        res.json({
          success: true,
          message: 'Order status retrieved successfully',
          data: result.data
        });
      } else {
        res.status(404).json({
          success: false,
          message: 'Order not found',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('BSE Star MF order status error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * 5. Redemption API
   */
  async placeRedemptionOrder(req, res) {
    try {
      const { redemptionData } = req.body;
      const userId = req.user.id;

      logger.info('BSE Star MF redemption order request', { userId, redemptionData });

      const result = await this.service.placeRedemptionOrder({
        ...redemptionData,
        userId
      });

      if (result.success) {
        res.status(201).json({
          success: true,
          message: 'Redemption order placed successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'Redemption order placement failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('BSE Star MF redemption order error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * 6. Transaction Report API
   */
  async getTransactionReport(req, res) {
    try {
      const filters = req.query;
      const userId = req.user.id;

      logger.info('BSE Star MF transaction report request', { userId, filters });

      const result = await this.service.getTransactionReport({
        ...filters,
        userId
      });

      if (result.success) {
        res.json({
          success: true,
          message: 'Transaction report retrieved successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'Failed to retrieve transaction report',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('BSE Star MF transaction report error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * 7. NAV & Holding Report API
   */
  async getNAVAndHoldingReport(req, res) {
    try {
      const filters = req.query;
      const userId = req.user.id;

      logger.info('BSE Star MF NAV and holding report request', { userId, filters });

      const result = await this.service.getNAVAndHoldingReport({
        ...filters,
        userId
      });

      if (result.success) {
        res.json({
          success: true,
          message: 'NAV and holding report retrieved successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'Failed to retrieve NAV and holding report',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('BSE Star MF NAV and holding report error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  async getCurrentNAV(req, res) {
    try {
      const { schemeCodes } = req.body;
      const userId = req.user.id;

      logger.info('BSE Star MF current NAV request', { userId, schemeCodes });

      const result = await this.service.getCurrentNAV(schemeCodes);

      if (result.success) {
        res.json({
          success: true,
          message: 'Current NAV retrieved successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'Failed to retrieve current NAV',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('BSE Star MF current NAV error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * 8. eMandate via BSE
   */
  async setupEMandate(req, res) {
    try {
      const { mandateData } = req.body;
      const userId = req.user.id;

      logger.info('BSE Star MF eMandate setup request', { userId, mandateData });

      const result = await this.service.setupEMandate({
        ...mandateData,
        userId
      });

      if (result.success) {
        res.status(201).json({
          success: true,
          message: 'eMandate setup initiated successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'eMandate setup failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('BSE Star MF eMandate setup error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  async getEMandateStatus(req, res) {
    try {
      const { mandateId } = req.params;
      const userId = req.user.id;

      logger.info('BSE Star MF eMandate status request', { userId, mandateId });

      const result = await this.service.getEMandateStatus(mandateId);

      if (result.success) {
        res.json({
          success: true,
          message: 'eMandate status retrieved successfully',
          data: result.data
        });
      } else {
        res.status(404).json({
          success: false,
          message: 'Mandate not found',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('BSE Star MF eMandate status error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  async cancelEMandate(req, res) {
    try {
      const { mandateId } = req.params;
      const { reason } = req.body;
      const userId = req.user.id;

      logger.info('BSE Star MF eMandate cancellation request', { userId, mandateId, reason });

      const result = await this.service.cancelEMandate(mandateId, reason);

      if (result.success) {
        res.json({
          success: true,
          message: 'eMandate cancelled successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'eMandate cancellation failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('BSE Star MF eMandate cancellation error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Additional helper endpoints
   */
  async getClientFolios(req, res) {
    try {
      const { clientId } = req.params;
      const userId = req.user.id;

      logger.info('BSE Star MF client folios request', { userId, clientId });

      const result = await this.service.getClientFolios(clientId);

      if (result.success) {
        res.json({
          success: true,
          message: 'Client folios retrieved successfully',
          data: result.data
        });
      } else {
        res.status(404).json({
          success: false,
          message: 'Client not found',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('BSE Star MF client folios error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  async getSchemePerformance(req, res) {
    try {
      const { schemeCode } = req.params;
      const { period } = req.query;
      const userId = req.user.id;

      logger.info('BSE Star MF scheme performance request', { userId, schemeCode, period });

      const result = await this.service.getSchemePerformance(schemeCode, period);

      if (result.success) {
        res.json({
          success: true,
          message: 'Scheme performance retrieved successfully',
          data: result.data
        });
      } else {
        res.status(404).json({
          success: false,
          message: 'Scheme not found',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('BSE Star MF scheme performance error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  async healthCheck(req, res) {
    try {
      logger.info('BSE Star MF health check request');

      const result = await this.service.healthCheck();

      if (result.success) {
        res.json({
          success: true,
          message: 'BSE Star MF service is healthy',
          data: result.data
        });
      } else {
        res.status(503).json({
          success: false,
          message: 'BSE Star MF service is unhealthy',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('BSE Star MF health check error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }
}

module.exports = new BSEStarMFController(); 