const bseStarMFService = require('../services/bseStarMFService');
const demoBSEStarMFService = require('../services/demoBSEStarMFService');
const logger = require('../utils/logger');
const appConfig = require('../config/app');
const { MfOrder } = require('../models');

const getIdempotencyKey = (req) => req.headers['x-idempotency-key'] || req.body?.idempotencyKey;

const buildOrderResponse = (order) => ({
  orderId: order.orderId,
  bseOrderId: order.bseOrderId,
  status: order.status,
  amount: order.amount,
  schemeCode: order.schemeCode,
  message: order.providerResponse?.message
});

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
      const idempotencyKey = getIdempotencyKey(req);

      logger.info('BSE Star MF lumpsum order request', { userId, orderData, idempotencyKey });

      if (idempotencyKey) {
        const existingOrder = await MfOrder.findOne({
          userId,
          orderType: 'LUMPSUM',
          idempotencyKey
        });

        if (existingOrder) {
          return res.status(200).json({
            success: true,
            message: 'Order already processed',
            data: existingOrder.providerResponse || buildOrderResponse(existingOrder)
          });
        }
      }

      const result = await this.service.placeLumpsumOrder({
        ...orderData,
        userId
      });

      const orderRecord = {
        userId,
        orderType: 'LUMPSUM',
        status: result.success ? (result.data.status || 'PENDING') : 'FAILED',
        idempotencyKey: idempotencyKey || undefined,
        amount: orderData.amount,
        schemeCode: orderData.schemeCode,
        clientId: orderData.clientId,
        orderId: result.success ? result.data.orderId : undefined,
        bseOrderId: result.success ? result.data.bseOrderId : undefined,
        providerResponse: result.success ? result.data : undefined,
        error: result.success ? undefined : result.error,
        source: 'BSE_STAR_MF'
      };

      try {
        await MfOrder.create(orderRecord);
      } catch (err) {
        if (err?.code === 11000 && idempotencyKey) {
          const existingOrder = await MfOrder.findOne({ userId, orderType: 'LUMPSUM', idempotencyKey });
          if (existingOrder) {
            return res.status(200).json({
              success: true,
              message: 'Order already processed',
              data: existingOrder.providerResponse || buildOrderResponse(existingOrder)
            });
          }
        }
        logger.error('Failed to persist MF order record', { error: err?.message, userId });
      }

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
        try {
          await MfOrder.findOneAndUpdate(
            { orderId },
            {
              status: result.data.status,
              bseOrderId: result.data.bseOrderId,
              providerResponse: result.data,
              lastCheckedAt: new Date()
            },
            { new: true }
          );
        } catch (err) {
          logger.warn('Failed to update MF order status', { error: err?.message, orderId });
        }

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
      const idempotencyKey = getIdempotencyKey(req);

      logger.info('BSE Star MF redemption order request', { userId, redemptionData, idempotencyKey });

      if (idempotencyKey) {
        const existingOrder = await MfOrder.findOne({
          userId,
          orderType: 'REDEMPTION',
          idempotencyKey
        });

        if (existingOrder) {
          return res.status(200).json({
            success: true,
            message: 'Redemption already processed',
            data: existingOrder.providerResponse || buildOrderResponse(existingOrder)
          });
        }
      }

      const result = await this.service.placeRedemptionOrder({
        ...redemptionData,
        userId
      });

      const orderRecord = {
        userId,
        orderType: 'REDEMPTION',
        status: result.success ? (result.data.status || 'PENDING') : 'FAILED',
        idempotencyKey: idempotencyKey || undefined,
        amount: redemptionData.amount,
        units: redemptionData.units,
        schemeCode: redemptionData.schemeCode,
        clientId: redemptionData.clientId,
        folioNumber: redemptionData.folioNumber,
        redemptionType: redemptionData.redemptionType,
        orderId: result.success ? result.data.redemptionId : undefined,
        bseOrderId: result.success ? result.data.bseRedemptionId : undefined,
        providerResponse: result.success ? result.data : undefined,
        error: result.success ? undefined : result.error,
        source: 'BSE_STAR_MF'
      };

      try {
        await MfOrder.create(orderRecord);
      } catch (err) {
        if (err?.code === 11000 && idempotencyKey) {
          const existingOrder = await MfOrder.findOne({ userId, orderType: 'REDEMPTION', idempotencyKey });
          if (existingOrder) {
            return res.status(200).json({
              success: true,
              message: 'Redemption already processed',
              data: existingOrder.providerResponse || buildOrderResponse(existingOrder)
            });
          }
        }
        logger.error('Failed to persist MF redemption record', { error: err?.message, userId });
      }

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
