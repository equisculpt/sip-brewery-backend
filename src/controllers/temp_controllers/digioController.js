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
          message: 'KYC status retrieved successfully',
          data: result.data
        });
      } else {
        res.status(404).json({
          success: false,
          message: 'KYC request not found',
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

  async downloadKYCDocuments(req, res) {
    try {
      const { kycId } = req.params;
      const { type } = req.query;
      const userId = req.user.id;

      logger.info('Digio KYC documents download request', { userId, kycId, type });

      const result = await this.service.downloadKYCDocuments(kycId, type);

      if (result.success) {
        res.json({
          success: true,
          message: 'KYC documents retrieved successfully',
          data: result.data
        });
      } else {
        res.status(404).json({
          success: false,
          message: 'KYC documents not found',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Digio KYC documents download error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * 2. eMandate Setup API (via NPCI/NACH)
   */
  async setupEMandate(req, res) {
    try {
      const { mandateData } = req.body;
      const userId = req.user.id;

      logger.info('Digio eMandate setup request', { userId, mandateData });

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
      logger.error('Digio eMandate setup error', { error: error.message });
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

      logger.info('Digio eMandate status request', { userId, mandateId });

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
      logger.error('Digio eMandate status error', { error: error.message });
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

      logger.info('Digio eMandate cancellation request', { userId, mandateId, reason });

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
      logger.error('Digio eMandate cancellation error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * 3. PAN Check + CKYC Pull
   */
  async verifyPAN(req, res) {
    try {
      const { panNumber } = req.body;
      const userId = req.user.id;

      logger.info('Digio PAN verification request', { userId, panNumber });

      const result = await this.service.verifyPAN(panNumber);

      if (result.success) {
        res.json({
          success: true,
          message: 'PAN verification completed',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'PAN verification failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Digio PAN verification error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  async pullCKYC(req, res) {
    try {
      const { ckycData } = req.body;
      const userId = req.user.id;

      logger.info('Digio CKYC pull request', { userId, ckycData });

      const result = await this.service.pullCKYC({
        ...ckycData,
        userId
      });

      if (result.success) {
        res.json({
          success: true,
          message: 'CKYC data retrieved successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'CKYC pull failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Digio CKYC pull error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * 4. eSign flow for agreements
   */
  async initiateESign(req, res) {
    try {
      const { esignData } = req.body;
      const userId = req.user.id;

      logger.info('Digio eSign initiation request', { userId, esignData });

      const result = await this.service.initiateESign({
        ...esignData,
        userId
      });

      if (result.success) {
        res.status(201).json({
          success: true,
          message: 'eSign initiated successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'eSign initiation failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Digio eSign initiation error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  async getESignStatus(req, res) {
    try {
      const { esignId } = req.params;
      const userId = req.user.id;

      logger.info('Digio eSign status request', { userId, esignId });

      const result = await this.service.getESignStatus(esignId);

      if (result.success) {
        res.json({
          success: true,
          message: 'eSign status retrieved successfully',
          data: result.data
        });
      } else {
        res.status(404).json({
          success: false,
          message: 'eSign request not found',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Digio eSign status error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  async downloadSignedDocument(req, res) {
    try {
      const { esignId } = req.params;
      const userId = req.user.id;

      logger.info('Digio signed document download request', { userId, esignId });

      const result = await this.service.downloadSignedDocument(esignId);

      if (result.success) {
        res.json({
          success: true,
          message: 'Signed document retrieved successfully',
          data: result.data
        });
      } else {
        res.status(404).json({
          success: false,
          message: 'Signed document not found',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Digio signed document download error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  async verifyDocumentSignature(req, res) {
    try {
      const { documentHash } = req.body;
      const userId = req.user.id;

      logger.info('Digio document signature verification request', { userId, documentHash });

      const result = await this.service.verifyDocumentSignature(documentHash);

      if (result.success) {
        res.json({
          success: true,
          message: 'Document signature verification completed',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'Document signature verification failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Digio document signature verification error', { error: error.message });
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
  async getConsentHistory(req, res) {
    try {
      const { customerId } = req.params;
      const userId = req.user.id;

      logger.info('Digio consent history request', { userId, customerId });

      const result = await this.service.getConsentHistory(customerId);

      if (result.success) {
        res.json({
          success: true,
          message: 'Consent history retrieved successfully',
          data: result.data
        });
      } else {
        res.status(404).json({
          success: false,
          message: 'Customer not found',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Digio consent history error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  async revokeConsent(req, res) {
    try {
      const { consentId } = req.params;
      const { reason } = req.body;
      const userId = req.user.id;

      logger.info('Digio consent revocation request', { userId, consentId, reason });

      const result = await this.service.revokeConsent(consentId, reason);

      if (result.success) {
        res.json({
          success: true,
          message: 'Consent revoked successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'Consent revocation failed',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Digio consent revocation error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  async healthCheck(req, res) {
    try {
      logger.info('Digio health check request');

      const result = await this.service.healthCheck();

      if (result.success) {
        res.json({
          success: true,
          message: 'Digio service is healthy',
          data: result.data
        });
      } else {
        res.status(503).json({
          success: false,
          message: 'Digio service is unhealthy',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Digio health check error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  async getUsageStats(req, res) {
    try {
      const { startDate, endDate } = req.query;
      const userId = req.user.id;

      logger.info('Digio usage stats request', { userId, startDate, endDate });

      const result = await this.service.getUsageStats(startDate, endDate);

      if (result.success) {
        res.json({
          success: true,
          message: 'Usage statistics retrieved successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: 'Failed to retrieve usage statistics',
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Digio usage stats error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Webhook endpoints for callbacks
   */
  async kycCallback(req, res) {
    try {
      const { kycId, status, data } = req.body;

      logger.info('Digio KYC callback received', { kycId, status, data });

      // Update KYC status in database
      // This would typically update the user's KYC status in your database

      res.json({
        success: true,
        message: 'KYC callback processed successfully'
      });
    } catch (error) {
      logger.error('Digio KYC callback error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Callback processing failed',
        error: error.message
      });
    }
  }

  async mandateCallback(req, res) {
    try {
      const { mandateId, status, data } = req.body;

      logger.info('Digio mandate callback received', { mandateId, status, data });

      // Update mandate status in database
      // This would typically update the mandate status in your database

      res.json({
        success: true,
        message: 'Mandate callback processed successfully'
      });
    } catch (error) {
      logger.error('Digio mandate callback error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Callback processing failed',
        error: error.message
      });
    }
  }

  async esignCallback(req, res) {
    try {
      const { esignId, status, data } = req.body;

      logger.info('Digio eSign callback received', { esignId, status, data });

      // Update eSign status in database
      // This would typically update the eSign status in your database

      res.json({
        success: true,
        message: 'eSign callback processed successfully'
      });
    } catch (error) {
      logger.error('Digio eSign callback error', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Callback processing failed',
        error: error.message
      });
    }
  }
}

module.exports = new DigioController(); 