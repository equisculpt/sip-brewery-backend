const axios = require('axios');
const logger = require('../utils/logger');
const crypto = require('crypto');

class DigioService {
  constructor() {
    this.baseUrl = process.env.DIGIO_BASE_URL || 'https://api.digio.in';
    this.clientId = process.env.DIGIO_CLIENT_ID;
    this.clientSecret = process.env.DIGIO_CLIENT_SECRET;
    this.apiKey = process.env.DIGIO_API_KEY;
    this.timeout = parseInt(process.env.DIGIO_TIMEOUT) || 30000;
    this.maxRetries = parseInt(process.env.DIGIO_MAX_RETRIES) || 3;
  }

  /**
   * Generate authentication headers for Digio API
   */
  generateAuthHeaders() {
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${this.apiKey}`,
      'X-Digio-ClientId': this.clientId,
      'X-Digio-Timestamp': Date.now().toString()
    };
  }

  /**
   * Make API request with retry logic
   */
  async makeRequest(endpoint, method = 'GET', data = null) {
    let lastError;
    
    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        const config = {
          method,
          url: `${this.baseUrl}${endpoint}`,
          headers: this.generateAuthHeaders(),
          timeout: this.timeout
        };

        if (data) {
          config.data = data;
        }

        logger.info('Digio API request', {
          endpoint,
          method,
          attempt,
          hasData: !!data
        });

        const response = await axios(config);
        
        logger.info('Digio API response', {
          endpoint,
          status: response.status,
          success: response.data.success
        });

        return response.data;
      } catch (error) {
        lastError = error;
        logger.error('Digio API request failed', {
          endpoint,
          attempt,
          error: error.message,
          status: error.response?.status
        });

        if (attempt === this.maxRetries) {
          throw new Error(`Digio API request failed after ${this.maxRetries} attempts: ${error.message}`);
        }

        // Wait before retry (exponential backoff)
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
      }
    }
  }

  /**
   * 1. KYC Verification API
   */
  async initiateKYC(kycData) {
    try {
      const payload = {
        requestId: `KYC_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        customerDetails: {
          name: kycData.name,
          dateOfBirth: kycData.dateOfBirth,
          gender: kycData.gender,
          panNumber: kycData.panNumber,
          aadhaarNumber: kycData.aadhaarNumber,
          mobile: kycData.mobile,
          email: kycData.email,
          address: {
            line1: kycData.address.line1,
            line2: kycData.address.line2 || '',
            city: kycData.address.city,
            state: kycData.address.state,
            pincode: kycData.address.pincode,
            country: 'INDIA'
          }
        },
        kycType: kycData.kycType || 'AADHAAR_BASED', // AADHAAR_BASED, PAN_BASED, OFFLINE
        callbackUrl: process.env.DIGIO_CALLBACK_URL,
        redirectUrl: process.env.DIGIO_REDIRECT_URL,
        expiryTime: 24 * 60 * 60 // 24 hours in seconds
      };

      const response = await this.makeRequest('/api/v1/kyc/initiate', 'POST', payload);
      
      return {
        success: true,
        data: {
          kycId: response.kycId,
          requestId: response.requestId,
          status: response.status,
          kycUrl: response.kycUrl,
          expiresAt: response.expiresAt,
          message: response.message
        }
      };
    } catch (error) {
      logger.error('Digio KYC initiation failed', { error: error.message, kycData });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get KYC status
   */
  async getKYCStatus(kycId) {
    try {
      const response = await this.makeRequest(`/api/v1/kyc/status/${kycId}`);
      
      return {
        success: true,
        data: {
          kycId: response.kycId,
          requestId: response.requestId,
          status: response.status, // PENDING, COMPLETED, FAILED, EXPIRED
          submittedAt: response.submittedAt,
          completedAt: response.completedAt,
          verificationDetails: response.verificationDetails || {},
          documents: response.documents || [],
          message: response.message,
          failureReason: response.failureReason
        }
      };
    } catch (error) {
      logger.error('Digio KYC status fetch failed', { error: error.message, kycId });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Download KYC documents
   */
  async downloadKYCDocuments(kycId, documentType = 'ALL') {
    try {
      const response = await this.makeRequest(`/api/v1/kyc/${kycId}/documents?type=${documentType}`);
      
      return {
        success: true,
        data: {
          kycId: response.kycId,
          documents: response.documents || [],
          downloadUrls: response.downloadUrls || {}
        }
      };
    } catch (error) {
      logger.error('Digio KYC documents download failed', { error: error.message, kycId, documentType });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * 2. eMandate Setup API (via NPCI/NACH)
   */
  async setupEMandate(mandateData) {
    try {
      const payload = {
        requestId: `MANDATE_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        customerDetails: {
          name: mandateData.customerName,
          mobile: mandateData.mobile,
          email: mandateData.email,
          panNumber: mandateData.panNumber
        },
        bankDetails: {
          accountNumber: mandateData.accountNumber,
          ifscCode: mandateData.ifscCode,
          accountHolderName: mandateData.accountHolderName
        },
        mandateDetails: {
          amount: mandateData.amount,
          frequency: mandateData.frequency, // MONTHLY, WEEKLY, QUARTERLY
          startDate: mandateData.startDate,
          endDate: mandateData.endDate,
          purpose: mandateData.purpose || 'MUTUAL_FUND_INVESTMENT',
          merchantId: process.env.DIGIO_MERCHANT_ID,
          subMerchantId: process.env.DIGIO_SUB_MERCHANT_ID
        },
        callbackUrl: process.env.DIGIO_MANDATE_CALLBACK_URL,
        redirectUrl: process.env.DIGIO_MANDATE_REDIRECT_URL,
        expiryTime: 24 * 60 * 60 // 24 hours in seconds
      };

      const response = await this.makeRequest('/api/v1/emandate/setup', 'POST', payload);
      
      return {
        success: true,
        data: {
          mandateId: response.mandateId,
          requestId: response.requestId,
          status: response.status,
          mandateUrl: response.mandateUrl,
          npcUrl: response.npcUrl,
          expiresAt: response.expiresAt,
          message: response.message
        }
      };
    } catch (error) {
      logger.error('Digio eMandate setup failed', { error: error.message, mandateData });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get eMandate status
   */
  async getEMandateStatus(mandateId) {
    try {
      const response = await this.makeRequest(`/api/v1/emandate/status/${mandateId}`);
      
      return {
        success: true,
        data: {
          mandateId: response.mandateId,
          requestId: response.requestId,
          status: response.status, // PENDING, ACTIVE, REJECTED, EXPIRED
          customerDetails: response.customerDetails || {},
          bankDetails: response.bankDetails || {},
          mandateDetails: response.mandateDetails || {},
          submittedAt: response.submittedAt,
          activatedAt: response.activatedAt,
          message: response.message,
          failureReason: response.failureReason
        }
      };
    } catch (error) {
      logger.error('Digio eMandate status fetch failed', { error: error.message, mandateId });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Cancel eMandate
   */
  async cancelEMandate(mandateId, reason = 'USER_REQUESTED') {
    try {
      const payload = {
        mandateId,
        reason,
        cancellationDate: new Date().toISOString().split('T')[0]
      };

      const response = await this.makeRequest('/api/v1/emandate/cancel', 'POST', payload);
      
      return {
        success: true,
        data: {
          mandateId: response.mandateId,
          status: response.status,
          cancellationDate: response.cancellationDate,
          message: response.message
        }
      };
    } catch (error) {
      logger.error('Digio eMandate cancellation failed', { error: error.message, mandateId });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * 3. PAN Check + CKYC Pull
   */
  async verifyPAN(panNumber) {
    try {
      const payload = {
        panNumber,
        requestId: `PAN_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      };

      const response = await this.makeRequest('/api/v1/pan/verify', 'POST', payload);
      
      return {
        success: true,
        data: {
          panNumber: response.panNumber,
          requestId: response.requestId,
          status: response.status, // VALID, INVALID, NOT_FOUND
          name: response.name,
          dateOfBirth: response.dateOfBirth,
          gender: response.gender,
          message: response.message
        }
      };
    } catch (error) {
      logger.error('Digio PAN verification failed', { error: error.message, panNumber });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Pull CKYC data
   */
  async pullCKYC(ckycData) {
    try {
      const payload = {
        requestId: `CKYC_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        panNumber: ckycData.panNumber,
        aadhaarNumber: ckycData.aadhaarNumber,
        mobile: ckycData.mobile,
        email: ckycData.email
      };

      const response = await this.makeRequest('/api/v1/ckyc/pull', 'POST', payload);
      
      return {
        success: true,
        data: {
          ckycNumber: response.ckycNumber,
          requestId: response.requestId,
          status: response.status, // FOUND, NOT_FOUND, ERROR
          customerDetails: response.customerDetails || {},
          kycDetails: response.kycDetails || {},
          message: response.message
        }
      };
    } catch (error) {
      logger.error('Digio CKYC pull failed', { error: error.message, ckycData });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * 4. eSign flow for agreements
   */
  async initiateESign(esignData) {
    try {
      const payload = {
        requestId: `ESIGN_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        customerDetails: {
          name: esignData.customerName,
          mobile: esignData.mobile,
          email: esignData.email,
          panNumber: esignData.panNumber,
          aadhaarNumber: esignData.aadhaarNumber
        },
        documentDetails: {
          title: esignData.documentTitle,
          description: esignData.documentDescription,
          documentUrl: esignData.documentUrl,
          documentType: esignData.documentType || 'AGREEMENT'
        },
        signDetails: {
          signType: esignData.signType || 'AADHAAR_BASED', // AADHAAR_BASED, PAN_BASED, DSC
          signLocation: esignData.signLocation || 'BOTTOM_RIGHT',
          signReason: esignData.signReason || 'AGREEMENT_SIGNING'
        },
        callbackUrl: process.env.DIGIO_ESIGN_CALLBACK_URL,
        redirectUrl: process.env.DIGIO_ESIGN_REDIRECT_URL,
        expiryTime: 24 * 60 * 60 // 24 hours in seconds
      };

      const response = await this.makeRequest('/api/v1/esign/initiate', 'POST', payload);
      
      return {
        success: true,
        data: {
          esignId: response.esignId,
          requestId: response.requestId,
          status: response.status,
          esignUrl: response.esignUrl,
          expiresAt: response.expiresAt,
          message: response.message
        }
      };
    } catch (error) {
      logger.error('Digio eSign initiation failed', { error: error.message, esignData });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get eSign status
   */
  async getESignStatus(esignId) {
    try {
      const response = await this.makeRequest(`/api/v1/esign/status/${esignId}`);
      
      return {
        success: true,
        data: {
          esignId: response.esignId,
          requestId: response.requestId,
          status: response.status, // PENDING, COMPLETED, FAILED, EXPIRED
          customerDetails: response.customerDetails || {},
          documentDetails: response.documentDetails || {},
          signDetails: response.signDetails || {},
          submittedAt: response.submittedAt,
          completedAt: response.completedAt,
          signedDocumentUrl: response.signedDocumentUrl,
          message: response.message,
          failureReason: response.failureReason
        }
      };
    } catch (error) {
      logger.error('Digio eSign status fetch failed', { error: error.message, esignId });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Download signed document
   */
  async downloadSignedDocument(esignId) {
    try {
      const response = await this.makeRequest(`/api/v1/esign/${esignId}/download`);
      
      return {
        success: true,
        data: {
          esignId: response.esignId,
          documentUrl: response.documentUrl,
          downloadUrl: response.downloadUrl,
          documentHash: response.documentHash,
          signedAt: response.signedAt
        }
      };
    } catch (error) {
      logger.error('Digio signed document download failed', { error: error.message, esignId });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Verify document signature
   */
  async verifyDocumentSignature(documentHash) {
    try {
      const payload = {
        documentHash,
        requestId: `VERIFY_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      };

      const response = await this.makeRequest('/api/v1/esign/verify', 'POST', payload);
      
      return {
        success: true,
        data: {
          documentHash: response.documentHash,
          requestId: response.requestId,
          isValid: response.isValid,
          signatureDetails: response.signatureDetails || {},
          verificationDate: response.verificationDate,
          message: response.message
        }
      };
    } catch (error) {
      logger.error('Digio document signature verification failed', { error: error.message, documentHash });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get customer consent history
   */
  async getConsentHistory(customerId) {
    try {
      const response = await this.makeRequest(`/api/v1/consent/history/${customerId}`);
      
      return {
        success: true,
        data: {
          customerId: response.customerId,
          consents: response.consents || [],
          totalConsents: response.totalConsents || 0
        }
      };
    } catch (error) {
      logger.error('Digio consent history fetch failed', { error: error.message, customerId });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Revoke customer consent
   */
  async revokeConsent(consentId, reason = 'USER_REQUESTED') {
    try {
      const payload = {
        consentId,
        reason,
        revocationDate: new Date().toISOString().split('T')[0]
      };

      const response = await this.makeRequest('/api/v1/consent/revoke', 'POST', payload);
      
      return {
        success: true,
        data: {
          consentId: response.consentId,
          status: response.status,
          revocationDate: response.revocationDate,
          message: response.message
        }
      };
    } catch (error) {
      logger.error('Digio consent revocation failed', { error: error.message, consentId });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Health check for Digio API
   */
  async healthCheck() {
    try {
      const response = await this.makeRequest('/api/v1/health');
      
      return {
        success: true,
        data: {
          status: response.status,
          timestamp: response.timestamp,
          version: response.version,
          services: response.services || {}
        }
      };
    } catch (error) {
      logger.error('Digio health check failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get API usage statistics
   */
  async getUsageStats(startDate, endDate) {
    try {
      const queryParams = new URLSearchParams({
        startDate,
        endDate
      });

      const response = await this.makeRequest(`/api/v1/stats/usage?${queryParams.toString()}`);
      
      return {
        success: true,
        data: {
          period: { startDate, endDate },
          kycRequests: response.kycRequests || 0,
          mandateRequests: response.mandateRequests || 0,
          esignRequests: response.esignRequests || 0,
          panVerifications: response.panVerifications || 0,
          ckycPulls: response.ckycPulls || 0,
          totalRequests: response.totalRequests || 0,
          successRate: response.successRate || 0
        }
      };
    } catch (error) {
      logger.error('Digio usage stats fetch failed', { error: error.message, startDate, endDate });
      return {
        success: false,
        error: error.message
      };
    }
  }
}

module.exports = new DigioService(); 