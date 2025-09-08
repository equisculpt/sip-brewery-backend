const axios = require('axios');
const logger = require('../utils/logger');
const crypto = require('crypto');

class BSEStarMFService {
  constructor() {
    this.baseUrl = process.env.BSE_STAR_MF_BASE_URL || 'https://api.bseindia.com';
    this.clientId = process.env.BSE_STAR_MF_CLIENT_ID;
    this.clientSecret = process.env.BSE_STAR_MF_CLIENT_SECRET;
    this.apiKey = process.env.BSE_STAR_MF_API_KEY;
    this.secretKey = process.env.BSE_STAR_MF_SECRET_KEY;
    this.timeout = parseInt(process.env.BSE_STAR_MF_TIMEOUT) || 30000;
    this.maxRetries = parseInt(process.env.BSE_STAR_MF_MAX_RETRIES) || 3;
  }

  /**
   * Generate authentication headers for BSE Star MF API
   */
  generateAuthHeaders() {
    const timestamp = Math.floor(Date.now() / 1000);
    const signature = this.generateSignature(timestamp);
    
    return {
      'Content-Type': 'application/json',
      'X-BSE-ClientId': this.clientId,
      'X-BSE-Timestamp': timestamp.toString(),
      'X-BSE-Signature': signature,
      'X-BSE-ApiKey': this.apiKey
    };
  }

  /**
   * Generate signature for API authentication
   */
  generateSignature(timestamp) {
    const message = `${this.clientId}${timestamp}`;
    const signature = crypto
      .createHmac('sha256', this.secretKey)
      .update(message)
      .digest('hex');
    return signature;
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

        logger.info('BSE Star MF API request', {
          endpoint,
          method,
          attempt,
          hasData: !!data
        });

        const response = await axios(config);
        
        logger.info('BSE Star MF API response', {
          endpoint,
          status: response.status,
          success: response.data.success
        });

        return response.data;
      } catch (error) {
        lastError = error;
        logger.error('BSE Star MF API request failed', {
          endpoint,
          attempt,
          error: error.message,
          status: error.response?.status
        });

        if (attempt === this.maxRetries) {
          throw new Error(`BSE Star MF API request failed after ${this.maxRetries} attempts: ${error.message}`);
        }

        // Wait before retry (exponential backoff)
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
      }
    }
  }

  /**
   * 1. Client Creation API (AddClient/ModifyClient)
   */
  async createClient(clientData) {
    try {
      const payload = {
        clientType: 'INDIVIDUAL',
        firstName: clientData.firstName,
        lastName: clientData.lastName,
        dateOfBirth: clientData.dateOfBirth,
        gender: clientData.gender,
        panNumber: clientData.panNumber,
        aadhaarNumber: clientData.aadhaarNumber,
        email: clientData.email,
        mobile: clientData.mobile,
        address: {
          line1: clientData.address.line1,
          line2: clientData.address.line2 || '',
          city: clientData.address.city,
          state: clientData.address.state,
          pincode: clientData.address.pincode,
          country: 'INDIA'
        },
        bankDetails: {
          accountNumber: clientData.bankDetails.accountNumber,
          ifscCode: clientData.bankDetails.ifscCode,
          accountHolderName: clientData.bankDetails.accountHolderName
        },
        nomineeDetails: clientData.nomineeDetails || [],
        kycDetails: {
          kycStatus: clientData.kycStatus || 'PENDING',
          kycNumber: clientData.kycNumber || '',
          kycDate: clientData.kycDate || ''
        }
      };

      const response = await this.makeRequest('/api/v1/client/create', 'POST', payload);
      
      return {
        success: true,
        data: {
          clientId: response.clientId,
          bseClientId: response.bseClientId,
          status: response.status,
          message: response.message
        }
      };
    } catch (error) {
      logger.error('BSE Star MF client creation failed', { error: error.message, clientData });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Modify existing client
   */
  async modifyClient(clientId, clientData) {
    try {
      const payload = {
        clientId,
        ...clientData
      };

      const response = await this.makeRequest('/api/v1/client/modify', 'PUT', payload);
      
      return {
        success: true,
        data: {
          clientId: response.clientId,
          status: response.status,
          message: response.message
        }
      };
    } catch (error) {
      logger.error('BSE Star MF client modification failed', { error: error.message, clientId });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * 2. Scheme Master Data API
   */
  async getSchemeMasterData(filters = {}) {
    try {
      const queryParams = new URLSearchParams();
      
      if (filters.category) queryParams.append('category', filters.category);
      if (filters.fundHouse) queryParams.append('fundHouse', filters.fundHouse);
      if (filters.isActive !== undefined) queryParams.append('isActive', filters.isActive);
      if (filters.limit) queryParams.append('limit', filters.limit);
      if (filters.offset) queryParams.append('offset', filters.offset);

      const endpoint = `/api/v1/scheme/master?${queryParams.toString()}`;
      const response = await this.makeRequest(endpoint);
      
      return {
        success: true,
        data: {
          schemes: response.schemes || [],
          totalCount: response.totalCount || 0,
          pagination: response.pagination || {}
        }
      };
    } catch (error) {
      logger.error('BSE Star MF scheme master data fetch failed', { error: error.message, filters });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get specific scheme details
   */
  async getSchemeDetails(schemeCode) {
    try {
      const response = await this.makeRequest(`/api/v1/scheme/${schemeCode}`);
      
      return {
        success: true,
        data: response
      };
    } catch (error) {
      logger.error('BSE Star MF scheme details fetch failed', { error: error.message, schemeCode });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * 3. Lumpsum Order Placement API (used for Smart SIP too)
   */
  async placeLumpsumOrder(orderData) {
    try {
      const payload = {
        clientId: orderData.clientId,
        schemeCode: orderData.schemeCode,
        amount: orderData.amount,
        orderType: 'PURCHASE', // PURCHASE, REDEMPTION, SWITCH
        orderCategory: 'LUMPSUM',
        paymentMode: orderData.paymentMode || 'ONLINE', // ONLINE, CHEQUE, DD
        bankAccount: orderData.bankAccount,
        nomineeDetails: orderData.nomineeDetails || [],
        folioNumber: orderData.folioNumber || '',
        isSmartSIP: orderData.isSmartSIP || false,
        smartSIPDetails: orderData.smartSIPDetails || null
      };

      const response = await this.makeRequest('/api/v1/order/lumpsum', 'POST', payload);
      
      return {
        success: true,
        data: {
          orderId: response.orderId,
          bseOrderId: response.bseOrderId,
          status: response.status,
          orderDate: response.orderDate,
          estimatedUnits: response.estimatedUnits,
          nav: response.nav,
          message: response.message
        }
      };
    } catch (error) {
      logger.error('BSE Star MF lumpsum order placement failed', { error: error.message, orderData });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * 4. Order Status API
   */
  async getOrderStatus(orderId) {
    try {
      const response = await this.makeRequest(`/api/v1/order/status/${orderId}`);
      
      return {
        success: true,
        data: {
          orderId: response.orderId,
          bseOrderId: response.bseOrderId,
          status: response.status, // PENDING, PROCESSING, COMPLETED, FAILED, CANCELLED
          orderDate: response.orderDate,
          completionDate: response.completionDate,
          amount: response.amount,
          units: response.units,
          nav: response.nav,
          charges: response.charges || {},
          message: response.message,
          failureReason: response.failureReason
        }
      };
    } catch (error) {
      logger.error('BSE Star MF order status fetch failed', { error: error.message, orderId });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * 5. Redemption API
   */
  async placeRedemptionOrder(redemptionData) {
    try {
      const payload = {
        clientId: redemptionData.clientId,
        schemeCode: redemptionData.schemeCode,
        folioNumber: redemptionData.folioNumber,
        redemptionType: redemptionData.redemptionType, // UNITS, AMOUNT
        units: redemptionData.units || 0,
        amount: redemptionData.amount || 0,
        bankAccount: redemptionData.bankAccount,
        nomineeDetails: redemptionData.nomineeDetails || [],
        redemptionMode: redemptionData.redemptionMode || 'NORMAL' // NORMAL, SWITCH
      };

      const response = await this.makeRequest('/api/v1/order/redemption', 'POST', payload);
      
      return {
        success: true,
        data: {
          redemptionId: response.redemptionId,
          bseRedemptionId: response.bseRedemptionId,
          status: response.status,
          redemptionDate: response.redemptionDate,
          units: response.units,
          estimatedAmount: response.estimatedAmount,
          nav: response.nav,
          message: response.message
        }
      };
    } catch (error) {
      logger.error('BSE Star MF redemption order placement failed', { error: error.message, redemptionData });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * 6. Transaction Report API
   */
  async getTransactionReport(filters = {}) {
    try {
      const queryParams = new URLSearchParams();
      
      if (filters.clientId) queryParams.append('clientId', filters.clientId);
      if (filters.schemeCode) queryParams.append('schemeCode', filters.schemeCode);
      if (filters.folioNumber) queryParams.append('folioNumber', filters.folioNumber);
      if (filters.startDate) queryParams.append('startDate', filters.startDate);
      if (filters.endDate) queryParams.append('endDate', filters.endDate);
      if (filters.orderType) queryParams.append('orderType', filters.orderType);
      if (filters.limit) queryParams.append('limit', filters.limit);
      if (filters.offset) queryParams.append('offset', filters.offset);

      const endpoint = `/api/v1/report/transactions?${queryParams.toString()}`;
      const response = await this.makeRequest(endpoint);
      
      return {
        success: true,
        data: {
          transactions: response.transactions || [],
          totalCount: response.totalCount || 0,
          pagination: response.pagination || {}
        }
      };
    } catch (error) {
      logger.error('BSE Star MF transaction report fetch failed', { error: error.message, filters });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * 7. NAV & Holding Report API
   */
  async getNAVAndHoldingReport(filters = {}) {
    try {
      const queryParams = new URLSearchParams();
      
      if (filters.clientId) queryParams.append('clientId', filters.clientId);
      if (filters.schemeCode) queryParams.append('schemeCode', filters.schemeCode);
      if (filters.folioNumber) queryParams.append('folioNumber', filters.folioNumber);
      if (filters.date) queryParams.append('date', filters.date);

      const endpoint = `/api/v1/report/holdings?${queryParams.toString()}`;
      const response = await this.makeRequest(endpoint);
      
      return {
        success: true,
        data: {
          holdings: response.holdings || [],
          totalValue: response.totalValue || 0,
          totalUnits: response.totalUnits || 0,
          reportDate: response.reportDate,
          navData: response.navData || {}
        }
      };
    } catch (error) {
      logger.error('BSE Star MF NAV and holding report fetch failed', { error: error.message, filters });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get current NAV for schemes
   */
  async getCurrentNAV(schemeCodes) {
    try {
      const payload = {
        schemeCodes: Array.isArray(schemeCodes) ? schemeCodes : [schemeCodes]
      };

      const response = await this.makeRequest('/api/v1/nav/current', 'POST', payload);
      
      return {
        success: true,
        data: {
          navData: response.navData || {},
          lastUpdated: response.lastUpdated
        }
      };
    } catch (error) {
      logger.error('BSE Star MF current NAV fetch failed', { error: error.message, schemeCodes });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * 8. eMandate via BSE (required for auto-debit)
   */
  async setupEMandate(mandateData) {
    try {
      const payload = {
        clientId: mandateData.clientId,
        bankAccount: mandateData.bankAccount,
        amount: mandateData.amount,
        frequency: mandateData.frequency, // MONTHLY, WEEKLY, QUARTERLY
        startDate: mandateData.startDate,
        endDate: mandateData.endDate,
        purpose: 'MUTUAL_FUND_INVESTMENT',
        merchantId: process.env.BSE_MERCHANT_ID,
        subMerchantId: process.env.BSE_SUB_MERCHANT_ID
      };

      const response = await this.makeRequest('/api/v1/emandate/setup', 'POST', payload);
      
      return {
        success: true,
        data: {
          mandateId: response.mandateId,
          bseMandateId: response.bseMandateId,
          status: response.status,
          mandateUrl: response.mandateUrl,
          expiresAt: response.expiresAt,
          message: response.message
        }
      };
    } catch (error) {
      logger.error('BSE Star MF eMandate setup failed', { error: error.message, mandateData });
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
          status: response.status, // PENDING, ACTIVE, REJECTED, EXPIRED
          bankAccount: response.bankAccount,
          amount: response.amount,
          frequency: response.frequency,
          startDate: response.startDate,
          endDate: response.endDate,
          lastDebitDate: response.lastDebitDate,
          nextDebitDate: response.nextDebitDate,
          message: response.message
        }
      };
    } catch (error) {
      logger.error('BSE Star MF eMandate status fetch failed', { error: error.message, mandateId });
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
      logger.error('BSE Star MF eMandate cancellation failed', { error: error.message, mandateId });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get client folios
   */
  async getClientFolios(clientId) {
    try {
      const response = await this.makeRequest(`/api/v1/client/${clientId}/folios`);
      
      return {
        success: true,
        data: {
          folios: response.folios || [],
          totalFolios: response.totalFolios || 0
        }
      };
    } catch (error) {
      logger.error('BSE Star MF client folios fetch failed', { error: error.message, clientId });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get scheme performance data
   */
  async getSchemePerformance(schemeCode, period = '1Y') {
    try {
      const response = await this.makeRequest(`/api/v1/scheme/${schemeCode}/performance?period=${period}`);
      
      return {
        success: true,
        data: {
          schemeCode,
          period,
          performance: response.performance || {},
          benchmark: response.benchmark || {},
          riskMetrics: response.riskMetrics || {}
        }
      };
    } catch (error) {
      logger.error('BSE Star MF scheme performance fetch failed', { error: error.message, schemeCode, period });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Health check for BSE Star MF API
   */
  async healthCheck() {
    try {
      const response = await this.makeRequest('/api/v1/health');
      
      return {
        success: true,
        data: {
          status: response.status,
          timestamp: response.timestamp,
          version: response.version
        }
      };
    } catch (error) {
      logger.error('BSE Star MF health check failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }
}

module.exports = new BSEStarMFService(); 