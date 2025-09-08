const logger = require('../utils/logger');

class DemoBSEStarMFService {
  constructor() {
    this.clients = new Map();
    this.orders = new Map();
    this.schemes = this.generateDemoSchemes();
    this.mandates = new Map();
    this.transactions = new Map();
    this.holdings = new Map();
  }

  /**
   * Generate demo scheme data
   */
  generateDemoSchemes() {
    const schemes = [
      {
        schemeCode: 'HDFCMIDCAP',
        schemeName: 'HDFC Mid-Cap Opportunities Fund',
        fundHouse: 'HDFC Mutual Fund',
        category: 'Equity',
        subCategory: 'Mid Cap',
        nav: 45.67,
        navDate: new Date().toISOString().split('T')[0],
        minInvestment: 5000,
        riskLevel: 'moderate',
        rating: 4.5,
        isActive: true
      },
      {
        schemeCode: 'ICICIBLUECHIP',
        schemeName: 'ICICI Prudential Bluechip Fund',
        fundHouse: 'ICICI Prudential Mutual Fund',
        category: 'Equity',
        subCategory: 'Large Cap',
        nav: 52.34,
        navDate: new Date().toISOString().split('T')[0],
        minInvestment: 5000,
        riskLevel: 'moderate',
        rating: 4.2,
        isActive: true
      },
      {
        schemeCode: 'SBIGOLD',
        schemeName: 'SBI Gold ETF',
        fundHouse: 'SBI Mutual Fund',
        category: 'Gold',
        subCategory: 'ETF',
        nav: 58.90,
        navDate: new Date().toISOString().split('T')[0],
        minInvestment: 1000,
        riskLevel: 'low',
        rating: 3.8,
        isActive: true
      },
      {
        schemeCode: 'AXISDEBT',
        schemeName: 'Axis Corporate Debt Fund',
        fundHouse: 'Axis Mutual Fund',
        category: 'Debt',
        subCategory: 'Corporate Bond',
        nav: 12.45,
        navDate: new Date().toISOString().split('T')[0],
        minInvestment: 5000,
        riskLevel: 'low',
        rating: 4.0,
        isActive: true
      }
    ];

    return new Map(schemes.map(scheme => [scheme.schemeCode, scheme]));
  }

  /**
   * 1. Client Creation API (AddClient/ModifyClient)
   */
  async createClient(clientData) {
    try {
      const clientId = `CLIENT_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const bseClientId = `BSE_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      const client = {
        clientId,
        bseClientId,
        ...clientData,
        status: 'ACTIVE',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };

      this.clients.set(clientId, client);

      logger.info('Demo BSE Star MF client created', { clientId, bseClientId });

      return {
        success: true,
        data: {
          clientId,
          bseClientId,
          status: 'ACTIVE',
          message: 'Client created successfully'
        }
      };
    } catch (error) {
      logger.error('Demo BSE Star MF client creation failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async modifyClient(clientId, clientData) {
    try {
      const client = this.clients.get(clientId);
      if (!client) {
        return {
          success: false,
          error: 'Client not found'
        };
      }

      const updatedClient = {
        ...client,
        ...clientData,
        updatedAt: new Date().toISOString()
      };

      this.clients.set(clientId, updatedClient);

      logger.info('Demo BSE Star MF client modified', { clientId });

      return {
        success: true,
        data: {
          clientId,
          status: 'ACTIVE',
          message: 'Client modified successfully'
        }
      };
    } catch (error) {
      logger.error('Demo BSE Star MF client modification failed', { error: error.message });
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
      let schemes = Array.from(this.schemes.values());

      // Apply filters
      if (filters.category) {
        schemes = schemes.filter(scheme => scheme.category.toLowerCase() === filters.category.toLowerCase());
      }
      if (filters.fundHouse) {
        schemes = schemes.filter(scheme => scheme.fundHouse.toLowerCase().includes(filters.fundHouse.toLowerCase()));
      }
      if (filters.isActive !== undefined) {
        schemes = schemes.filter(scheme => scheme.isActive === filters.isActive);
      }

      // Apply pagination
      const limit = filters.limit || 20;
      const offset = filters.offset || 0;
      const paginatedSchemes = schemes.slice(offset, offset + limit);

      logger.info('Demo BSE Star MF scheme master data fetched', { 
        totalSchemes: schemes.length,
        returnedSchemes: paginatedSchemes.length 
      });

      return {
        success: true,
        data: {
          schemes: paginatedSchemes,
          totalCount: schemes.length,
          pagination: {
            page: Math.floor(offset / limit) + 1,
            limit,
            total: schemes.length,
            pages: Math.ceil(schemes.length / limit)
          }
        }
      };
    } catch (error) {
      logger.error('Demo BSE Star MF scheme master data fetch failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getSchemeDetails(schemeCode) {
    try {
      const scheme = this.schemes.get(schemeCode);
      if (!scheme) {
        return {
          success: false,
          error: 'Scheme not found'
        };
      }

      logger.info('Demo BSE Star MF scheme details fetched', { schemeCode });

      return {
        success: true,
        data: scheme
      };
    } catch (error) {
      logger.error('Demo BSE Star MF scheme details fetch failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * 3. Lumpsum Order Placement API
   */
  async placeLumpsumOrder(orderData) {
    try {
      const orderId = `ORD_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const bseOrderId = `BSE_ORD_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      const scheme = this.schemes.get(orderData.schemeCode);
      if (!scheme) {
        return {
          success: false,
          error: 'Scheme not found'
        };
      }

      const estimatedUnits = orderData.amount / scheme.nav;

      const order = {
        orderId,
        bseOrderId,
        clientId: orderData.clientId,
        schemeCode: orderData.schemeCode,
        schemeName: scheme.schemeName,
        amount: orderData.amount,
        estimatedUnits,
        nav: scheme.nav,
        orderType: 'PURCHASE',
        orderCategory: 'LUMPSUM',
        status: 'PENDING',
        orderDate: new Date().toISOString(),
        paymentMode: orderData.paymentMode || 'ONLINE',
        isSmartSIP: orderData.isSmartSIP || false
      };

      this.orders.set(orderId, order);

      // Simulate order processing
      setTimeout(() => {
        order.status = 'COMPLETED';
        order.completionDate = new Date().toISOString();
        order.units = estimatedUnits;
        this.orders.set(orderId, order);
      }, 5000);

      logger.info('Demo BSE Star MF lumpsum order placed', { orderId, bseOrderId });

      return {
        success: true,
        data: {
          orderId,
          bseOrderId,
          status: 'PENDING',
          orderDate: order.orderDate,
          estimatedUnits,
          nav: scheme.nav,
          message: 'Order placed successfully'
        }
      };
    } catch (error) {
      logger.error('Demo BSE Star MF lumpsum order placement failed', { error: error.message });
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
      const order = this.orders.get(orderId);
      if (!order) {
        return {
          success: false,
          error: 'Order not found'
        };
      }

      logger.info('Demo BSE Star MF order status fetched', { orderId, status: order.status });

      return {
        success: true,
        data: {
          orderId: order.orderId,
          bseOrderId: order.bseOrderId,
          status: order.status,
          orderDate: order.orderDate,
          completionDate: order.completionDate,
          amount: order.amount,
          units: order.units,
          nav: order.nav,
          charges: {
            transactionCharge: 0,
            gst: 0,
            totalCharges: 0
          },
          message: 'Order status retrieved successfully'
        }
      };
    } catch (error) {
      logger.error('Demo BSE Star MF order status fetch failed', { error: error.message });
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
      const redemptionId = `RED_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const bseRedemptionId = `BSE_RED_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      const scheme = this.schemes.get(redemptionData.schemeCode);
      if (!scheme) {
        return {
          success: false,
          error: 'Scheme not found'
        };
      }

      const estimatedAmount = redemptionData.redemptionType === 'UNITS' 
        ? redemptionData.units * scheme.nav 
        : redemptionData.amount;

      const redemption = {
        redemptionId,
        bseRedemptionId,
        clientId: redemptionData.clientId,
        schemeCode: redemptionData.schemeCode,
        schemeName: scheme.schemeName,
        redemptionType: redemptionData.redemptionType,
        units: redemptionData.units || 0,
        amount: redemptionData.amount || 0,
        estimatedAmount,
        nav: scheme.nav,
        status: 'PENDING',
        redemptionDate: new Date().toISOString()
      };

      this.orders.set(redemptionId, redemption);

      // Simulate redemption processing
      setTimeout(() => {
        redemption.status = 'COMPLETED';
        redemption.completionDate = new Date().toISOString();
        this.orders.set(redemptionId, redemption);
      }, 3000);

      logger.info('Demo BSE Star MF redemption order placed', { redemptionId, bseRedemptionId });

      return {
        success: true,
        data: {
          redemptionId,
          bseRedemptionId,
          status: 'PENDING',
          redemptionDate: redemption.redemptionDate,
          units: redemption.units,
          estimatedAmount,
          nav: scheme.nav,
          message: 'Redemption order placed successfully'
        }
      };
    } catch (error) {
      logger.error('Demo BSE Star MF redemption order placement failed', { error: error.message });
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
      let transactions = Array.from(this.orders.values()).filter(order => 
        order.orderType === 'PURCHASE' || order.orderType === 'REDEMPTION'
      );

      // Apply filters
      if (filters.clientId) {
        transactions = transactions.filter(txn => txn.clientId === filters.clientId);
      }
      if (filters.schemeCode) {
        transactions = transactions.filter(txn => txn.schemeCode === filters.schemeCode);
      }
      if (filters.startDate) {
        transactions = transactions.filter(txn => new Date(txn.orderDate) >= new Date(filters.startDate));
      }
      if (filters.endDate) {
        transactions = transactions.filter(txn => new Date(txn.orderDate) <= new Date(filters.endDate));
      }

      // Apply pagination
      const limit = filters.limit || 20;
      const offset = filters.offset || 0;
      const paginatedTransactions = transactions.slice(offset, offset + limit);

      logger.info('Demo BSE Star MF transaction report fetched', { 
        totalTransactions: transactions.length,
        returnedTransactions: paginatedTransactions.length 
      });

      return {
        success: true,
        data: {
          transactions: paginatedTransactions,
          totalCount: transactions.length,
          pagination: {
            page: Math.floor(offset / limit) + 1,
            limit,
            total: transactions.length,
            pages: Math.ceil(transactions.length / limit)
          }
        }
      };
    } catch (error) {
      logger.error('Demo BSE Star MF transaction report fetch failed', { error: error.message });
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
      const clientId = filters.clientId;
      const holdings = this.generateDemoHoldings(clientId);

      logger.info('Demo BSE Star MF NAV and holding report fetched', { clientId });

      return {
        success: true,
        data: {
          holdings,
          totalValue: holdings.reduce((sum, holding) => sum + holding.currentValue, 0),
          totalUnits: holdings.reduce((sum, holding) => sum + holding.units, 0),
          reportDate: new Date().toISOString(),
          navData: Object.fromEntries(
            Array.from(this.schemes.values()).map(scheme => [
              scheme.schemeCode,
              { nav: scheme.nav, navDate: scheme.navDate }
            ])
          )
        }
      };
    } catch (error) {
      logger.error('Demo BSE Star MF NAV and holding report fetch failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Generate demo holdings for a client
   */
  generateDemoHoldings(clientId) {
    const holdings = [];
    const schemes = Array.from(this.schemes.values());

    // Generate random holdings for the client
    schemes.forEach(scheme => {
      if (Math.random() > 0.5) { // 50% chance of having holding
        const units = Math.floor(Math.random() * 1000) + 100;
        const currentValue = units * scheme.nav;
        const purchaseValue = currentValue * (0.9 + Math.random() * 0.2); // Random purchase value

        holdings.push({
          schemeCode: scheme.schemeCode,
          schemeName: scheme.schemeName,
          fundHouse: scheme.fundHouse,
          category: scheme.category,
          units,
          currentValue,
          purchaseValue,
          absoluteReturn: currentValue - purchaseValue,
          absoluteReturnPercent: ((currentValue - purchaseValue) / purchaseValue) * 100,
          nav: scheme.nav,
          navDate: scheme.navDate
        });
      }
    });

    return holdings;
  }

  async getCurrentNAV(schemeCodes) {
    try {
      const codes = Array.isArray(schemeCodes) ? schemeCodes : [schemeCodes];
      const navData = {};

      codes.forEach(code => {
        const scheme = this.schemes.get(code);
        if (scheme) {
          navData[code] = {
            nav: scheme.nav,
            navDate: scheme.navDate
          };
        }
      });

      logger.info('Demo BSE Star MF current NAV fetched', { schemeCodes: codes });

      return {
        success: true,
        data: {
          navData,
          lastUpdated: new Date().toISOString()
        }
      };
    } catch (error) {
      logger.error('Demo BSE Star MF current NAV fetch failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * 8. eMandate via BSE
   */
  async setupEMandate(mandateData) {
    try {
      const mandateId = `MANDATE_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const bseMandateId = `BSE_MANDATE_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      const mandate = {
        mandateId,
        bseMandateId,
        clientId: mandateData.clientId,
        bankAccount: mandateData.bankAccount,
        amount: mandateData.amount,
        frequency: mandateData.frequency,
        startDate: mandateData.startDate,
        endDate: mandateData.endDate,
        status: 'PENDING',
        createdAt: new Date().toISOString()
      };

      this.mandates.set(mandateId, mandate);

      // Simulate mandate activation
      setTimeout(() => {
        mandate.status = 'ACTIVE';
        mandate.activatedAt = new Date().toISOString();
        this.mandates.set(mandateId, mandate);
      }, 10000);

      logger.info('Demo BSE Star MF eMandate setup', { mandateId, bseMandateId });

      return {
        success: true,
        data: {
          mandateId,
          bseMandateId,
          status: 'PENDING',
          mandateUrl: `https://demo.bseindia.com/mandate/${mandateId}`,
          expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
          message: 'eMandate setup initiated successfully'
        }
      };
    } catch (error) {
      logger.error('Demo BSE Star MF eMandate setup failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getEMandateStatus(mandateId) {
    try {
      const mandate = this.mandates.get(mandateId);
      if (!mandate) {
        return {
          success: false,
          error: 'Mandate not found'
        };
      }

      logger.info('Demo BSE Star MF eMandate status fetched', { mandateId, status: mandate.status });

      return {
        success: true,
        data: {
          mandateId: mandate.mandateId,
          status: mandate.status,
          bankAccount: mandate.bankAccount,
          amount: mandate.amount,
          frequency: mandate.frequency,
          startDate: mandate.startDate,
          endDate: mandate.endDate,
          createdAt: mandate.createdAt,
          activatedAt: mandate.activatedAt,
          message: 'Mandate status retrieved successfully'
        }
      };
    } catch (error) {
      logger.error('Demo BSE Star MF eMandate status fetch failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async cancelEMandate(mandateId, reason = 'USER_REQUESTED') {
    try {
      const mandate = this.mandates.get(mandateId);
      if (!mandate) {
        return {
          success: false,
          error: 'Mandate not found'
        };
      }

      mandate.status = 'CANCELLED';
      mandate.cancelledAt = new Date().toISOString();
      mandate.cancellationReason = reason;
      this.mandates.set(mandateId, mandate);

      logger.info('Demo BSE Star MF eMandate cancelled', { mandateId, reason });

      return {
        success: true,
        data: {
          mandateId: mandate.mandateId,
          status: 'CANCELLED',
          cancellationDate: mandate.cancelledAt,
          message: 'eMandate cancelled successfully'
        }
      };
    } catch (error) {
      logger.error('Demo BSE Star MF eMandate cancellation failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Additional helper methods
   */
  async getClientFolios(clientId) {
    try {
      const holdings = this.generateDemoHoldings(clientId);
      const folios = holdings.map(holding => ({
        folioNumber: `FOLIO_${clientId}_${holding.schemeCode}`,
        schemeCode: holding.schemeCode,
        schemeName: holding.schemeName,
        units: holding.units,
        currentValue: holding.currentValue
      }));

      logger.info('Demo BSE Star MF client folios fetched', { clientId });

      return {
        success: true,
        data: {
          folios,
          totalFolios: folios.length
        }
      };
    } catch (error) {
      logger.error('Demo BSE Star MF client folios fetch failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getSchemePerformance(schemeCode, period = '1Y') {
    try {
      const scheme = this.schemes.get(schemeCode);
      if (!scheme) {
        return {
          success: false,
          error: 'Scheme not found'
        };
      }

      // Generate demo performance data
      const performance = {
        '1M': { return: 2.5, benchmark: 2.1 },
        '3M': { return: 7.2, benchmark: 6.8 },
        '6M': { return: 12.5, benchmark: 11.9 },
        '1Y': { return: 18.3, benchmark: 17.2 },
        '3Y': { return: 45.6, benchmark: 42.1 },
        '5Y': { return: 78.9, benchmark: 72.3 }
      };

      const riskMetrics = {
        volatility: 15.2,
        sharpeRatio: 0.85,
        beta: 0.95,
        alpha: 2.1,
        maxDrawdown: -12.5
      };

      logger.info('Demo BSE Star MF scheme performance fetched', { schemeCode, period });

      return {
        success: true,
        data: {
          schemeCode,
          period,
          performance: performance[period] || performance['1Y'],
          allPeriods: performance,
          riskMetrics
        }
      };
    } catch (error) {
      logger.error('Demo BSE Star MF scheme performance fetch failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async healthCheck() {
    try {
      logger.info('Demo BSE Star MF health check');

      return {
        success: true,
        data: {
          status: 'HEALTHY',
          timestamp: new Date().toISOString(),
          version: '1.0.0',
          services: {
            clientManagement: 'ACTIVE',
            orderManagement: 'ACTIVE',
            schemeData: 'ACTIVE',
            mandateManagement: 'ACTIVE'
          }
        }
      };
    } catch (error) {
      logger.error('Demo BSE Star MF health check failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }
}

module.exports = new DemoBSEStarMFService(); 