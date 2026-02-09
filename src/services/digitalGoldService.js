const axios = require('axios');
const logger = require('../utils/logger');
const redis = require('../config/redis');

/**
 * Digital Gold Investment Service
 * Integrates with SafeGold/Augmont for digital gold transactions
 * 
 * Features:
 * - Buy/Sell digital gold (24K, 22K, 18K)
 * - Gold SIP (systematic gold purchase)
 * - Real-time gold price tracking
 * - Gold portfolio management
 * - Gold delivery options
 * - Gold-backed loans
 */

class DigitalGoldService {
  constructor() {
    this.provider = process.env.GOLD_PROVIDER || 'safegold'; // safegold, augmont, mmtc
    this.apiKey = process.env.GOLD_API_KEY;
    this.apiSecret = process.env.GOLD_API_SECRET;
    this.baseURL = this.getProviderURL();
    
    // Gold types
    this.goldTypes = {
      '24K': { purity: 0.999, name: '24 Karat Gold' },
      '22K': { purity: 0.916, name: '22 Karat Gold' },
      '18K': { purity: 0.750, name: '18 Karat Gold' }
    };
  }

  getProviderURL() {
    const urls = {
      safegold: 'https://api.safegold.com/v1',
      augmont: 'https://api.augmont.com/v1',
      mmtc: 'https://api.mmtcpamp.com/v1'
    };
    return urls[this.provider] || urls.safegold;
  }

  /**
   * Get current gold price (live)
   */
  async getCurrentGoldPrice(goldType = '24K') {
    try {
      // Check cache first (1 minute TTL)
      const cacheKey = `gold:price:${goldType}`;
      const cached = await redis.get(cacheKey);
      
      if (cached) {
        logger.debug(`Gold price cache hit for ${goldType}`);
        return JSON.parse(cached);
      }

      // Fetch from provider
      const response = await axios.get(`${this.baseURL}/price`, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        },
        params: {
          metal: 'gold',
          purity: this.goldTypes[goldType].purity
        }
      });

      const priceData = {
        goldType,
        purity: this.goldTypes[goldType].purity,
        buyPrice: response.data.buy_price, // Price per gram
        sellPrice: response.data.sell_price,
        currency: 'INR',
        unit: 'gram',
        timestamp: new Date().toISOString(),
        provider: this.provider,
        // Additional metrics
        change24h: response.data.change_24h || 0,
        changePercent24h: response.data.change_percent_24h || 0,
        high24h: response.data.high_24h || response.data.buy_price,
        low24h: response.data.low_24h || response.data.sell_price
      };

      // Cache for 1 minute
      await redis.setex(cacheKey, 60, JSON.stringify(priceData));

      logger.info(`Fetched gold price for ${goldType}: ₹${priceData.buyPrice}/gram`);

      return priceData;
    } catch (error) {
      logger.error('Failed to fetch gold price', { error: error.message, goldType });
      
      // Return fallback price (mock data for demo)
      return {
        goldType,
        purity: this.goldTypes[goldType].purity,
        buyPrice: 6500, // ₹6,500 per gram (approximate)
        sellPrice: 6450,
        currency: 'INR',
        unit: 'gram',
        timestamp: new Date().toISOString(),
        provider: 'fallback',
        change24h: 50,
        changePercent24h: 0.77,
        high24h: 6550,
        low24h: 6400
      };
    }
  }

  /**
   * Buy digital gold
   */
  async buyGold(userId, amount, goldType = '24K', paymentMethod = 'UPI') {
    try {
      logger.info(`User ${userId} buying gold`, { amount, goldType });

      // Get current price
      const priceData = await this.getCurrentGoldPrice(goldType);
      const quantityGrams = amount / priceData.buyPrice;

      // Create order with provider
      const orderResponse = await axios.post(`${this.baseURL}/orders/buy`, {
        user_id: userId,
        amount: amount,
        quantity: quantityGrams,
        metal: 'gold',
        purity: this.goldTypes[goldType].purity,
        payment_method: paymentMethod
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      const order = {
        orderId: orderResponse.data.order_id || `GOLD_BUY_${Date.now()}`,
        userId,
        type: 'BUY',
        goldType,
        amount,
        quantityGrams,
        pricePerGram: priceData.buyPrice,
        status: 'PENDING',
        paymentMethod,
        paymentStatus: 'PENDING',
        createdAt: new Date(),
        provider: this.provider
      };

      // Save to database (would be MongoDB in production)
      await this.saveGoldOrder(order);

      logger.info(`Gold buy order created: ${order.orderId}`);

      return {
        success: true,
        order,
        paymentUrl: orderResponse.data.payment_url,
        message: 'Gold purchase initiated successfully'
      };
    } catch (error) {
      logger.error('Gold purchase failed', { error: error.message, userId, amount });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Sell digital gold
   */
  async sellGold(userId, quantityGrams, goldType = '24K') {
    try {
      logger.info(`User ${userId} selling gold`, { quantityGrams, goldType });

      // Check user's gold holdings
      const holdings = await this.getUserGoldHoldings(userId);
      const availableGold = holdings.find(h => h.goldType === goldType);

      if (!availableGold || availableGold.quantity < quantityGrams) {
        return {
          success: false,
          error: 'Insufficient gold holdings'
        };
      }

      // Get current price
      const priceData = await this.getCurrentGoldPrice(goldType);
      const saleAmount = quantityGrams * priceData.sellPrice;

      // Create sell order with provider
      const orderResponse = await axios.post(`${this.baseURL}/orders/sell`, {
        user_id: userId,
        quantity: quantityGrams,
        metal: 'gold',
        purity: this.goldTypes[goldType].purity
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      const order = {
        orderId: orderResponse.data.order_id || `GOLD_SELL_${Date.now()}`,
        userId,
        type: 'SELL',
        goldType,
        quantityGrams,
        amount: saleAmount,
        pricePerGram: priceData.sellPrice,
        status: 'PENDING',
        createdAt: new Date(),
        provider: this.provider
      };

      await this.saveGoldOrder(order);

      logger.info(`Gold sell order created: ${order.orderId}`);

      return {
        success: true,
        order,
        message: 'Gold sale initiated successfully',
        creditAmount: saleAmount
      };
    } catch (error) {
      logger.error('Gold sale failed', { error: error.message, userId, quantityGrams });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Create Gold SIP (Systematic Investment Plan)
   */
  async createGoldSIP(userId, amount, frequency = 'MONTHLY', goldType = '24K', startDate = null) {
    try {
      logger.info(`Creating Gold SIP for user ${userId}`, { amount, frequency });

      const sip = {
        sipId: `GOLD_SIP_${Date.now()}`,
        userId,
        amount,
        frequency, // DAILY, WEEKLY, MONTHLY
        goldType,
        status: 'ACTIVE',
        startDate: startDate || new Date(),
        nextExecutionDate: this.calculateNextExecutionDate(frequency, startDate),
        totalInvested: 0,
        totalGoldAccumulated: 0,
        executionCount: 0,
        createdAt: new Date()
      };

      // Register SIP with provider
      await axios.post(`${this.baseURL}/sip/create`, {
        user_id: userId,
        amount,
        frequency: frequency.toLowerCase(),
        metal: 'gold',
        purity: this.goldTypes[goldType].purity
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      // Save to database
      await this.saveGoldSIP(sip);

      logger.info(`Gold SIP created: ${sip.sipId}`);

      return {
        success: true,
        sip,
        message: 'Gold SIP created successfully'
      };
    } catch (error) {
      logger.error('Gold SIP creation failed', { error: error.message, userId });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Execute Gold SIP
   */
  async executeGoldSIP(sipId) {
    try {
      const sip = await this.getGoldSIP(sipId);

      if (!sip || sip.status !== 'ACTIVE') {
        return { success: false, error: 'Invalid or inactive SIP' };
      }

      // Buy gold for SIP amount
      const result = await this.buyGold(sip.userId, sip.amount, sip.goldType, 'AUTO_DEBIT');

      if (result.success) {
        // Update SIP stats
        sip.totalInvested += sip.amount;
        sip.totalGoldAccumulated += result.order.quantityGrams;
        sip.executionCount += 1;
        sip.lastExecutionDate = new Date();
        sip.nextExecutionDate = this.calculateNextExecutionDate(sip.frequency);

        await this.updateGoldSIP(sip);

        logger.info(`Gold SIP executed: ${sipId}`);
      }

      return result;
    } catch (error) {
      logger.error('Gold SIP execution failed', { error: error.message, sipId });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get user's gold holdings
   */
  async getUserGoldHoldings(userId) {
    try {
      // Fetch from provider
      const response = await axios.get(`${this.baseURL}/holdings`, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`
        },
        params: {
          user_id: userId
        }
      });

      const holdings = response.data.holdings.map(h => ({
        goldType: this.getGoldTypeFromPurity(h.purity),
        quantity: h.quantity,
        averageBuyPrice: h.average_price,
        currentValue: h.current_value,
        totalInvested: h.total_invested,
        returns: h.current_value - h.total_invested,
        returnsPercent: ((h.current_value - h.total_invested) / h.total_invested) * 100
      }));

      return holdings;
    } catch (error) {
      logger.error('Failed to fetch gold holdings', { error: error.message, userId });
      
      // Return mock data for demo
      return [
        {
          goldType: '24K',
          quantity: 10.5,
          averageBuyPrice: 6300,
          currentValue: 68250,
          totalInvested: 66150,
          returns: 2100,
          returnsPercent: 3.17
        }
      ];
    }
  }

  /**
   * Get gold portfolio analytics
   */
  async getGoldPortfolioAnalytics(userId) {
    try {
      const holdings = await this.getUserGoldHoldings(userId);
      const sips = await this.getUserGoldSIPs(userId);
      const transactions = await this.getUserGoldTransactions(userId);

      const totalQuantity = holdings.reduce((sum, h) => sum + h.quantity, 0);
      const totalInvested = holdings.reduce((sum, h) => sum + h.totalInvested, 0);
      const currentValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
      const totalReturns = currentValue - totalInvested;
      const returnsPercent = (totalReturns / totalInvested) * 100;

      const activeSIPs = sips.filter(s => s.status === 'ACTIVE').length;
      const monthlyInvestment = sips
        .filter(s => s.status === 'ACTIVE' && s.frequency === 'MONTHLY')
        .reduce((sum, s) => sum + s.amount, 0);

      return {
        summary: {
          totalQuantity,
          totalInvested,
          currentValue,
          totalReturns,
          returnsPercent,
          activeSIPs,
          monthlyInvestment
        },
        holdings,
        sips,
        recentTransactions: transactions.slice(0, 10),
        allocation: this.calculateGoldAllocation(holdings)
      };
    } catch (error) {
      logger.error('Failed to get gold analytics', { error: error.message, userId });
      throw error;
    }
  }

  /**
   * Request physical gold delivery
   */
  async requestGoldDelivery(userId, quantityGrams, goldType = '24K', deliveryAddress) {
    try {
      logger.info(`Gold delivery requested`, { userId, quantityGrams, goldType });

      // Check holdings
      const holdings = await this.getUserGoldHoldings(userId);
      const available = holdings.find(h => h.goldType === goldType);

      if (!available || available.quantity < quantityGrams) {
        return {
          success: false,
          error: 'Insufficient gold for delivery'
        };
      }

      // Minimum delivery quantity (usually 10 grams)
      if (quantityGrams < 10) {
        return {
          success: false,
          error: 'Minimum delivery quantity is 10 grams'
        };
      }

      // Create delivery request
      const deliveryRequest = {
        requestId: `GOLD_DEL_${Date.now()}`,
        userId,
        goldType,
        quantityGrams,
        deliveryAddress,
        status: 'PENDING',
        estimatedDelivery: this.calculateDeliveryDate(),
        deliveryCharges: this.calculateDeliveryCharges(quantityGrams),
        createdAt: new Date()
      };

      // Submit to provider
      await axios.post(`${this.baseURL}/delivery/request`, {
        user_id: userId,
        quantity: quantityGrams,
        purity: this.goldTypes[goldType].purity,
        address: deliveryAddress
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      await this.saveDeliveryRequest(deliveryRequest);

      logger.info(`Gold delivery request created: ${deliveryRequest.requestId}`);

      return {
        success: true,
        deliveryRequest,
        message: 'Gold delivery request submitted successfully'
      };
    } catch (error) {
      logger.error('Gold delivery request failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get gold-backed loan eligibility
   */
  async getGoldLoanEligibility(userId) {
    try {
      const holdings = await this.getUserGoldHoldings(userId);
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);

      // Loan-to-Value ratio (typically 75%)
      const ltvRatio = 0.75;
      const maxLoanAmount = totalValue * ltvRatio;

      // Interest rate (typically 12-15% per annum)
      const interestRate = 12.5;

      return {
        eligible: totalValue > 10000, // Minimum ₹10,000 gold value
        goldValue: totalValue,
        maxLoanAmount,
        ltvRatio: ltvRatio * 100,
        interestRate,
        tenure: [6, 12, 24, 36], // months
        processingFee: maxLoanAmount * 0.01, // 1%
        message: totalValue > 10000 
          ? 'You are eligible for gold-backed loan'
          : 'Minimum ₹10,000 gold value required'
      };
    } catch (error) {
      logger.error('Failed to check loan eligibility', { error: error.message });
      throw error;
    }
  }

  // Helper methods

  calculateNextExecutionDate(frequency, startDate = null) {
    const base = startDate || new Date();
    const next = new Date(base);

    switch (frequency) {
      case 'DAILY':
        next.setDate(next.getDate() + 1);
        break;
      case 'WEEKLY':
        next.setDate(next.getDate() + 7);
        break;
      case 'MONTHLY':
        next.setMonth(next.getMonth() + 1);
        break;
    }

    return next;
  }

  getGoldTypeFromPurity(purity) {
    if (purity >= 0.999) return '24K';
    if (purity >= 0.916) return '22K';
    if (purity >= 0.750) return '18K';
    return '24K';
  }

  calculateGoldAllocation(holdings) {
    const total = holdings.reduce((sum, h) => sum + h.currentValue, 0);
    return holdings.map(h => ({
      goldType: h.goldType,
      percentage: (h.currentValue / total) * 100,
      value: h.currentValue
    }));
  }

  calculateDeliveryDate() {
    const date = new Date();
    date.setDate(date.getDate() + 7); // 7 days delivery
    return date;
  }

  calculateDeliveryCharges(quantityGrams) {
    // ₹500 base + ₹50 per gram
    return 500 + (quantityGrams * 50);
  }

  // Database operations (mock - would use MongoDB in production)

  async saveGoldOrder(order) {
    // Save to MongoDB
    logger.debug('Saving gold order', { orderId: order.orderId });
  }

  async saveGoldSIP(sip) {
    // Save to MongoDB
    logger.debug('Saving gold SIP', { sipId: sip.sipId });
  }

  async updateGoldSIP(sip) {
    // Update in MongoDB
    logger.debug('Updating gold SIP', { sipId: sip.sipId });
  }

  async getGoldSIP(sipId) {
    // Fetch from MongoDB
    return null;
  }

  async getUserGoldSIPs(userId) {
    // Fetch from MongoDB
    return [];
  }

  async getUserGoldTransactions(userId) {
    // Fetch from MongoDB
    return [];
  }

  async saveDeliveryRequest(request) {
    // Save to MongoDB
    logger.debug('Saving delivery request', { requestId: request.requestId });
  }
}

module.exports = new DigitalGoldService();
