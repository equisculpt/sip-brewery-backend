const axios = require('axios');
const logger = require('../utils/logger');

class NSEService {
  constructor() {
    this.baseURL = 'https://www.nseindia.com/api';
    this.headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
      'Accept': 'application/json, text/plain, */*',
      'Accept-Language': 'en-US,en;q=0.9',
      'Accept-Encoding': 'gzip, deflate, br',
      'Connection': 'keep-alive',
      'Upgrade-Insecure-Requests': '1'
    };
    this.session = null;
  }

  async initialize() {
    try {
      // Create a session to maintain cookies
      this.session = axios.create({
        baseURL: this.baseURL,
        headers: this.headers,
        timeout: 10000
      });

      // Get initial session cookies
      await this.session.get('/');
      logger.info('NSE Service initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize NSE Service:', error.message);
      throw error;
    }
  }

  async getMarketStatus() {
    try {
      const response = await this.session.get('/marketStatus');
      return response.data;
    } catch (error) {
      logger.error('Failed to get market status:', error.message);
      return {
        marketState: 'CLOSED',
        marketStatus: 'Market data unavailable',
        timestamp: new Date().toISOString()
      };
    }
  }

  async getEquityStockIndices() {
    try {
      const response = await this.session.get('/allIndices');
      return response.data;
    } catch (error) {
      logger.error('Failed to get equity stock indices:', error.message);
      return {
        indices: [],
        timestamp: new Date().toISOString()
      };
    }
  }

  async getNifty50Data() {
    try {
      const response = await this.session.get('/equity-stockIndices?index=NIFTY%2050');
      return response.data;
    } catch (error) {
      logger.error('Failed to get Nifty 50 data:', error.message);
      return {
        name: 'NIFTY 50',
        lastPrice: 0,
        change: 0,
        pChange: 0,
        timestamp: new Date().toISOString()
      };
    }
  }

  async getBankNiftyData() {
    try {
      const response = await this.session.get('/equity-stockIndices?index=NIFTY%20BANK');
      return response.data;
    } catch (error) {
      logger.error('Failed to get Bank Nifty data:', error.message);
      return {
        name: 'NIFTY BANK',
        lastPrice: 0,
        change: 0,
        pChange: 0,
        timestamp: new Date().toISOString()
      };
    }
  }

  async getStockData(symbol) {
    try {
      const response = await this.session.get(`/quote-equity?symbol=${symbol}`);
      return response.data;
    } catch (error) {
      logger.error(`Failed to get stock data for ${symbol}:`, error.message);
      return {
        symbol,
        lastPrice: 0,
        change: 0,
        pChange: 0,
        timestamp: new Date().toISOString()
      };
    }
  }

  async getGainersAndLosers() {
    try {
      const response = await this.session.get('/equity-stockIndices?index=NIFTY%2050');
      const data = response.data;
      
      // Simulate gainers and losers based on available data
      return {
        gainers: data.gainers || [],
        losers: data.losers || [],
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      logger.error('Failed to get gainers and losers:', error.message);
      return {
        gainers: [],
        losers: [],
        timestamp: new Date().toISOString()
      };
    }
  }

  async getMostActiveEquities() {
    try {
      const response = await this.session.get('/equity-stockIndices?index=NIFTY%2050');
      const data = response.data;
      
      // Simulate most active equities based on available data
      return {
        mostActive: data.mostActive || [],
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      logger.error('Failed to get most active equities:', error.message);
      return {
        mostActive: [],
        timestamp: new Date().toISOString()
      };
    }
  }

  async getMarketMood() {
    try {
      const niftyData = await this.getNifty50Data();
      const bankNiftyData = await this.getBankNiftyData();
      
      const niftyChange = niftyData.pChange || 0;
      const bankNiftyChange = bankNiftyData.pChange || 0;
      
      let mood = 'NEUTRAL';
      if (niftyChange > 1 && bankNiftyChange > 1) {
        mood = 'BULLISH';
      } else if (niftyChange < -1 && bankNiftyChange < -1) {
        mood = 'BEARISH';
      }
      
      return {
        mood,
        niftyChange,
        bankNiftyChange,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      logger.error('Failed to get market mood:', error.message);
      return {
        mood: 'NEUTRAL',
        niftyChange: 0,
        bankNiftyChange: 0,
        timestamp: new Date().toISOString()
      };
    }
  }

  getStatus() {
    return {
      service: 'NSE Service',
      status: this.session ? 'ACTIVE' : 'INACTIVE',
      timestamp: new Date().toISOString()
    };
  }
}

module.exports = new NSEService(); 