const { spawn } = require('child_process');
const axios = require('axios');
const logger = require('../utils/logger');

class NSECliService {
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
      logger.info('NSE CLI Service initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize NSE CLI Service:', error.message);
      throw error;
    }
  }

  async executeCommand(command, args = []) {
    try {
      logger.info(`Executing NSE CLI command: ${command} ${args.join(' ')}`);
      
      // Map CLI commands to API endpoints
      switch (command) {
        case 'get-equity-details':
          return await this.getEquityDetails(args[0]);
        case 'get-index-data':
          return await this.getIndexData(args[0]);
        case 'get-market-status':
          return await this.getMarketStatus();
        case 'get-gainers-losers':
          return await this.getGainersLosers();
        default:
          throw new Error(`Unknown command: ${command}`);
      }
    } catch (error) {
      logger.error(`Error executing NSE CLI command ${command}:`, error);
      throw error;
    }
  }

  async getEquityDetails(symbol) {
    try {
      const response = await this.session.get(`/quote-equity?symbol=${symbol}`);
      return {
        success: true,
        data: response.data,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      logger.error(`Failed to get equity details for ${symbol}:`, error.message);
      return {
        success: false,
        error: error.message,
        data: {
          symbol,
          lastPrice: 0,
          change: 0,
          pChange: 0
        },
        timestamp: new Date().toISOString()
      };
    }
  }

  async getIndexData(indexName) {
    try {
      const response = await this.session.get(`/equity-stockIndices?index=${encodeURIComponent(indexName)}`);
      return {
        success: true,
        data: response.data,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      logger.error(`Failed to get index data for ${indexName}:`, error.message);
      return {
        success: false,
        error: error.message,
        data: {
          name: indexName,
          lastPrice: 0,
          change: 0,
          pChange: 0
        },
        timestamp: new Date().toISOString()
      };
    }
  }

  async getMarketStatus() {
    try {
      const response = await this.session.get('/marketStatus');
      return {
        success: true,
        data: response.data,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      logger.error('Failed to get market status:', error.message);
      return {
        success: false,
        error: error.message,
        data: {
          marketState: 'CLOSED',
          marketStatus: 'Market data unavailable'
        },
        timestamp: new Date().toISOString()
      };
    }
  }

  async getGainersLosers() {
    try {
      const response = await this.session.get('/equity-stockIndices?index=NIFTY%2050');
      const data = response.data;
      
      return {
        success: true,
        data: {
          gainers: data.gainers || [],
          losers: data.losers || []
        },
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      logger.error('Failed to get gainers and losers:', error.message);
      return {
        success: false,
        error: error.message,
        data: {
          gainers: [],
          losers: []
        },
        timestamp: new Date().toISOString()
      };
    }
  }

  getStatus() {
    return {
      service: 'NSE CLI Service',
      status: this.session ? 'ACTIVE' : 'INACTIVE',
      timestamp: new Date().toISOString()
    };
  }
}

module.exports = new NSECliService(); 