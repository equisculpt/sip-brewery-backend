const WebSocket = require('ws');
const axios = require('axios');
const EventEmitter = require('events');
const logger = require('../utils/logger');

class RealTimeDataService extends EventEmitter {
  constructor() {
    super();
    this.wss = null;
    this.clients = new Map();
    this.navCache = new Map();
    this.marketDataCache = new Map();
    this.updateInterval = null;
    this.isRunning = false;
  }

  /**
   * Initialize WebSocket server
   */
  initialize(server) {
    this.wss = new WebSocket.Server({ server });
    
    this.wss.on('connection', (ws, req) => {
      this.handleConnection(ws, req);
    });

    logger.info('Real-time data service initialized');
  }

  /**
   * Handle new WebSocket connection
   */
  handleConnection(ws, req) {
    const clientId = this.generateClientId();
    this.clients.set(clientId, {
      ws,
      subscriptions: new Set(),
      lastPing: Date.now()
    });

    logger.info(`New WebSocket connection: ${clientId}`);

    // Send welcome message
    ws.send(JSON.stringify({
      type: 'connection',
      clientId,
      message: 'Connected to SIP Brewery Real-time Data Service'
    }));

    // Handle incoming messages
    ws.on('message', (data) => {
      try {
        const message = JSON.parse(data);
        this.handleMessage(clientId, message);
      } catch (error) {
        logger.error('Invalid message format:', error);
      }
    });

    // Handle client disconnect
    ws.on('close', () => {
      this.clients.delete(clientId);
      logger.info(`Client disconnected: ${clientId}`);
    });

    // Handle ping/pong for connection health
    ws.on('pong', () => {
      const client = this.clients.get(clientId);
      if (client) {
        client.lastPing = Date.now();
      }
    });
  }

  /**
   * Handle incoming WebSocket messages
   */
  handleMessage(clientId, message) {
    const client = this.clients.get(clientId);
    if (!client) return;

    switch (message.type) {
      case 'subscribe_nav':
        this.subscribeToNAV(clientId, message.schemeCodes);
        break;
      
      case 'subscribe_market':
        this.subscribeToMarketData(clientId, message.indices);
        break;
      
      case 'subscribe_portfolio':
        this.subscribeToPortfolio(clientId, message.portfolioId);
        break;
      
      case 'unsubscribe':
        this.unsubscribe(clientId, message.channel);
        break;
      
      case 'ping':
        client.ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
        break;
      
      default:
        logger.warn(`Unknown message type: ${message.type}`);
    }
  }

  /**
   * Subscribe to NAV updates
   */
  subscribeToNAV(clientId, schemeCodes) {
    const client = this.clients.get(clientId);
    if (!client) return;

    schemeCodes.forEach(code => {
      client.subscriptions.add(`nav_${code}`);
    });

    // Send current NAV data immediately
    schemeCodes.forEach(code => {
      const navData = this.navCache.get(code);
      if (navData) {
        client.ws.send(JSON.stringify({
          type: 'nav_update',
          schemeCode: code,
          data: navData
        }));
      }
    });

    logger.info(`Client ${clientId} subscribed to NAV updates for: ${schemeCodes.join(', ')}`);
  }

  /**
   * Subscribe to market data
   */
  subscribeToMarketData(clientId, indices) {
    const client = this.clients.get(clientId);
    if (!client) return;

    indices.forEach(index => {
      client.subscriptions.add(`market_${index}`);
    });

    // Send current market data immediately
    indices.forEach(index => {
      const marketData = this.marketDataCache.get(index);
      if (marketData) {
        client.ws.send(JSON.stringify({
          type: 'market_update',
          index,
          data: marketData
        }));
      }
    });

    logger.info(`Client ${clientId} subscribed to market data for: ${indices.join(', ')}`);
  }

  /**
   * Subscribe to portfolio updates
   */
  subscribeToPortfolio(clientId, portfolioId) {
    const client = this.clients.get(clientId);
    if (!client) return;

    client.subscriptions.add(`portfolio_${portfolioId}`);
    logger.info(`Client ${clientId} subscribed to portfolio updates: ${portfolioId}`);
  }

  /**
   * Unsubscribe from channel
   */
  unsubscribe(clientId, channel) {
    const client = this.clients.get(clientId);
    if (!client) return;

    client.subscriptions.delete(channel);
    logger.info(`Client ${clientId} unsubscribed from: ${channel}`);
  }

  /**
   * Start real-time data updates
   */
  async start() {
    if (this.isRunning) return;

    this.isRunning = true;
    
    // Update NAV data every 15 seconds
    this.updateInterval = setInterval(async () => {
      await this.updateNAVData();
      await this.updateMarketData();
      this.broadcastUpdates();
    }, 15000);

    // Initial data fetch
    await this.updateNAVData();
    await this.updateMarketData();

    logger.info('Real-time data service started');
  }

  /**
   * Stop real-time data updates
   */
  stop() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
    this.isRunning = false;
    logger.info('Real-time data service stopped');
  }

  /**
   * Update NAV data for all tracked funds
   */
  async updateNAVData() {
    try {
      // Get list of funds to track (from active subscriptions)
      const trackedFunds = this.getTrackedFunds();
      
      for (const schemeCode of trackedFunds) {
        try {
          const navData = await this.fetchNAVData(schemeCode);
          if (navData) {
            this.navCache.set(schemeCode, navData);
            this.emit('nav_updated', schemeCode, navData);
          }
        } catch (error) {
          logger.error(`Failed to fetch NAV for ${schemeCode}:`, error);
        }
      }
    } catch (error) {
      logger.error('Failed to update NAV data:', error);
    }
  }

  /**
   * Update market data
   */
  async updateMarketData() {
    try {
      const indices = ['NIFTY50', 'SENSEX', 'NIFTYBANK', 'NIFTYIT'];
      
      for (const index of indices) {
        try {
          const marketData = await this.fetchMarketData(index);
          if (marketData) {
            this.marketDataCache.set(index, marketData);
            this.emit('market_updated', index, marketData);
          }
        } catch (error) {
          logger.error(`Failed to fetch market data for ${index}:`, error);
        }
      }
    } catch (error) {
      logger.error('Failed to update market data:', error);
    }
  }

  /**
   * Fetch NAV data from API
   */
  async fetchNAVData(schemeCode) {
    try {
      const response = await axios.get(`https://api.mfapi.in/mf/${schemeCode}`);
      const data = response.data;
      
      if (data && data.data && data.data.length > 0) {
        const latest = data.data[0];
        return {
          schemeCode,
          schemeName: data.meta?.scheme_name || 'Unknown',
          nav: parseFloat(latest.nav),
          date: latest.date,
          timestamp: Date.now(),
          change: this.calculateNAVChange(data.data),
          changePercent: this.calculateNAVChangePercent(data.data)
        };
      }
    } catch (error) {
      logger.error(`Error fetching NAV for ${schemeCode}:`, error);
    }
    return null;
  }

  /**
   * Fetch market data
   */
  async fetchMarketData(index) {
    try {
      // For now, use mock data. In production, integrate with real market data APIs
      const mockData = {
        NIFTY50: { value: 22000 + Math.random() * 500, change: Math.random() * 100 - 50 },
        SENSEX: { value: 72000 + Math.random() * 1000, change: Math.random() * 200 - 100 },
        NIFTYBANK: { value: 48000 + Math.random() * 800, change: Math.random() * 150 - 75 },
        NIFTYIT: { value: 35000 + Math.random() * 600, change: Math.random() * 120 - 60 }
      };

      const data = mockData[index];
      if (data) {
        return {
          index,
          value: data.value,
          change: data.change,
          changePercent: (data.change / (data.value - data.change)) * 100,
          timestamp: Date.now(),
          volume: Math.random() * 1000000
        };
      }
    } catch (error) {
      logger.error(`Error fetching market data for ${index}:`, error);
    }
    return null;
  }

  /**
   * Calculate NAV change
   */
  calculateNAVChange(navData) {
    if (navData.length < 2) return 0;
    return parseFloat(navData[0].nav) - parseFloat(navData[1].nav);
  }

  /**
   * Calculate NAV change percentage
   */
  calculateNAVChangePercent(navData) {
    if (navData.length < 2) return 0;
    const current = parseFloat(navData[0].nav);
    const previous = parseFloat(navData[1].nav);
    return ((current - previous) / previous) * 100;
  }

  /**
   * Get list of funds being tracked
   */
  getTrackedFunds() {
    const funds = new Set();
    
    for (const client of this.clients.values()) {
      for (const subscription of client.subscriptions) {
        if (subscription.startsWith('nav_')) {
          funds.add(subscription.replace('nav_', ''));
        }
      }
    }
    
    return Array.from(funds);
  }

  /**
   * Broadcast updates to subscribed clients
   */
  broadcastUpdates() {
    for (const [clientId, client] of this.clients) {
      // Check connection health
      if (Date.now() - client.lastPing > 30000) {
        logger.warn(`Client ${clientId} connection stale, removing`);
        this.clients.delete(clientId);
        continue;
      }

      // Send NAV updates
      for (const subscription of client.subscriptions) {
        if (subscription.startsWith('nav_')) {
          const schemeCode = subscription.replace('nav_', '');
          const navData = this.navCache.get(schemeCode);
          if (navData) {
            this.sendToClient(client, {
              type: 'nav_update',
              schemeCode,
              data: navData
            });
          }
        }
        
        if (subscription.startsWith('market_')) {
          const index = subscription.replace('market_', '');
          const marketData = this.marketDataCache.get(index);
          if (marketData) {
            this.sendToClient(client, {
              type: 'market_update',
              index,
              data: marketData
            });
          }
        }
      }
    }
  }

  /**
   * Send message to specific client
   */
  sendToClient(client, message) {
    if (client.ws.readyState === WebSocket.OPEN) {
      try {
        client.ws.send(JSON.stringify(message));
      } catch (error) {
        logger.error('Failed to send message to client:', error);
      }
    }
  }

  /**
   * Generate unique client ID
   */
  generateClientId() {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get service status
   */
  getStatus() {
    return {
      isRunning: this.isRunning,
      connectedClients: this.clients.size,
      trackedFunds: this.getTrackedFunds().length,
      navCacheSize: this.navCache.size,
      marketDataCacheSize: this.marketDataCache.size
    };
  }
}

module.exports = RealTimeDataService; 